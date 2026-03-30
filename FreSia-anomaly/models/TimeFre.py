import torch
import torch.nn as nn
import torch.fft
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal
from einops import rearrange
import torch.nn.functional as F # 导入 functional 模块


class FrequencyEncoder(nn.Module):
    """
    提取频域特征并映射到 Latent Space
    """
    def __init__(self, top_k_freq, d_model):
        super().__init__()
        self.top_k = top_k_freq
        self.projection = nn.Sequential(nn.Linear(2 * self.top_k, d_model//2),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Linear(d_model//2, d_model))

    def forward(self, x):
        x = x.to(torch.float32)
        fft_out = torch.fft.rfft(x, dim=1, norm='ortho')
        
        amp = torch.abs(fft_out)
        phase = torch.angle(fft_out)
        
        topk_amp, topk_indices = torch.topk(amp[:, 1:, :], self.top_k, dim=1)
        
        # gather indices need expansion to match phase dimensions
        topk_indices = topk_indices 
        topk_phase = torch.gather(phase[:, 1:, :], 1, topk_indices)
        
        # [Batch, num_nodes, Top-K*2]
        freq_feats = torch.cat([topk_amp, topk_phase], dim=1).permute(0,2,1)
        
        freq_embedding = self.projection(freq_feats) # B N d_model
        
        return freq_embedding

class InstructionAwareAdapter(nn.Module):
    """
    核心模块：结合频域特征和文本指令，生成 Soft Prompts
    """
    def __init__(self, d_model, prompt_pool_size, pool_dim, dropout):
        super().__init__()
        self.d_model = d_model
        
        self.prompt_pool = nn.Parameter(torch.randn(prompt_pool_size, pool_dim))
        
        self.freq_query_proj = nn.Sequential(nn.Linear(d_model, d_model//2),
                                             nn.ReLU(),
                                             nn.Dropout(dropout),
                                             nn.Linear(d_model//2, pool_dim))
        
        self.gate_net = nn.Sequential(
            nn.Linear(pool_dim+d_model, d_model),
            nn.Sigmoid(),
            # nn.Dropout(dropout)
        )

    def forward(self, freq_embed, text_embed):

        # Query
        q_freq = self.freq_query_proj(freq_embed) # [Batch, nvar, D]

        # q_freq = F.normalize(q_freq, dim=-1)
        # prompt_pool = F.normalize(self.prompt_pool, dim=-1)
        
        # Attention Scores: [Batch, Pool_Size]
        scores = torch.matmul(q_freq, self.prompt_pool.t()) / (self.d_model ** 0.5)
        # scores = torch.matmul(q_freq, prompt_pool.t())*5.0
        attn_weights = torch.softmax(scores, dim=-1) # B N prompt_pool_size
        
        # Retrieved Prompts: [Batch, N, d_model]
        retrieved_prompts = torch.matmul(attn_weights, self.prompt_pool)
        
        text_embed = text_embed.permute(0, 2, 1)
        cat_features = torch.cat([retrieved_prompts, text_embed], dim=-1)  # B N 2*d_model
        
        alpha = self.gate_net(cat_features)
        alpha_smoothed = torch.clamp(alpha, min=0.1, max=0.9)
        
        final_prompt = alpha_smoothed * text_embed + (1 - alpha_smoothed) * retrieved_prompts
    
        return final_prompt


class TimeEncoderWithCLS(nn.Module):
    def __init__(self, patch_len, patch_num, transformer_dim, d_model, head, e_layers, dropout_n):
        super().__init__()
        self.patch_len = patch_len
        self.transformer_dim = transformer_dim
        self.patch_projector = nn.Linear(patch_len, transformer_dim)
        
        # 2. Learnable CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        # 3. Transformer Encoder (Standard)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=head, batch_first=True, dropout=dropout_n)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, e_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.ff_1 = nn.Sequential(
            nn.Linear(transformer_dim * patch_num,  d_model),
        )

    def forward(self, x_patches):
        """
        Input x_patches: [Batch, Num_Nodes, Patch_Num, Patch_Len]
        """
        B, N, P, L = x_patches.shape
        # D = self.d_model
        
        x_reshaped = x_patches.reshape(B * N, P, L)
        
        x_embed = self.patch_projector(x_reshaped)

        cls_token_expanded = self.cls_token.expand(B*N, -1, -1)
        
        enc_input = torch.cat([cls_token_expanded, x_embed], dim=1)
        
        enc_output = self.transformer_encoder(enc_input)
        
        E_Time_CLS = enc_output[:, 0, :].reshape(B, N, -1)
        
        E_Time_Seq = enc_output[:, 1:, :].reshape(B, N, P, -1)
        E_Time_Seq = self.flatten(E_Time_Seq)
        E_Time_Seq = self.ff_1(E_Time_Seq)  # B, N, D
        
        return E_Time_Seq, E_Time_CLS



class SequentialInstructionGatedFusionBlock(nn.Module):
    def __init__(self, transformer_dim, d_model, num_nodes, dropout_n, d_ff):
        super().__init__()
        self.num_nodes = num_nodes
        self.transformer_dim = transformer_dim
        self.d_model = d_model
        self.dropout_n = dropout_n  
        self.d_ff = d_ff
        
        # self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, store_attn=False)

        self.enrichment_K = nn.Sequential(nn.Linear(transformer_dim, d_model), nn.GELU())
        self.enrichment_V = nn.Sequential(nn.Linear(transformer_dim, d_model), nn.GELU())
        
        self.norm = nn.LayerNorm(d_model)
        # self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, E_Time_Seq, soft_prompt, E_Time_CLS):
    
        C_K = self.enrichment_K(E_Time_CLS) # B, N, transformer.dim
        C_V = self.enrichment_V(E_Time_CLS) # B, N, transformer.dim

        
        
        K_enriched = soft_prompt + C_K
        V_enriched = soft_prompt + C_V
        K_enriched = K_enriched.permute(0, 2, 1)  # B d_model N
        V_enriched = V_enriched.permute(0, 2, 1)  # B d_model N

        # --- Step 3: Cross-Attention ---
        # Q = Time (Data), K/V = Prompt (Enriched Knowledge)
        E_Time_Seq = E_Time_Seq.permute(0, 2, 1)  # B d_model N
        fused_out = self.cross_attn(
            E_Time_Seq,   # B N D
            K_enriched,     # B D N
            V_enriched    # B D N
        )
        
        # --- Step 4: Residual + Norm + FFN ---
        fused_out = (E_Time_Seq + fused_out).permute(0, 2, 1)  # B N D
        fused_out = self.norm(fused_out)
        # fused_out = fused_out + self.ffn(fused_out)
        
        return fused_out # [B, N, D]





class SpectraCoTModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_nodes, top_k_freq, d_model, prompt_pool_size, pool_dim, patch_len, stride, padding_patch = 'end',
                 transformer_dim=64, transformer_head=8, e_layer=1, dropout_n=0.1, d_ff=32):
        super().__init__()
        # self.config = config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.top_k_freq = top_k_freq
        self.d_model = d_model
        self.prompt_pool_size = prompt_pool_size
        self.num_nodes = num_nodes
        self.patch_len = patch_len
        self.stride = stride
        self.dropout_n = dropout_n
        self.transformer_dim = transformer_dim
        self.transformer_head = transformer_head
        self.e_layer = e_layer

        self.normalize_layers = Normalize(self.num_nodes, affine=False)
        
        self.freq_encoder = FrequencyEncoder(top_k_freq, d_model)
        self.adapter = InstructionAwareAdapter(d_model, prompt_pool_size, pool_dim, dropout_n)
        
        self.padding_patch = padding_patch
        self.patch_num = int((seq_len - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1


        self.ts_encoder = TimeEncoderWithCLS(patch_len, self.patch_num, transformer_dim, d_model, transformer_head, e_layer, dropout_n)
        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_model, nhead = self.transformer_head, batch_first=True, 
                                                               norm_first = True,dropout = self.dropout_n, layer_norm_eps=1e-4)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer)


        self.cross = SequentialInstructionGatedFusionBlock(transformer_dim, d_model, num_nodes, dropout_n, d_ff)
            
    
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(), 
            nn.Dropout(self.dropout_n),
            nn.Linear(d_model, pred_len)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, text_embed_input):
        # x: [Batch, Seq_Len, num_nodes]
        B, L, N = x.shape

        seg_num = 25
        x_enc = rearrange(x, 'b (n s) m -> b n s m', s=seg_num)
        means = x_enc.mean(2, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=2, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x = rearrange(x_enc, 'b n s m -> b (n s) m')
        
        freq_emb = self.freq_encoder(x)# B N d_model
        # text_embed_input = self.ffn(text_embed_input.permute(0,2,1)).permute(0,2,1)
        
        # 2. 生成 Adapter Prompt
        soft_prompt = self.adapter(freq_emb, text_embed_input) 
        # soft_prompt = F.normalize(soft_prompt, dim=-1)
        # soft_prompt = soft_prompt + 1e-4 * torch.randn_like(soft_prompt)
        soft_prompt = self.prompt_encoder(soft_prompt)  # [B, N, D]
        # soft_prompt = self.prompt_encoder(text_embed_input.permute(0,2,1))  # [B, N, D]
    
        x = x.permute(0, 2, 1) # [B, N, L]

        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
        E_Time_Seq, E_Time_CLS = self.ts_encoder(x)  # E_Time_Seq:B N d_model       E_Time_CLS : B N transformer_dim
        # soft_prompt = torch.clamp(soft_prompt, min=-1e4, max=1e4)
        # E_Time_Seq = torch.clamp(E_Time_Seq, min=-1e4, max=1e4)

        x = self.cross(E_Time_Seq, soft_prompt, E_Time_CLS) #B N d_model
        y = self.output_head(x).permute(0, 2, 1) #B L N

        # denorm
        # y = self.normalize_layers(y, 'denorm')
        dec_out = rearrange(y, 'b (n s) m -> b n s m', s=seg_num)
        dec_out = dec_out * \
                  (stdev[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        dec_out = dec_out + \
                  (means[:, :, 0, :].unsqueeze(2).repeat(
                      1, 1, seg_num, 1))
        y = rearrange(dec_out, 'b n s m -> b (n s) m')

        return y
       