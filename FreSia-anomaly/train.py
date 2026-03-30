import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
from data_provider.data_loader_emb import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,\
    PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
# from models.TimeCMA import Dual
from models.TimeFre import SpectraCoTModel
from utils.metrics import MSE, MAE, metric
import faulthandler
import h5py
import torch.nn.functional as F
import torch.nn as nn
from utils.tools import adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
# from tensorboardX import SummaryWriter
faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:1", help="")
    parser.add_argument("--data", type=str, default="SMD")
    parser.add_argument("--root_path", type=str, default="/dataset/SMD")
    # parser.add_argument("--channel", type=int, default=64, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=38, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=100, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=100, help="out_len")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    # parser.add_argument("--learning_rate", type=float, default=0.0007356633287514458, help="learning rate")
    parser.add_argument("--learning_rate", type=float, default=0.000000000793271, help="stage2 start learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.1, help="dropout rate of neural network layers")
    parser.add_argument("--d_model", type=int, default=768, help="hidden dimensions and llm dimension")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="/gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=50, help="")
    parser.add_argument('--seed', type=int, default=2025, help='random seed')

    #########################################
    parser.add_argument("--top_k_freq", type=int, default=2)
    parser.add_argument("--prompt_pool_size", type=int, default=3)
    parser.add_argument("--pool_dim", type=int, default=768)
    parser.add_argument("--lambda_ortho", type=float, default=0.001685948)
    parser.add_argument("--prompt_dropout_rate", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--padding_patch", type=str, default="end")
    parser.add_argument("--transformer_dim", type=int, default=8)
    parser.add_argument("--transformer_head", type=int, default=1)
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_ff", type=int, default=8)
    parser.add_argument("--anomaly_ratio", type=float, default=0.5)

    #########################################
    parser.add_argument(
        "--es_patience",
        type=int,
        default=10,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    return parser.parse_args()




class trainer:
    def __init__(
        self,
        seq_len,
        pred_len,
        num_nodes,
        top_k_freq,
        d_model,
        prompt_pool_size,
        pool_dim,
        lrate,
        wdecay,
        device,
        epochs,
        lambda_ortho,
        prompt_dropout_rate,
        patch_len,
        stride,
        padding_patch, 
        transformer_dim,
        transformer_head,
        e_layer,
        dropout_n,
        d_ff,
        anomaly_ratio
    ):
        # self.model = Dual(
        #     device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
        #     dropout_n=dropout_n, d_model=d_model, e_layer=e_layer, d_layer=d_layer, head=head
        # )
        # self.config = SpectraCoTConfig()
        self.lambda_ortho = lambda_ortho
        self.prompt_dropout_rate = prompt_dropout_rate
        self.device = device
        self.anomaly_ratio = anomaly_ratio
        self.model = SpectraCoTModel(seq_len, pred_len, num_nodes, top_k_freq, d_model, prompt_pool_size, pool_dim, patch_len, stride, padding_patch, 
                                     transformer_dim, transformer_head, e_layer, dropout_n, d_ff).to(device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 50), eta_min=1e-6)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,           
            patience=10,          
            verbose=True,
            min_lr=1e-6
        )
        # self.loss = MSE
        self.mae_loss = MAE
        self.MAE = MAE
        self.loss = torch.nn.HuberLoss(delta=1.0).to(self.device)
        self.clip = 5
        print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        # print(self.model)



    def train_stage2(self, input, embeddings, real):

        self.model.train()
        self.optimizer.zero_grad()
        B, L, N = input.shape

        generic_embedding_path = "Embeddings/generic_prompt.h5"

        if os.path.exists(generic_embedding_path):
            with h5py.File(generic_embedding_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data) # 1 1 d_model
                generic_embedding = tensor.permute(0, 2, 1)
        else:
            raise FileNotFoundError(f"No generic embedding file found at {generic_embedding_path}") 
        
        generic_embedding = generic_embedding.repeat(B, 1, N).to(self.device)  # B d_model N

        mask_prob = torch.rand(B, 1, N, device=self.device)
        use_specific_mask = (mask_prob > self.prompt_dropout_rate).float()

        input_text_embed = use_specific_mask * embeddings + \
                            (1 - use_specific_mask) * generic_embedding
        predict = self.model(input, input_text_embed)

        loss = self.loss(predict, real)
        # loss = self.mae_loss(predict, real)
        # loss = self.huber_criterion(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.MAE(predict, real)
        return loss.item(), mae.item()


    def train(self, input, embeddings, real):

        B, L, N = input.shape

        generic_embedding_path = "Embeddings/generic_prompt.h5"

        if os.path.exists(generic_embedding_path):
            with h5py.File(generic_embedding_path, 'r') as hf:
                data = hf['embeddings'][:]
                tensor = torch.from_numpy(data) # 1 1 d_model
                generic_embedding = tensor.permute(0, 2, 1)
        else:
            raise FileNotFoundError(f"No generic embedding file found at {generic_embedding_path}") 
        
        generic_embedding = generic_embedding.repeat(B, 1, N).to(self.device)  # B d_model N
        # mse, mae = self.train_stage1(input, real)
        mse, mae = self.train_stage2(input, embeddings, real)
        return mse, mae
        


    def eval_stage2(self, input, embeddings, real_val):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input, embeddings)
        loss = self.loss(predict, real_val)
        mae = self.MAE(predict, real_val)
        return loss.item(), mae.item()

def load_data(args):
    data_map = {
        'ETTh1': Dataset_ETT_hour,
        'ETTh2': Dataset_ETT_hour,
        'ETTm1': Dataset_ETT_minute,
        'ETTm2': Dataset_ETT_minute,
        'PSM': PSMSegLoader,
        'MSL': MSLSegLoader,
        'SMAP': SMAPSegLoader,
        'SMD': SMDSegLoader,
        'SWAT': SWATSegLoader,
        'UEA': UEAloader
    }
    data_class = data_map.get(args.data, Dataset_Custom)
    train_set = data_class(root_path=args.root_path, win_size=args.seq_len, flag='train', data_path=args.data)
    val_set = data_class(root_path=args.root_path, win_size=args.seq_len, flag='val', data_path=args.data)
    test_set = data_class(root_path=args.root_path, win_size=args.seq_len, flag='test', data_path=args.data)

    scaler = train_set.scaler

    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True, num_workers=args.num_workers)

    # return train_loader, val_loader, test_loader, scaler
    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)



def main():
    args = parse_args()
    train_set, val_set, test_set, train_loader, val_loader, test_loader,scaler = load_data(args)

    print()
    seed_it(args.seed)
    device = torch.device(args.device)
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    
    loss = 9999999
    mae_loss = 9999999
    test_log = 999999
    # epochs_since_best_mae = 0
    epochs_since_best_mae = 0

    path = os.path.join(args.save, args.data, 
                        f"{args.pred_len}_{args.top_k_freq}_{args.prompt_pool_size}_{args.lambda_ortho}_{args.prompt_dropout_rate}_{args.learning_rate}_{args.seed}/")
    if not os.path.exists(path):
        os.makedirs(path)
     
    his_loss = []
    val_time = []
    train_time = []
    print(args)

    # writer = SummaryWriter('./run')

    engine = trainer(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_nodes=args.num_nodes,
        top_k_freq=args.top_k_freq,
        d_model=args.d_model,
        prompt_pool_size=args.prompt_pool_size,
        pool_dim=args.pool_dim,
        # lrate=args.learning_rate,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        lambda_ortho=args.lambda_ortho,
        prompt_dropout_rate=args.prompt_dropout_rate,
        patch_len=args.patch_len,
        stride=args.stride,
        padding_patch=args.padding_patch,
        transformer_dim=args.transformer_dim,
        transformer_head=args.transformer_head,
        e_layer=args.e_layer,
        dropout_n=args.dropout_n,
        d_ff=args.d_ff,
        anomaly_ratio=args.anomaly_ratio
    )

    print("Start training...", flush=True)
    

    print("stage 2: specific prompt training #####222222222222222222222222222#######", flush=True)


    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_mae = []
        
        for iter, (x,y, embeddings) in enumerate(train_loader):
            if iter == 870:
                print("0000000000")
            trainx = torch.Tensor(x).to(device).float() # [B, L, N]
            trainy = torch.Tensor(y).to(device).float()
            # trainx_mark = torch.Tensor(x_mark).to(device) 
            train_embedding = torch.Tensor(embeddings).to(device).float()
            metrics = engine.train_stage2(trainx, train_embedding, trainx)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # validation
        val_loss = []
        val_mae = []
        s1 = time.time()

        for iter, (x,y, embeddings) in enumerate(val_loader):
            valx = torch.Tensor(x).to(device).float()
            valy = torch.Tensor(y).to(device).float()
            # valx_mark = torch.Tensor(x_mark).to(device)
            val_embedding = torch.Tensor(embeddings).to(device).float()
            metrics = engine.eval_stage2(valx, val_embedding, valx)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae)
        mvalid_loss = np.mean(val_loss)
        mvalid_mae = np.mean(val_mae)

        # writer.add_scalar('Loss/Train_stage2', mtrain_loss, i)
        # writer.add_scalar('Loss/Valid_stage2', mvalid_loss, i)

        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} "
        print(
            log.format(i, mtrain_loss, mtrain_mae),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}, Valid MAE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_mae),
            flush=True,
        )

        if mvalid_loss < loss:
        # if mvalid_mae < mae_loss:
            print("###Update tasks appear###")
            if i <= 10:
                # It is not necessary to print the results of the testset when epoch is less than n, because the model has not yet converged.
                loss = mvalid_loss
                mae_loss = mvalid_mae
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                test_outputs = []
                test_y = []
                engine.model.eval()
                for iter, (x,y, embeddings) in enumerate(val_loader):
                    testx = torch.Tensor(x).to(device).float()
                    testy = torch.Tensor(y).to(device).float()
                    # testx_mark = torch.Tensor(x_mark).to(device)
                    test_embedding = torch.Tensor(embeddings).to(device).float()
                    with torch.no_grad():
                        preds = engine.model(testx, test_embedding)
                    test_outputs.append(preds.detach().cpu())
                    test_y.append(testx.detach().cpu())
                
                test_pre = torch.cat(test_outputs, dim=0)
                test_real = torch.cat(test_y, dim=0)

                amse = []
                amae = []
                
                for j in range(args.pred_len):
                    pred = test_pre[:, j,].to(device)
                    real = test_real[:, j, ].to(device)
                    metrics = metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MSE: {:.4f}, Test MAE: {:.4f}"
                    amse.append(metrics[0])
                    amae.append(metrics[1])

                log = "stage2 On average horizons, Test MSE: {:.4f}, Test MAE: {:.4f}"
                print(
                    log.format(
                        np.mean(amse), np.mean(amae)
                    )
                )

                if np.mean(amse) < test_log:
                    test_log = np.mean(amse)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss: {:.4f}".format(np.mean(amse)), end=", ")
                    print("Test low! Updating! Valid Loss: {:.4f}".format(mvalid_loss), end=", ")

                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        engine.scheduler.step(loss)
        # engine.scheduler.step(mvalid_mae)
        # engine.scheduler.step()

        if epochs_since_best_mae >= args.es_patience and i >= args.epochs//2: # early stop
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test ----------------------------------------------------------------------------------
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   
    engine.model.load_state_dict(torch.load(path + "best_model.pth"))

    # (1) stastic on the train set
    attens_energy = []
    anomaly_criterion = nn.MSELoss(reduce=False)
    engine.model.eval()
    with torch.no_grad():
        for iter, (x,y, embeddings) in enumerate(train_loader):
            trainx = torch.Tensor(x).to(device).float() # [B, L, N]
            trainy = torch.Tensor(y).to(device).float()
            # trainx_mark = torch.Tensor(x_mark).to(device) 
            train_embedding = torch.Tensor(embeddings).to(device).float()

            
            predict = engine.model(trainx, train_embedding)
            # criterion
            score = torch.mean(anomaly_criterion(trainx, predict), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
        
    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)

    # (2) find the threshold
    attens_energy = []
    test_labels = []
    anomaly_criterion = nn.MSELoss(reduce=False)
    for i, (batch_x, batch_y, embeddings) in enumerate(test_loader):
        batch_x = batch_x.float().to(engine.device)
        # reconstruction
        outputs = engine.model(batch_x, embeddings.to(engine.device))
        # criterion
        score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
        score = score.detach().cpu().numpy()
        attens_energy.append(score)
        test_labels.append(batch_y)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    test_energy = np.array(attens_energy)
    combined_energy = np.concatenate([train_energy, test_energy], axis=0)
    threshold = np.percentile(combined_energy, 100 - engine.anomaly_ratio)
    print("Threshold :", threshold)

    # (3) evaluation on the test set
    pred = (test_energy > threshold).astype(int)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
    test_labels = np.array(test_labels)
    gt = test_labels.astype(int)

    print("pred:   ", pred.shape)
    print("gt:     ", gt.shape)

    # (4) detection adjustment
    gt, pred = adjustment(gt, pred)

    pred = np.array(pred)
    gt = np.array(gt)
    print("pred: ", pred.shape)
    print("gt:   ", gt.shape)

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision,
        recall, f_score))





if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))