#!/bin/bash
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0

# 公用设置
device="cuda:1"
data="SWAT"
root_path="/dataset/SWaT"
num_nodes=51
seed=2025
seq_len=100

log_path="./Results/${data}/"
mkdir -p $log_path



pred_len=100
d_model=768
pool_dim=768
batch_size=128
d_ff=64
dropout_n=0.0
e_layer=2
epochs=30
patch_len=12
stride=12
prompt_dropout_rate=0.1
prompt_pool_size=12
learning_rate=6.086792272218907e-06
top_k_freq=9
transformer_dim=128
transformer_head=4
anomaly_ratio=0.5

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_t${top_k_freq}_pps${prompt_pool_size}_lo${lambda_ortho}_pdr${prompt_dropout_size}_bs${batch_size}.log"
nohup python train.py \
  --device $device \
  --data $data \
  --root_path $root_path \
  --anomaly_ratio $anomaly_ratio \
  --num_nodes $num_nodes \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs $epochs\
  --d_model $d_model \
  --pool_dim $pool_dim\
  --batch_size $batch_size \
  --d_ff $d_ff\
  --dropout_n $dropout_n\
  --e_layer $e_layer \
  --patch_len $patch_len \
  --stride $stride \
  --prompt_pool_size $prompt_pool_size \
  --prompt_dropout_rate $prompt_dropout_rate \
  --learning_rate $learning_rate\
  --top_k_freq $top_k_freq \
  --transformer_dim $transformer_dim\
  --transformer_head $transformer_head > $log_file 
  

