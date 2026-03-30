 #!/bin/bash
export PYTHONPATH=/TimeCMA-main:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=1
device="cuda:0"

data="MSL"
root_path="/dataset/MSL"
divides=("train" "val" "test")
num_nodes=55
input_len=100
output_len=0
d_model=768
# d_model=512
l_layers=12
freq="h"
# top_k_freq=3
top_k_freq=5

for divide in "${divides[@]}"; do
  log_file="./Results/emb_logs/${data}_${divide}.log"
  
  nohup python storage/store_prompt_emb.py \
    --device $device \
    --divide $divide \
    --root_path $root_path\
    --data $data \
    --num_nodes $num_nodes \
    --input_len $input_len \
    --output_len $output_len \
    --d_model $d_model \
    --l_layers $l_layers \
    --freq $freq \
    --top_k_freq $top_k_freq > $log_file

done 

