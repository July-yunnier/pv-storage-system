# 全部的实验



# for model in 'PatchTST' 'ModernTCN' 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
for model in 'ConvPVNet'
do
  nohup python -u run_PV8.py \
    --device 'cuda:7' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 2 \
    --dp 0.02 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models4/PV2/model_${model}_26_1_4_woglu.pth" > logs4/PV2/log_${model}_26_1_4_woglu.log
done


# for model in 'PatchTST' 'ModernTCN' 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
for model in 'ConvPVNet'
do
  nohup python -u run_PV8.py \
    --device 'cuda:7' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 20 \
    --D 64 \
    --P 6 \
    --S 3 \
    --r 2 \
    --dp 0.03 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models4/PV2/model_${model}_52_4_4_woglu.pth" > logs4/PV2/log_${model}_52_4_4_woglu.log
done


# for model in 'PatchTST' 'ModernTCN' 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
for model in 'ConvPVNet'
do
  nohup python -u run_PV8.py \
    --device 'cuda:7' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 104 \
    --output_size 16 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 20 \
    --D 64 \
    --P 4 \
    --S 4 \
    --r 2 \
    --dp 0.05 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models4/PV2/model_${model}_104_16_4_woglu.pth" > logs4/PV2/log_${model}_104_16_4_woglu.log
done



