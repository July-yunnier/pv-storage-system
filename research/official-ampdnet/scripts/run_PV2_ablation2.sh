# 消融实验


# for model in 'ConvPVNet12'
# for model in 'ConvPVNet'
# for model in 'ConvPVNet12' 'ConvPVNet' 'ModernTCN'
for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV6.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models3/PV1/model_${model}_26_1_4_new_2.pth" > logs3/PV1/log_${model}_26_1_4_new_2.log
done


for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV6.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models3/PV2/model_${model}_26_1_4_new_2.pth" > logs3/PV2/log_${model}_26_1_4_new_2.log
done



