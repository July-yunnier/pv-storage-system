# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST'


# for model in 'ConvPVNet'
# do
# for input_channels in 1 2 3 4 5 6 7
# do
#   nohup python -u run_PV1.py \
#     --device 'cuda:5' \
#     --model_name $model \
#     --dataset 'data/Solar6_processed.csv' \
#     --input_size 104 \
#     --output_size 52 \
#     --in_channels $input_channels \
#     --y_step 1 \
#     --hidden_size 64 \
#     --num_layers 1 \
#     --patch_len 8 \
#     --stride 4 \
#     --model_save_pth "models/PV6/model_${model}_104_52_${input_channels}.pth" > logs/PV6/log_${model}_104_52_${input_channels}.log
# done
# done


# for model in 'BiGRU' 'BiLSTM' 'GRU' 'LSTM'
# for model in 'BiGRU' 'BiLSTM' 'GRU' 'LSTM'
# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST' 'ConvPVNet'

# for model in 'Transformer' 'DSN' 'ModernTCN' 'TCN'
for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:3' \
    --model_name $model \
    --dataset 'data/Solar2_processed.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models2/PV2/model_${model}_26_1_4.pth" > logs2/PV2/log_${model}_26_1_4.log
done


for model in 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:3' \
    --model_name $model \
    --dataset 'data/Solar2_processed.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models2/PV2/model_${model}_26_1_4.pth" > logs2/PV2/log_${model}_26_1_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:3' \
    --model_name $model \
    --dataset 'data/Solar2_processed.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 16 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models2/PV2/model_${model}_26_1_4.pth" > logs2/PV2/log_${model}_26_1_4.log
done


