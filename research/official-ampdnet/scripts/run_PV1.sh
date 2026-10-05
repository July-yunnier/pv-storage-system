#

# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST' 'Transformer'


# for model in 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN'

# for model in 'ModernTCN'
# for model in 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST'
# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST' 'ModernTCN'
# for model in 'MLP' 'CNN' 'LSTM'
# for model in 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN'
# for model in 'TSMixer' 'PatchMixer' 'iTransformer'
# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST'

# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST'
# do
#   nohup python -u run_PV1.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed.csv' \
#     --input_size 156 \
#     --output_size 52 \
#     --in_channels 4 \
#     --y_step 1 \
#     --hidden_size 64 \
#     --num_layers 1 \
#     --patch_len 8 \
#     --stride 4 \
#     --model_save_pth "models/PV1/model_${model}_156_52_4.pth" > logs/PV1/log_${model}_156_52_4.log
# done


# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST'
# do
#   nohup python -u run_PV1.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed.csv' \
#     --input_size 156 \
#     --output_size 52 \
#     --in_channels 4 \
#     --y_step 1 \
#     --hidden_size 64 \
#     --num_layers 1 \
#     --patch_len 8 \
#     --stride 4 \
#     --model_save_pth "models/PV2/model_${model}_156_52_4.pth" > logs/PV2/log_${model}_156_52_4.log
# done


# for model in 'MLP' 'CNN' 'LSTM' 'GRU' 'BiLSTM' 'BiGRU' 'CNN-LSTM' 'CNN-GRU' 'DLinear' 'Linear' 'ModernTCN' 'TSMixer' 'PatchMixer' 'iTransformer' 'PatchTST' 'ConvPVNet'
# for model in 'TCN'
# for model in 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'BiLSTM' 'BiGRU'

# for model in 'Transformer' 'DSN' 'ModernTCN' 'TCN'



for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_26_1_4.pth" > logs/PV1/log_${model}_26_1_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_26_1_4.pth" > logs/PV1/log_${model}_26_1_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_26_1_4.pth" > logs/PV1/log_${model}_26_1_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_26_1_4.pth" > logs/PV2/log_${model}_26_1_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_26_1_4.pth" > logs/PV2/log_${model}_26_1_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_26_1_4.pth" > logs/PV2/log_${model}_26_1_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_26_1_4.pth" > logs/PV6/log_${model}_26_1_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_26_1_4.pth" > logs/PV6/log_${model}_26_1_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 26 \
    --output_size 1 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_26_1_4.pth" > logs/PV6/log_${model}_26_1_4.log
done














for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_52_4_4.pth" > logs/PV1/log_${model}_52_4_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_52_4_4.pth" > logs/PV1/log_${model}_52_4_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_52_4_4.pth" > logs/PV1/log_${model}_52_4_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_52_4_4.pth" > logs/PV2/log_${model}_52_4_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_52_4_4.pth" > logs/PV2/log_${model}_52_4_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_52_4_4.pth" > logs/PV2/log_${model}_52_4_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_52_4_4.pth" > logs/PV6/log_${model}_52_4_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_52_4_4.pth" > logs/PV6/log_${model}_52_4_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_52_4_4.pth" > logs/PV6/log_${model}_52_4_4.log
done















for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_104_52_4.pth" > logs/PV1/log_${model}_104_52_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_104_52_4.pth" > logs/PV1/log_${model}_104_52_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV1/model_${model}_104_52_4.pth" > logs/PV1/log_${model}_104_52_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_104_52_4.pth" > logs/PV2/log_${model}_104_52_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_104_52_4.pth" > logs/PV2/log_${model}_104_52_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV2/model_${model}_104_52_4.pth" > logs/PV2/log_${model}_104_52_4.log
done


for model in 'ConvPVNet' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_104_52_4.pth" > logs/PV6/log_${model}_104_52_4.log
done


for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_104_52_4.pth" > logs/PV6/log_${model}_104_52_4.log
done


for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
do
  nohup python -u run_PV5.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar6_processed_3.csv' \
    --input_size 104 \
    --output_size 52 \
    --in_channels 4 \
    --y_step 1 \
    --hidden_size 64 \
    --num_layers 1 \
    --patch_len 8 \
    --stride 4 \
    --model_save_pth "models/PV6/model_${model}_104_52_4.pth" > logs/PV6/log_${model}_104_52_4.log
done

