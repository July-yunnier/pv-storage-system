


# for model in 'PatchTST' 'ModernTCN' 'iTransformer' 'DLinear' 'ConvPVNet'
# for model in 'ConvPVNet'
# for model in 'TSMixer'
for model in 'Transformer2'
do
  nohup python -u run_PV8.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 39 \
    --output_size 1 \
    --in_channels 1 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 2 \
    --dp 0.02 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models3/PV2/model_${model}_39_1_1.pth" > logs3/PV2/log_${model}_39_1_1.log
done



# for model in 'Transformer2'
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 26 \
#     --output_size 1 \
#     --in_channels 1 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models3/PV1/model_${model}_26_1_1.pth" > logs3/PV1/log_${model}_26_1_1.log
# done



# for model in 'PatchTST' 'ModernTCN' 'iTransformer' 'DLinear' 'ConvPVNet'
# for model in 'ConvPVNet'
# for model in 'TSMixer'
# for model in 'Transformer2'
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 52 \
#     --output_size 1 \
#     --in_channels 1 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models3/PV2/model_${model}_52_1_1.pth" > logs3/PV2/log_${model}_52_1_1.log
# done



# for model in 'PatchTST' 'ModernTCN' 'iTransformer' 'DLinear' 'ConvPVNet'
# for model in 'ConvPVNet'
# for model in 'TSMixer'
# for model in 'Transformer2'
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 78 \
#     --output_size 1 \
#     --in_channels 1 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models3/PV2/model_${model}_78_1_1.pth" > logs3/PV2/log_${model}_78_1_1.log
# done



# for model in 'PatchTST' 'ModernTCN' 'iTransformer' 'DLinear' 'ConvPVNet'
# # for model in 'ConvPVNet'
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:3' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 52 \
#     --output_size 52 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV1/model_${model}_52_52_4.pth" > logs4/PV1/log_${model}_52_52_4.log
# done





# for model in 'TimesNet' 'TimeMixer'
# # for model in 'ConvPVNet'
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:5' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 26 \
#     --output_size 26 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV2/model_${model}_26_26_4.pth" > logs4/PV2/log_${model}_26_26_4.log
# done
