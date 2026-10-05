# 消融实验



for model in 'ConvPVNet'
do
for input_length in 4 8 16 26 52
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size $input_length \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.04 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_${input_length}_4_4.pth" > logs4/PV1/log_${model}_${input_length}_4_4.log
done
done





####################################################################################################################################





for model in 'ConvPVNet'
do
for input_length in 4 8 16 26 52
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size $input_length \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.05 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_${input_length}_4_4.pth" > logs4/PV2/log_${model}_${input_length}_4_4.log
done
done















# for model in 'ConvPVNet'
# do
# for input_length in 8 16 26 52 104
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:4' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size $input_length \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 128 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV1/model_${model}_${input_length}_16_4.pth" > logs4/PV1/log_${model}_${input_length}_16_4.log
# done
# done





# ####################################################################################################################################





# for model in 'ConvPVNet'
# do
# for input_length in 8 16 26 52 104
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:4' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size $input_length \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.03 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV2/model_${model}_${input_length}_16_4.pth" > logs4/PV2/log_${model}_${input_length}_16_4.log
# done
# done
