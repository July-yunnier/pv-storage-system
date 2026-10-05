# 消融实验




# for model in 'ConvPVNet'
for model in 'PVformer2'
do
for input_layers in 2 3 4 5
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 20 \
    --D 128 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.051 \
    --num_layers $input_layers \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_52_4_4_L${input_layers}.pth" > logs4/PV1/log_${model}_52_4_4_L${input_layers}.log
done
done





####################################################################################################################################





# for model in 'ConvPVNet'
for model in 'PVformer2'
do
for input_layers in 2 3 4 5
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 52 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 20 \
    --D 128 \
    --P 4 \
    --S 2 \
    --r 2 \
    --dp 0.01 \
    --num_layers $input_layers \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_52_4_4_L${input_layers}.pth" > logs4/PV2/log_${model}_52_4_4_L${input_layers}.log
done
done







# for model in 'ConvPVNet'
# do
# for input_layers in 2 3 4 5
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:6' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 16 \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 128 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.02 \
#     --num_layers $input_layers \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV1/model_${model}_16_16_4_L${input_layers}.pth" > logs4/PV1/log_${model}_16_16_4_L${input_layers}.log
# done
# done





# ####################################################################################################################################





# for model in 'ConvPVNet'
# do
# for input_layers in 2 3 4 5
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:6' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 16 \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.03 \
#     --num_layers $input_layers \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV2/model_${model}_16_16_4_L${input_layers}.pth" > logs4/PV2/log_${model}_16_16_4_L${input_layers}.log
# done
# done
