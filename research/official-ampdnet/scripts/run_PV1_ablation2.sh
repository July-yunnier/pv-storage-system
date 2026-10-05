# 消融实验





for model in 'ConvPVNet'
do
for input_variates in 1 2 3 4 5 6 7
do
  nohup python -u run_PV8.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 8 \
    --output_size 4 \
    --in_channels $input_variates \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.04 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_8_4_${input_variates}_inverse.pth" > logs4/PV1/log_${model}_8_4_${input_variates}_inverse.log
done
done





####################################################################################################################################





for model in 'ConvPVNet'
do
for input_variates in 1 2 3 4 5 6 7
do
  nohup python -u run_PV8.py \
    --device 'cuda:5' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 8 \
    --output_size 4 \
    --in_channels $input_variates \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.05 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_8_4_${input_variates}_inverse.pth" > logs4/PV2/log_${model}_8_4_${input_variates}_inverse.log
done
done







# for model in 'ConvPVNet'
# do
# for input_variates in 1 2 3 4 5 6 7
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:5' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 16 \
#     --output_size 16 \
#     --in_channels $input_variates \
#     --y_step 1 \
#     --epochs 40 \
#     --D 128 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.02 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV1/model_${model}_16_16_${input_variates}.pth" > logs4/PV1/log_${model}_16_16_${input_variates}.log
# done
# done





# ####################################################################################################################################





# for model in 'ConvPVNet'
# do
# for input_variates in 1 2 3 4 5 6 7
# do
#   nohup python -u run_PV8.py \
#     --device 'cuda:5' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 16 \
#     --output_size 16 \
#     --in_channels $input_variates \
#     --y_step 1 \
#     --epochs 40 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 1 \
#     --dp 0.03 \
#     --num_layers 1 \
#     --norm 'minmax' \
#     --model_save_pth "models4/PV2/model_${model}_16_16_${input_variates}.pth" > logs4/PV2/log_${model}_16_16_${input_variates}.log
# done
# done
