# 消融实验



# ConvPVNet1是
# ConvPVNet2是
# ConvPVNet3是
# ConvPVNet4是
# ConvPVNet5是
# ConvPV1是只有ConvGLU1
# ConvPV2是只有ConvGLU2
# ConvPV3是ConvGLU1+ConvGLU1
# ConvPV4是ConvGLU2+ConvGLU2
# ConvPV5是未解耦的单个ConvGLU



# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 8 \
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
    --model_save_pth "models4/PV1/model_${model}_8_4_4.pth" > logs4/PV1/log_${model}_8_4_4.log
done



# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 16 \
    --output_size 16 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 128 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.02 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_16_16_4.pth" > logs4/PV1/log_${model}_16_16_4.log
done



# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.01 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_26_26_4.pth" > logs4/PV1/log_${model}_26_26_4.log
done





##########################################################################################





# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 8 \
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
    --model_save_pth "models4/PV2/model_${model}_8_4_4.pth" > logs4/PV2/log_${model}_8_4_4.log
done



# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 16 \
    --output_size 16 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 1 \
    --dp 0.03 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_16_16_4.pth" > logs4/PV2/log_${model}_16_16_4.log
done



# for model in 'ConvPVNet' 'ConvPVNet1' 'ConvPVNet6' 'ConvPVNet7' 'ConvPVNet8' 'ConvPVNet9' 'ConvPVNet10' 'ConvPV1' 'ConvPV2' 'ConvPV3' 'ConvPV4' 'ConvPV5'
# for model in 'ConvPVNet'
for model in 'ConvPV7'
do
  nohup python -u run_PV8.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D 64 \
    --P 4 \
    --S 2 \
    --r 2 \
    --dp 0.03 \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_26_26_4.pth" > logs4/PV2/log_${model}_26_26_4.log
done
