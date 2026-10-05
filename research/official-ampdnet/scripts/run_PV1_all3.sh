


for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 8 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_8_4_4_search.pth" >> logs4/PV1/log_${model}_8_4_4_search.log
done
done
done
done



for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 16 \
    --output_size 16 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_16_16_4_search.pth" >> logs4/PV1/log_${model}_16_16_4_search.log
done
done
done
done



for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV1/model_${model}_26_26_4_search.pth" >> logs4/PV1/log_${model}_26_26_4_search.log
done
done
done
done





#######################################################################################





for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 8 \
    --output_size 4 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_8_4_4_search.pth" >> logs4/PV2/log_${model}_8_4_4_search.log
done
done
done
done



for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 16 \
    --output_size 16 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_16_16_4_search.pth" >> logs4/PV2/log_${model}_16_16_4_search.log
done
done
done
done



for model in 'ConvPVNet'
do
for d in 64 128
do
for dp in 0.01 0.02 0.03 0.04 0.05
do
for r in 1 2
do
  nohup python -u run_PV8.py \
    --device 'cuda:4' \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --y_step 1 \
    --epochs 40 \
    --D $d \
    --P 4 \
    --S 2 \
    --r $r \
    --dp $dp \
    --num_layers 1 \
    --norm 'minmax' \
    --model_save_pth "models4/PV2/model_${model}_26_26_4_search.pth" >> logs4/PV2/log_${model}_26_26_4_search.log
done
done
done
done



