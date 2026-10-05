for model in 'ConvPVNet1' 'ConvPVNet2' 'ConvPVNet3' 'ConvPVNet4' 'ConvPVNet5' 'ConvPVNet6' 'ConvPVNet7'
do
  nohup python -u run_PV1.py \
    --device 'cuda:6' \
    --model_name $model \
    --dataset 'data/Solar2_processed.csv' \
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