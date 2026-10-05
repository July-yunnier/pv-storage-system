# for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
# for model in 'TSMixer' 'PatchMixer' 'PatchTST' 'iTransformer' 'TSLANet' 'Pathformer' 'Leddam'
# for model in 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'BiLSTM' 'BiGRU'
# for model in 'ConvPVNet'
# for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV2_H1.py \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 26 \
#     --output_size 1 \
#     --in_channels 4 \
#     --y_step 1 \
#     --num_layers 1
# done


# # for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV2_H4.py \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 52 \
#     --output_size 4 \
#     --in_channels 4 \
#     --y_step 1 \
#     --num_layers 1
# done


# # for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV2_H16.py \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 104 \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --num_layers 1
# done


for model in 'ConvPVNet'
do
  python -u myplot_PV2_H26.py \
    --model_name $model \
    --dataset 'data/Solar2_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --D 64 \
    --dp 0.03 \
    --y_step 1 \
    --num_layers 1
done