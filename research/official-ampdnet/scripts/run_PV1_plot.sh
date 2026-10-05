# for model in 'MLP' 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'Transformer' 'DSN' 'ModernTCN' 'TCN'
# for model in 'LSTM' 'GRU' 'CNN-LSTM' 'CNN-GRU' 'BiLSTM' 'BiGRU'
# for model in 'ConvPVNet'
# for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV1_H1.py \
#     --device 'cuda:4' \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 26 \
#     --output_size 1 \
#     --in_channels 4 \
#     --y_step 1 \
#     --D 128 \
#     --r 2 \
#     --dp 0.01
# done


# # for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV1_H4.py \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 52 \
#     --output_size 4 \
#     --in_channels 4 \
#     --y_step 1 \
#     --D 128 \
#     --r 1 \
#     --dp 0.051
# done


# # for model in 'PVformer2'
# for model in 'ConvPVNet'
# do
#   python -u myplot_PV1_H16.py \
#     --model_name $model \
#     --dataset 'data/Solar1_processed_3.csv' \
#     --input_size 104 \
#     --output_size 16 \
#     --in_channels 4 \
#     --y_step 1 \
#     --D 64 \
#     --r 1 \
#     --dp 0.021
# done


for model in 'ConvPVNet'
do
  python -u myplot_PV1_H26.py \
    --model_name $model \
    --dataset 'data/Solar1_processed_3.csv' \
    --input_size 26 \
    --output_size 26 \
    --in_channels 4 \
    --y_step 1 \
    --D 64 \
    --r 1 \
    --dp 0.01
done