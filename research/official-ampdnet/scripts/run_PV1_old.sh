# for model in 'PatchTST' 'ModernTCN' 'iTransformer' 'DLinear' 'ConvPVNet' 'CNN-LSTM'

# for model in 'TSMixer'
# for model in 'Transformer2'
# for model in 'ConvPVNet' 'CNN-LSTM' 'ModernTCN' 'PatchTST' 'Transformer'
# # for model in 'ConvPVNet'
# # for model in 'PatchTST'
# do
#   python -u run_PV8_old.py \
#     --device 'cuda:0' \
#     --model_name $model \
#     --dataset 'data/Solar2_processed_3.csv' \
#     --input_size 96 \
#     --output_size 4 \
#     --in_channels 6 \
#     --y_step 1 \
#     --epochs 20 \
#     --D 64 \
#     --P 4 \
#     --S 2 \
#     --r 2 \
#     --dp 0.02 \
#     --num_layers 2 \
#     --norm 'minmax' \
#     --plot_date 2020-12-21 \
#     --weather_type "晴天" \
#     --plot_days 7 \
#     --model_save_pth "models/PV2/model_${model}_39_1_1.pth" 
# # done
# models=('ConvPVNet' 'CNN-LSTM' 'ModernTCN' 'PatchTST' 'Transformer' 'DLinear')
# models=('ConvPVNet')
#models=('DLinear')
models=('ConvPVNet')
# 预测任务配置：格式为"任务名称:input_size:output_size"
# tasks=('15min:48:1' '1h:96:4' '4h:192:16')
#tasks=('15min_48:48:1' '15min_24:24:1' '15min_16:16:1' '15min_12:12:1' '15min_8:8:1')
# tasks=('15min_8:8:1' '1h_32:32:4' '4h_64:64:16')
# tasks=('1h_32:32:4' '4h_64:64:16')
# tasks=('15min_8:8:1')
# tasks=('1h_32:32:4')
tasks=('4h_64:64:16')
# tasks=('1h_8:8:4' '1h_16:16:4' '1h_24:24:4' '1h_32:32:4' '1h_64:64:4' '4h_16:16:16' '4h_24:24:16' '4h_32:32:16' '4h_64:64:16' '4h_128:128:16')
# tasks=('4h_16:16:16' '4h_24:24:16' '4h_32:32:16' '4h_64:64:16' '4h_128:128:16')
# tasks=('1h_16:16:4')
# 可选：控制并行数量
MAX_PARALLEL=3

# 外层循环：模型
for model in "${models[@]}"; do
  # 内层循环：预测任务
  for task_config in "${tasks[@]}"; do
    # 解析配置字符串
    IFS=':' read -r task_name input_size output_size <<< "$task_config"
    
    echo "🚀 model: $model | task: $task_name | input_length: $input_size | output_length: $output_size"
    
    # 后台运行（可选，需根据硬件调整）
    python -u run_PV8_old.py \
      --device 'cuda:6' \
      --model_name "$model" \
      --dataset 'data/Solar1_processed_3.csv' \
      --input_size "$input_size" \
      --output_size "$output_size" \
      --in_channels 4 \
      --y_step 1 \
      --epochs 10 \
      --D 64 \
      --P 4 \
      --S 2 \
      --r 2 \
      --dp 0.0 \
      --num_layers 2 \
      --hidden_size 16 \
      --stride 4 \
      --norm 'minmax' \
      --plot_date  None \
      --weather_type "晴天" \
      --plot_days 7 \
      --model_save_pth "models/PV_best11/model_${model}_${task_name}.pth" 
    
    # 控制并行任务数（等待任一任务完成再继续）
    if (( $(jobs -r | wc -l) >= MAX_PARALLEL )); then
      wait -n
    fi
    
  done
  # 等待当前模型所有任务完成（可选）
  # wait
done

# 等待所有任务完成
wait
echo "done!"