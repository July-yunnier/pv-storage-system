# coding=utf-8

# model hyper-parameters
# rand_seed = 314
f_x = './data/x_site_1B.pkl'
f_y = './data/y_site_1B.pkl'

# model_name = 'STCN'  # ['RNN', 'GRU', 'LSTM', 'TCN', 'STCN']
model_name = 'ModernTCN'  # ['MLP', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'CNN-LSTM', 'CNN-GRU', 'TCN', 'DLinear', 'NLinear', 'Linear', 'RLinear', 'ModernTCN', 'TSMixer', 'PatchMixer', 'iTransformer', 'PatchTST', 'Transformer']
# device = 'cpu'  # 'cpu' or 'cuda'

# 输入窗口大小
input_size = 156
# 输出窗口大小
output_size = 52
# 输入变量数
in_channels = 4

hidden_size = 64
num_layers = 1
# TCN的层数
levels = 3
# TCN的卷积核大小
kernel_size = 4
dropout = 0.05

# 每次训练的batchsize,我依次试了640,320,256,128,64
# batchsize=64试了发现速度太慢,且性能提升很小很有限,遂放弃
batch_size = 128
lr = 0.001
# 训练的epoch数
n_epochs = 10
model_save_pth = './models/model_{}.pth'.format(model_name)


def print_params():
    print('\n------ Parameters ------')
    # print('rand_seed = {}'.format(rand_seed))
    print('f_x = {}'.format(f_x))
    print('f_y = {}'.format(f_y))
    # print('device = {}'.format(device))
    print('input_size = {}'.format(input_size))
    print('hidden_size = {}'.format(hidden_size))
    print('num_layers = {}'.format(num_layers))
    print('output_size = {}'.format(output_size))
    print('levels (for TCN) = {}'.format(levels))
    print('kernel_size (for TCN) = {}'.format(kernel_size))
    print('dropout (for TCN) = {}'.format(dropout))
    print('in_channels (for STCN) = {}'.format(in_channels))
    print('batch_size = {}'.format(batch_size))
    print('lr = {}'.format(lr))
    print('n_epochs = {}'.format(n_epochs))
    print('model_save_pth = {}'.format(model_save_pth))
    print('------------------------\n')
