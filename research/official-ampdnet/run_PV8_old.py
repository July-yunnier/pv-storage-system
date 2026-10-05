# coding:utf-8

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
# from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.metrics import mean_absolute_error # 评分 MAE 的计算函数
import models
import models2
import ModernTCN
import TSMixer
import PatchMixer
import PatchTST
import Trans_test
import iTrans_test

import ConvPVNet
import ConvPVNet2

import TimeMixer
import TimesNet
import TSLANet
import Pathformer

import Leddam2024

import config as cfg
import argparse
import os


# 设置随机种子
seed = 44
# seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True #将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False #关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
np.random.seed(seed)



# 
parser = argparse.ArgumentParser(description='PV forecasting')
# 模型的配置config
parser.add_argument('--device', type=str, default='cuda:3', help='GPU or CPU')
parser.add_argument('--model_name', type=str, default='LSTM')
parser.add_argument('--dataset', type=str, default='data/Solar1_processed.csv')
parser.add_argument('--input_size', type=int, default=39)
parser.add_argument('--output_size', type=int, default=1)
parser.add_argument('--in_channels', type=int, default=5)
parser.add_argument('--y_step', type=int, default=1)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--D', type=int, default=64)
parser.add_argument('--P', type=int, default=4)
parser.add_argument('--S', type=int, default=2)
parser.add_argument('--r', type=int, default=2)
parser.add_argument('--dp', type=float, default=0.02)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--norm', type=str, default='minmax')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=1, help='layers of models')
parser.add_argument('--patch_len', type=int, default=5, help='patch_len of PatchMixer, PatchTST, ModernTCN and ConvPVNet')
parser.add_argument('--stride', type=int, default=3, help='stride of PatchMixer, PatchTST, ModernTCN and ConvPVNet')
parser.add_argument('--model_save_pth', type=str, default='./models/Australia/model_MLP_39_1_5.pth')
parser.add_argument('--weight_decay',type=float,default=1e-4,help='L2 penalty coeffcinet')
# ... 其他参数 ...
parser.add_argument('--plot_date', type=str, default=None, help='Date to plot, format: YYYY-MM-DD')
parser.add_argument('--weather_type', type=str, default='未知天气', help='Weather type for plot title: 晴天/阴雨天/突变天')
parser.add_argument('--plot_days',type=int,default='1',help='画图天数')
args = parser.parse_args()



net = None
if args.model_name == 'MLP':
    net = models.SimpleMLP(input_size=args.input_size*args.in_channels, output_size=args.output_size)
elif args.model_name == 'CNN':
    net = models.CNN(input_size=args.input_size, in_channels=args.in_channels, output_size=args.output_size)
# 对于nn.LSTM来说, input_size为输入数据的特征数/变量数
elif args.model_name == 'LSTM':
    net = models.SimpleLSTM(input_size=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'LSTM2':
    net = models.SimpleLSTM2(input_dim=args.in_channels, hidden_dim=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'GRU':
    net = models.SimpleGRU(input_size=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'BiLSTM':
    net = models.BiLSTM(input_size=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'BiGRU':
    net = models.BiGRU(input_size=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'CNN-LSTM':
    net = models.CNNLSTM(in_channels=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'CNN-GRU':
    net = models.CNNGRU(in_channels=args.in_channels, hidden_size=args.hidden_size, output_size=args.output_size, num_layers=args.num_layers)
elif args.model_name == 'TCN':
    net = models.TCN(input_size=args.in_channels, output_size=args.output_size, num_channels=[32]*1, kernel_size=4, dropout=0.05)
elif args.model_name == 'DLinear':
    # net = models2.DLinear(input_size=args.input_size*args.in_channels, output_size=args.output_size)
    net = models2.DLinear(input_size=args.input_size, output_size=args.output_size)
elif args.model_name == 'Linear':
    net = models2.Linear(input_size=args.input_size*args.in_channels, output_size=args.output_size)
# elif args.model_name == 'NLinear':
#     net = models2.NLinear(input_size=args.input_size*args.in_channels, output_size=args.output_size)
elif args.model_name == 'ModernTCN':
    net = ModernTCN.ModernTCN2(M=args.in_channels, L=args.input_size, T=args.output_size, D=args.D, P=args.P, S=args.S, r=args.r, kernel_sizes=[5,51], num_layers=args.num_layers)
elif args.model_name == 'TSMixer':
    net = TSMixer.TSMixer(input_size=args.input_size, output_size=args.output_size, in_channels=args.in_channels)
elif args.model_name == 'PatchMixer':
    net = PatchMixer.PatchMixer(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, patch_len=args.P, stride=args.S)
elif args.model_name == 'PatchMixer2':
    net = PatchMixer2.PatchMixer2(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, patch_len=args.P, stride=args.S)
elif args.model_name == 'iTransformer':
    net = iTrans_test.iTransformer(input_size=args.input_size, output_size=args.output_size, d_model=128, enc_in=args.in_channels, factor=5, dropout=args.dp, output_attention=False, n_heads=4, d_ff=128, activation=F.gelu, e_layers=1)
elif args.model_name == 'PatchTST':
    net = PatchTST.PatchTST(input_size=args.input_size, enc_in=args.in_channels, output_size=args.output_size, d_model=args.D, dropout=args.dp, factor=5, output_attention=False, n_heads=4, d_ff=128, activation=F.gelu, e_layers=args.num_layers, patch_len=args.P, stride=args.S)


elif args.model_name == 'Transformer':
    net = Trans_test.Transformer(enc_in=args.in_channels, dec_in=args.in_channels, output_size=args.output_size, e_layers=1, d_layers=1)
elif args.model_name == 'Transformer2':
    net = Trans_test.Transformer2(enc_in=args.in_channels, dec_in=args.in_channels, c_out=1, input_size=args.input_size, output_size=args.output_size, e_layers=1)


elif args.model_name == 'ConvPVNet':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=args.num_layers)
elif args.model_name == 'ConvPatchPVNet':
    net = ConvPVNet2.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=args.num_layers)

elif args.model_name == 'ConvPVNet1':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5], dilat=[1], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet2':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5], dilat=[2], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet3':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5], dilat=[3], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet4':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5], dilat=[4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet5':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,1,1,1], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)


# 我20240530要跑的模型: ConvPVNet1, ConvPVNet6, ConvPVNet7, ConvPVNet8, ConvPVNet9, ConvPVNet10, ConvPVNet11
elif args.model_name == 'ConvPVNet6':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5], dilat=[1,2], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet7':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5], dilat=[1,2,3], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet8':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet9':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5,5], dilat=[1,2,3,4,5], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet10':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5,5,5], dilat=[1,2,3,4,5,6], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPVNet11':
    net = ConvPVNet.ConvPVNet(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5,5,5,5], dilat=[1,2,3,4,5,6,7], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)


elif args.model_name == 'ConvPV1':
    net = ConvPVNet.ConvPV1(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPV2':
    net = ConvPVNet.ConvPV2(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPV3':
    net = ConvPVNet.ConvPV3(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
elif args.model_name == 'ConvPV4':
    net = ConvPVNet.ConvPV4(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
# 未解耦的ConvPVNet, 只用了一个ConvGLU
elif args.model_name == 'ConvPV5':
    net = ConvPVNet.ConvPV5(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
# 未解耦的ConvPVNet, 用了2个ConvGLU
elif args.model_name == 'ConvPV6':
    net = ConvPVNet.ConvPV6(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)
# ConvPV7是去掉AMSConv的模型
elif args.model_name == 'ConvPV7':
    net = ConvPVNet.ConvPV7(M=args.in_channels, L=args.input_size, T=args.output_size, kernel_sizes=[5,5,5,5], dilat=[1,2,3,4], D=args.D, P=args.P, S=args.S, r=args.r, dp=args.dp, num_layers=1)

elif args.model_name == 'DSN':
    net = DSN.DSN(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size)


elif args.model_name == 'TimeMixer':
    net = TimeMixer.mymodel
elif args.model_name == 'TimesNet':
    net = TimesNet.mymodel


elif args.model_name == 'TSLANet':
    net = TSLANet.TSLANet(input_size=args.input_size, output_size=args.output_size, emb_dim=args.D, patch_size=args.P, num_layers=args.num_layers, dropout=args.dp)
elif args.model_name == 'Pathformer':
    net = Pathformer.mymodel
elif args.model_name == 'PatchTST_hll':
    net = PatchTST_hll.PatchTST2(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=64, P=4, S=2, r=2, dropout=args.dp, num_layers=2)
elif args.model_name == 'Leddam':
    net = Leddam2024.Leddam2024(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, d_model=args.D, dropout=args.dp, pe_type='sincos', num_layers=args.num_layers)

elif args.model_name == 'PVformer':
    net = PVformer.PVformer(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)

# 注: PVformer2 即为我的 BigDIA 2024 的 TSformer 模型
elif args.model_name == 'PVformer2':
    net = PVformer.PVformer2(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)
elif args.model_name == 'PVformer2_wo_trend':
    net = PVformer.PVformer2_wo_trend(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)
elif args.model_name == 'PVformer2_wo_seasonal':
    net = PVformer.PVformer2_wo_seasonal(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)
elif args.model_name == 'PVformer2_Inverse':
    net = PVformer.PVformer2_Inverse(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)

elif args.model_name == 'PVformer3':
    net = PVformer.PVformer3(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)
elif args.model_name == 'PVformer4':
    net = PVformer.PVformer4(in_channels=args.in_channels, input_size=args.input_size, output_size=args.output_size, n_head=4, D1=128, D2=args.D, P=args.P, S=args.S, r=args.r, dp1=args.dp, dp2=args.dp, num_layers=args.num_layers)




# TimeMixer, TimeXer, UniRepLKNet, MICN, TDformer, Pathformer, SE block, GLU, ConvGLU, 季节趋势性分解


device = torch.device(args.device if torch.cuda.is_available() else "cpu")
net = net.to(device)


# x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=args.f_x, f_y=args.f_y)






# PV1数据集
# file_path = 'data/Solar1_processed.csv'
file_path = args.dataset
dtype_dict = {'TSI': np.float32, 'DNI': np.float32, 'GHI': np.float32, 
              'Temp': np.float32, 'Humidity': np.float32, 'Atmos': np.float32, 
              'Power': np.float32}
data = pd.read_csv(file_path, dtype=dtype_dict)
# data = pd.read_csv(file_path)
print('Solar_processed_shape:{}\n'.format(data.shape))
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)
feature = ['Power', 'TSI', 'DNI', 'GHI', 'Temp', 'Humidity', 'Atmos']
# 2. 更新特征列表和通道数
#if args.model_name =='PatchTST':
#    print('当前训练为：PATCHTST')
#    feat_names = feature[:args.in_channels]
#else:
    
#    time_features = ['hour_sin', 'hour_cos']
#    feature_add_time = time_features + feature 
#    feat_names = feature_add_time[:args.in_channels]
feat_names = feature[:args.in_channels]
# 3. 确保后续create_dataset2函数能正确处理新特征
# 您的create_dataset2已正确实现，无需修改


print(f"特征列表: {feat_names}")


# feature_inverted = ['Atmos', 'Humidity', 'Temp', 'GHI', 'DNI', 'TSI', 'Power']
# feature_wo_power = ['TSI', 'DNI', 'GHI', 'Temp', 'Humidity', 'Atmos']
# feat_names = feature[:args.in_channels]
#feat_names = feature_inverted[:args.in_channels]
# feat_names = feature_wo_power[:args.in_channels]
print('feat_names:{}\n'.format(feat_names))



# # 归一化数据集的时间步以外的所有变量
# # 获得数据集的时间步2019-01-01 06:00:00到2020-06-30 18:45:00的所有数据,建立一个子数据集
# # 2019年一整年+2020年上半年作为训练集验证集
# data1 = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:45:00')]
# print('dataset1_shape:{}\n'.format(data1.shape))
# # Get the maximum and minimum values for normalization
# max_value = data1[feature].max()
# min_value = data1[feature].min()
# # 打印Power的最大值最小值
# # print('max_power:{}\nmin_power:{}\n'.format(max_value['Power'], min_value['Power']))
# print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))
# # Normalize the dataset
# data[feature] = (data[feature] - min_value) / (max_value - min_value)
# # print('data:{}\n'.format(data))
# print('data_shape:{}\n'.format(data.shape))


# # 将子数据集data1去除掉得到data2
# # 2020年下半年的数据作为测试集
# data2 = data.drop(data1.index)


# 得到data数据集的时间步2019-01-01 06:00:00到2020-06-30 18:45:00的feature列对应的最大值和最小值
# max_value = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:00:00')][feature].max()
# min_value = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:00:00')][feature].min()
# print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))
# # Normalize the dataset
# data[feature] = (data[feature] - min_value) / (max_value - min_value)
# data1 = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:00:00')]
# print('dataset1_shape:{}\n'.format(data1.shape))
# data2 = data.drop(data1.index)
# print('dataset2_shape:{}\n'.format(data2.shape))

# max_value是data的前1000行的最大值, min_value是data的前1000行的最小值
train_len = int((len(data)-args.input_size-args.output_size+1-args.y_step+1)*0.6)+args.input_size+args.output_size-1+args.y_step-1
print('train_len:{}\n'.format(train_len))
max_value = data.head(train_len)[feature].max()
min_value = data.head(train_len)[feature].min()
print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))


# Normalize the dataset
if args.norm == 'minmax':
    data[feature] = (data[feature] - min_value) / (max_value - min_value)

elif args.norm == 'meanstd':
    mean_value = data.head(train_len)[feature].mean()
    std_value = data.head(train_len)[feature].std()
    print('mean_value:{}\nstd_value:{}\n'.format(mean_value, std_value))
    data[feature] = (data[feature] - mean_value) / std_value
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    print('时间戳范围: {} 到 {}'.format(data['timestamp'].min(), data['timestamp'].max()))
else:
    print("警告: 数据中没有timestamp列，无法绘制时间序列图")
    data['timestamp'] = None




# 用滑动窗口的方法来获得时间序列的样本
# 这个函数不行,会占用巨多的CPU内存,跑起来非常慢
# Create the dataset with a sliding window
def create_dataset(dataset, feat_names, x_length, y_length, y_step):
    data_x, data_y = [], []
    for start_id in range(0, len(dataset)-x_length-y_length+1-y_step+1, 1):
        # iloc取切片操作是前闭后开
        data_x.append(dataset[feat_names].iloc[start_id: start_id+x_length])
        data_y.append(dataset['Power'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
    return np.array(data_x), np.array(data_y)

# 样本量: len(self.data) - self.x_length - self.y_length + 1 - self.y_step + 1


# 在这个优化后的版本中, 我首先计算出 data_x 和 data_y 的大小, 并预先分配内存
# 然后, 我将 dataset 转换为 Numpy 数组, 并使用 Numpy 的切片操作来获取数据
# 这样可以避免在循环中不断扩展列表的开销, 并提高切片操作的速度
def create_dataset2(dataset, feat_names, x_length, y_length, y_step):
    num_samples = len(dataset) - x_length - y_length + 1 - y_step + 1
    num_features = len(feat_names)
    data_x = np.empty((num_samples, x_length, num_features), dtype=np.float32)
    data_y = np.empty((num_samples, y_length), dtype=np.float32)
    dataset_np = dataset[feat_names].to_numpy(dtype=np.float32)
    power_np = dataset['Power'].to_numpy(dtype=np.float32)
    for start_id in range(num_samples):
        data_x[start_id] = dataset_np[start_id: start_id+x_length]
        data_y[start_id] = power_np[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1]
    return data_x, data_y


# 滑动窗口获取样本
def create_dataset3(dataset, feat_names, x_length, y_length, y_step):
    num_samples = len(dataset) - x_length - y_length + 1 - y_step + 1
    num_features = len(feat_names)
    data_x1 = np.empty((num_samples, x_length, num_features), dtype=np.float32)
    data_x2 = np.empty((num_samples, y_length, num_features-1), dtype=np.float32)
    data_y = np.empty((num_samples, y_length), dtype=np.float32)
    dataset_np = dataset[feat_names].to_numpy(dtype=np.float32)
    # feat_names_wo_power是feat_names的第2个元素到最后一个元素
    feat_names_wo_power = feat_names[1:]
    dataset_wo_power_np = dataset[feat_names_wo_power].to_numpy(dtype=np.float32)
    power_np = dataset['Power'].to_numpy(dtype=np.float32)
    for start_id in range(num_samples):
        data_x1[start_id] = dataset_np[start_id: start_id+x_length]
        data_x2[start_id] = dataset_wo_power_np[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1]
        data_y[start_id] = power_np[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1]
    return data_x1, data_x2, data_y


# MAE的GPU版本
def mean_absolute_error(pred, target):
    return torch.abs(pred - target).mean()

# r2的GPU版本
def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot

# SMAPE的GPU版本
def smape(pred, target):
    return 2.0 * (torch.abs(pred - target) / (torch.abs(pred) + torch.abs(target))).mean()


# 根据lookbacks,horizons,特征数来分割形成样本,这里生成的是训练集验证集的样本
# 仅预测 y_t+1
# data1_x, data1_y = create_dataset(data1, feat_names, 39, 1, 1)
# 预测 y_t+1
data_x, data_y = create_dataset2(data, feat_names, args.input_size, args.output_size, args.y_step)
print('data1_x_shape: {}  data1_y_shape: {}'.format(data_x.shape, data_y.shape))


# 2014年到2016年的数据来得到训练集验证集
# 将data1_x和data1_y的前60%作为训练集,中间20%作为验证集,最后20%作为测试集
split_index1 = int(len(data_x) * 0.6)
split_index2 = int(len(data_x) * 0.8)
x_train = data_x[:split_index1]
y_train = data_y[:split_index1]
x_valid = data_x[split_index1:split_index2]
y_valid = data_y[split_index1:split_index2]
x_test = data_x[split_index2:]
y_test = data_y[split_index2:]
print('x_train_shape: {}  y_train_shape: {}'.format(x_train.shape, y_train.shape))
print('x_valid_shape: {}  y_valid_shape: {}'.format(x_valid.shape, y_valid.shape))
print('x_test_shape: {}  y_test_shape: {}'.format(x_test.shape, y_test.shape))



# split_index = int(len(data1_x) * 0.8)
# x_train = data1_x[:split_index]
# y_train = data1_y[:split_index]
# x_valid = data1_x[split_index:]
# y_valid = data1_y[split_index:]
# print('x_train_shape: {}  y_train_shape: {}'.format(x_train.shape, y_train.shape))
# print('x_valid_shape: {}  y_valid_shape: {}'.format(x_valid.shape, y_valid.shape))


# 2019年到2020年上半年的数据来得到训练集验证集
# 设置固定随机种子,每次运行都是选取一样的20%作为验证集
# np.random.seed(44)
# # 将data1_x和data1_y的随机80%作为训练集,剩余20%作为验证集
# split_index = int(len(data1_x) * 0.8)
# index = np.arange(len(data1_x))
# np.random.shuffle(index)
# x_train = data1_x[index[:split_index]]
# y_train = data1_y[index[:split_index]]
# x_valid = data1_x[index[split_index:]]
# y_valid = data1_y[index[split_index:]]
# print('x_train_shape: {}  y_train_shape: {}'.format(x_train.shape, y_train.shape))
# print('x_valid_shape: {}  y_valid_shape: {}'.format(x_valid.shape, y_valid.shape))


# 2020年下半年作为测试集
# 根据lookbacks,horizons,特征数来分割形成样本,这里生成的是测试集的样本
# 仅预测 y_t+1
# x_test, y_test = create_dataset(data2, feat_names, 39, 1, 1)
# 预测 y_t+1
# x_test, y_test = create_dataset(data2, feat_names, args.input_size, args.output_size, args.y_step)
# print('x_test_shape: {}  y_test_shape: {}'.format(x_test.shape, y_test.shape))







# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
x_val_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
# x_test_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)


# 将x_train,y_train,x_valid,y_valid,x_test,y_test从CPU内存中删除
# del x_train, y_train, x_valid, y_valid, x_test, y_test



# Create DataLoader for training, validation, and testing
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# 下面这段注释的代码是考虑batchsize的情况,每次把batchsize=128个样本拿进去训练
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True) 
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False) 


# torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08, weight_decay=0) 
# Adam 的默认参数: params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, 后面可以试一试权重衰减策略
# DLinear的库代码中,权重衰减也是设置的0
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)
criterion = nn.MSELoss().to(device)
mse_loss = nn.MSELoss().to(device)
mae_loss = nn.L1Loss().to(device)


epochs = args.epochs
print(net)

# 打印模型的所有可训练的参数总量
print('Number of trainable parameters: {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
# 打印模型的总参数量,包括可训练和不可训练的参数
print('Number of all parameters: {}'.format(sum(p.numel() for p in net.parameters())))
# 打印当前设备上模型占用的显存（GPU 内存）量, 单位为 GB
print('Memory usage before training: {}GB'.format(torch.cuda.memory_allocated(device)/1024**3))
# 打印模型的训练时间

print('Start training...')

best_valid_mae = float('inf')
# best_model_path = 'model/best_lstm_model.pth'
# mse_valid_list = []


# 训练模型
for epoch in range(epochs):
    net.train()
    train_mse = 0.0
    train_mae = 0.0
    train_r2 = 0.0
    for train_batch_x, train_batch_y in train_loader:
        # batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        if args.model_name == 'Transformer':
            dec_input = torch.zeros_like(train_batch_x).float()
            output = net(train_batch_x, x_mark_enc=None, x_dec=dec_input, x_mark_dec=None)
        else:
            output = net(train_batch_x)

        train_mse += mse_loss(output, train_batch_y).item()
        loss = mae_loss(output, train_batch_y)
        train_mae += loss.item()
        # train_r2 += metrics.r2_score(output.cpu().detach().numpy(), batch_y.cpu().detach().numpy())
        train_r2 += r2_score(output, train_batch_y)
        loss.backward()
        optimizer.step()
        

    train_mse /= len(train_loader)
    train_mae /= len(train_loader)
    train_r2 /= len(train_loader)
    if epoch == 10:
            print('Memory usage at epoch = 10: {}GB'.format(torch.cuda.memory_allocated(device)/1024**3))



    # Validation
    all_preds_val = []
    all_true_val = []
    net.eval()
    with torch.no_grad():
        valid_mse = 0.0
        valid_mae = 0.0
        valid_r2 = 0.0
        for val_batch_x, val_batch_y in val_loader:
            if args.model_name == 'Transformer':
                dec_input = torch.zeros_like(val_batch_x).float()
                val_output = net(val_batch_x, x_mark_enc=None, x_dec=dec_input, x_mark_dec=None)
            else:
                val_output = net(val_batch_x)
            # valid_mse += mse_loss(val_output, val_batch_y).item()
            # valid_mae += mae_loss(val_output, val_batch_y).item()
            # valid_r2 += metrics.r2_score(val_output.cpu(), val_batch_y.cpu()).item()
            # valid_r2 += r2_score(val_output, val_batch_y)
        # valid_mse /= len(val_loader)
        # valid_mae /= len(val_loader)
        # valid_r2 /= len(val_loader)   
            all_preds_val.append(val_output) 
            all_true_val.append(val_batch_y)
        all_preds_val = torch.cat(all_preds_val)
        all_true_val = torch.cat(all_true_val)
        valid_mse = mse_loss(all_preds_val, all_true_val).item()
        valid_mae = mean_absolute_error(all_preds_val, all_true_val).item()
        valid_r2 = r2_score(all_preds_val, all_true_val)
    
    if (epoch+1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], train_mse: {train_mse:.8f}, train_mae: {train_mae:.8f}, train_r2: {train_r2:.8f}, valid_mse: {valid_mse:.8f}, valid_mae: {valid_mae:.8f}, valid_r2: {valid_r2:.8f}')
        

    # Save the best model based on validation loss
    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        best_epoch = epoch
        # best_model_path = 'model/lstm_epoch{}.pth'.format(best_epoch + 1)
        # torch.save(net.state_dict(), best_model_path)
        torch.save(net.state_dict(), args.model_save_pth)


# Load the best model for testing
best_model_path = args.model_save_pth
net.load_state_dict(torch.load(best_model_path))
# criterion = nn.MSELoss().to(device)


# Testing the model
# 创建两个列表来存储预测值和真实值
all_preds_test = []
all_true_test = []

net.eval()
with torch.no_grad():
    nmse_test = 0.0
    nrmse_test = 0.0
    nmae_test = 0.0
    for test_batch_x, test_batch_y in test_loader:
        if args.model_name == 'Transformer':
            dec_input = torch.zeros_like(test_batch_x).float()
            pred = net(test_batch_x, x_mark_enc=None, x_dec=dec_input, x_mark_dec=None)
        else:
            pred = net(test_batch_x)
        # 将预测值和真实值添加到列表中
        all_preds_test.append(pred)
        all_true_test.append(test_batch_y)
        # mse_test += mse_loss(pred, test_batch_y).item()
        # mae_test += mae_loss(pred, test_batch_y).item()
    # mse_test /= len(test_loader)
    # rmse_test = np.sqrt(mse_test)
    # mae_test /= len(test_loader)

# 将列表转换为numpy数组
# all_preds = np.concatenate(all_preds)
all_preds_test = torch.cat(all_preds_test)
print('all_preds_test_shape: {}'.format(all_preds_test.shape))
# all_true = np.concatenate(all_true)
all_true_test = torch.cat(all_true_test)
print('all_true_test_shape: {}'.format(all_true_test.shape))


# 将all_preds_test, all_true_test反归一化为原始值
# if args.norm == 'minmax':
#     all_true_test_original = all_true_test * (max_value['Power'] - min_value['Power']) + min_value['Power']
#     all_preds_test_original = all_preds_test * (max_value['Power'] - min_value['Power']) + min_value['Power']
# if args.norm == 'meanstd':
#     all_true_test_original = all_true_test * std_value['Power'] + mean_value['Power']
#     all_preds_test_original = all_preds_test * std_value['Power'] + mean_value['Power']
if args.norm == 'minmax':
    all_true_test_original = all_true_test * (max_value['Power'] - min_value['Power']) + min_value['Power']
    all_preds_test_original = all_preds_test * (max_value['Power'] - min_value['Power']) + min_value['Power']
if args.norm == 'meanstd':
    all_true_test_original = all_true_test * std_value['Power'] + mean_value['Power']
    all_preds_test_original = all_preds_test * std_value['Power'] + mean_value['Power']

# 计算反归一化前的各项指标
nmse_test = mse_loss(all_preds_test, all_true_test).item()
nrmse_test = np.sqrt(nmse_test)
nmae_test = mean_absolute_error(all_preds_test, all_true_test).item()
nr2_test = r2_score(all_preds_test, all_true_test)
nsmape_test = smape(all_preds_test, all_true_test).item()


# 计算反归一化后的各项指标
mse_test = mse_loss(all_preds_test_original, all_true_test_original).item()
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(all_preds_test_original, all_true_test_original).item()
r2_test = r2_score(all_preds_test_original, all_true_test_original)
smape_test = smape(all_preds_test_original, all_true_test_original).item()


print('\nNMSE_test: {:.8f}  NRMSE_test: {:.8f}  NMAE_test: {:.8f}  NR2_test: {:.8f}  NSMAPE_test: {:.8f}'.format(nmse_test, nrmse_test, nmae_test, nr2_test, nsmape_test))
print('\nMSE_test: {:.8f}  RMSE_test: {:.8f}  MAE_test: {:.8f}  R2_test: {:.8f}  SMAPE_test: {:.8f}'.format(mse_test, rmse_test, mae_test, r2_test, smape_test))


# if args.model_name == 'PVformer':
#     # 打印PVformer的超参数,有args.dp1,args.dp2,args.D,args.P,args.S,args.r,args.num_layers
#     print('\nPVformer: dp1:{}, dp2:{}, D:{}, P:{}, S:{}, r:{}, num_layers:{}'.format(args.dp, args.dp, args.D, args.P, args.S, args.r, args.num_layers))

# elif args.model_name == 'PVformer2':
#     # 打印PVformer2的超参数,有args.dp1,args.dp2,args.D,args.P,args.S,args.r,args.num_layers
#     print('\nPVformer2: dp1:{}, dp2:{}, D:{}, P:{}, S:{}, r:{}, num_layers:{}\n'.format(args.dp, args.dp, args.D, args.P, args.S, args.r, args.num_layers))

# elif args.model_name == 'PVformer3':
#     # 打印PVformer2的超参数,有args.dp1,args.dp2,args.D,args.P,args.S,args.r,args.num_layers
#     print('\nPVformer3: dp1:{}, dp2:{}, D:{}, P:{}, S:{}, r:{}, num_layers:{}\n'.format(args.dp, args.dp, args.D, args.P, args.S, args.r, args.num_layers))

# elif args.model_name == 'PVformer4':
#     # 打印PVformer2的超参数,有args.dp1,args.dp2,args.D,args.P,args.S,args.r,args.num_layers
#     print('\nPVformer4: dp1:{}, dp2:{}, D:{}, P:{}, S:{}, r:{}, num_layers:{}\n'.format(args.dp, args.dp, args.D, args.P, args.S, args.r, args.num_layers))

# print('\nmodel_name:{}, dp1:{}, dp2:{}, D:{}, P:{}, S:{}, r:{}, num_layers:{}, norm_method:{}\n'.format(args.model_name, args.dp, args.dp, args.D, args.P, args.S, args.r, args.num_layers, args.norm))

# print('\n****************************************************************************\n')
# if args.plot_date is not None and 'timestamp' in data.columns:
#     try:
#         # ==================== 阶段1：参数准备与索引计算 ====================
#         plot_days = getattr(args, 'plot_days', 1)  # 默认绘制1天
#         print(f"\n{'='*60}")
#         if plot_days == 1:
#             print(f"🎨 开始绘图：{args.plot_date} ({args.weather_type})")
#         else:
#             print(f"🎨 开始绘制{plot_days}天图表：{args.plot_date} 起 ({args.weather_type})")
#         print(f"{'='*60}")
        
#         test_start_idx = split_index2
#         base_offset = args.input_size + args.y_step - 1
#         pred_indices = test_start_idx + np.arange(len(x_test)) + base_offset
        
#         # 边界检查：修正越界索引
#         if pred_indices[-1] >= len(data):
#             valid_count = len(data) - pred_indices[0]
#             pred_indices = pred_indices[:valid_count]
#             print(f"⚠️ 索引越界，修正后只保留 {len(pred_indices)} 个有效样本")
        
#         # 获取时间戳
#         test_timestamps = pd.to_datetime(data['timestamp'].iloc[pred_indices]).reset_index(drop=True)
#         print(f"📊 测试集时间范围: {test_timestamps.min()} 到 {test_timestamps.max()}")
        
#         # ==================== 阶段2：日期范围匹配 ====================
#         start_date = pd.to_datetime(args.plot_date).date()
#         end_date = start_date + pd.Timedelta(days=plot_days - 1)
        
#         if plot_days == 1:
#             date_mask = test_timestamps.dt.date == start_date
#             date_desc = f"日期 {start_date}"
#         else:
#             date_mask = (test_timestamps.dt.date >= start_date) & (test_timestamps.dt.date <= end_date)
#             date_desc = f"日期范围 {start_date} 至 {end_date}"
        
#         if date_mask.sum() == 0:
#             print(f"\n❌ 未找到 {date_desc} 的数据")
#             print(f"📅 测试集实际日期范围: {test_timestamps.min()} 到 {test_timestamps.max()}")
            
#             # 推荐可用日期
#             available_dates = test_timestamps.dt.date.unique()
#             print(f"\n💡 前10个可用日期:")
#             for i, d in enumerate(available_dates[:10], 1):
#                 print(f"   {i}. {d}")
#             raise ValueError("日期不存在，请从推荐列表中选择")
        
#         # ==================== 阶段3：数据准备 ====================
#         # 修复：使用 .cpu().numpy() 避免 Pandas 索引问题
#         preds_tensor = all_preds_test_original[date_mask].cpu()
#         true_tensor = all_true_test_original[date_mask].cpu()
#         times = test_timestamps[date_mask]
        
#         # 按时间排序
#         sorted_idx = times.argsort().values
#         preds = preds_tensor[sorted_idx]
#         true_vals = true_tensor[sorted_idx]
#         times_sorted = times.iloc[sorted_idx].reset_index(drop=True)
        
#         # 创建时间标签
#         if plot_days == 1:
#             time_labels = times_sorted.dt.strftime('%H:%M')
#         else:
#             time_labels = times_sorted.dt.strftime('%m-%d %H:%M')
        
#         print(f"✅ 成功提取 {len(times_sorted)} 个时刻点的数据")
#         if plot_days > 1:
#             print(f"📅 涵盖日期: {start_date} 至 {end_date}")
        
#     except Exception as e:
#         print(f"\n{'='*60}")
#         print(f"❌ 绘图时发生错误: {type(e).__name__}: {e}")
#         print(f"{'='*60}")
#         import traceback
#         traceback.print_exc()
#         print("\n⚠️ 程序继续运行，跳过绘图")
        
#     else:
#         # ==================== 阶段4：绘制图表 ====================
#         try:
#             # 动态调整图表大小
#             fig_width = 12 if plot_days == 1 else min(18, 6 + plot_days * 2)
#             fig_height = 6 if plot_days == 1 else 8
#             plt.figure(figsize=(fig_width, fig_height))
            
#             # 绘制曲线
#             marker_size = 6 if plot_days == 1 else 5
#             plt.plot(range(len(time_labels)), true_vals.numpy(), 
#                      label='True',
#                      color='#1f77b4', linewidth=2.5, 
#                      )
#             plt.plot(range(len(time_labels)), preds.numpy(), 
#                      label='Predict',  
#                      color='#ff7f0e', linewidth=2.5, alpha=0.8,
#                     )
            
#             # 多日图表增强
#             if plot_days > 1:
#                 # 添加每日分隔线
#                 unique_dates = times_sorted.dt.normalize().unique()
#                 for i, date in enumerate(unique_dates[1:], 1):
#                     day_start_idx = times_sorted[times_sorted.dt.normalize() == date].index[0]
#                     plt.axvline(x=day_start_idx, color='gray', linestyle='--', alpha=0.3, linewidth=1)
                
#                 # x轴标签优化
#                 n_points = len(time_labels)
#                 step = max(1, n_points // 20)  # 大约显示20个标签
#                 plt.xticks(range(0, n_points, step), 
#                           time_labels.iloc[::step], 
#                           rotation=45, ha='right')
#             else:
#                 # 单日图表x轴
#                 plt.xticks(range(len(time_labels)), time_labels, rotation=45, ha='right')
            
#             # 图表美化
#             plt.xlabel('Time', fontsize=12, fontweight='bold')
#             plt.ylabel('Power (MW)', fontsize=12, fontweight='bold')
            
#             # 标题
#             if plot_days == 1:
#                 title = f'{args.weather_type} - {start_date}'
#             else:
#                 title = f'{args.weather_type} - {start_date} to {end_date} ({plot_days} days)'
#             plt.title(title, fontsize=14, fontweight='bold')
            
#             plt.legend(fontsize=11, loc='best', framealpha=0.9)
#             plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
#             plt.tight_layout()
            
#             # 保存图片
#             filename_suffix = f"{start_date}_{plot_days}days_{args.output_size}length_{args.weather_type}.png" if plot_days > 1 else f"{start_date}_{args.output_size}length_{args.weather_type}.png"
#             output_filename = f'logs/pic2/forecast_{args.model_name}_{filename_suffix}'
#             plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
#             print(f"\n✅ 图表已保存至: {output_filename}")
            
#             # 尝试显示弹窗
#             try:
#                 plt.show(block=True)
#                 print("✅ 弹窗已显示")
#             except Exception as show_error:
#                 print(f"⚠️ 无法显示弹窗: {show_error}")
            
#         finally:
#             plt.close('all')
            
#         # ==================== 阶段5：输出数据表格与指标 ====================
#         # 数据表格
#         print(f"\n{'='*80}")
#         if plot_days == 1:
#             print(f"📅 日期: {start_date}  (天气类型: {args.weather_type})")
#             print(f"{'='*60}")
#             print(f"{'时刻':<10} {'真实功率(MW)':<15} {'预测功率(MW)':<15} {'绝对误差(MW)':<15}")
#             print(f"{'-'*60}")
#         else:
#             print(f"📅 日期范围: {start_date} 至 {end_date}  (天气类型: {args.weather_type})")
#             print(f"{'='*80}")
#             print(f"{'日期':<12} {'时刻':<10} {'真实功率(MW)':<15} {'预测功率(MW)':<15} {'绝对误差(MW)':<15}")
#             print(f"{'-'*80}")
        
#         for i in range(len(time_labels)):
#             if plot_days == 1:
#                 print(f"{time_labels.iloc[i]:<10} "
#                       f"{true_vals[i].item():>12.2f}     {preds[i].item():>12.2f}     {abs(true_vals[i].item() - preds[i].item()):>12.2f}")
#             else:
#                 date_str = times_sorted.iloc[i].strftime('%Y-%m-%d')
#                 time_str = times_sorted.iloc[i].strftime('%H:%M')
#                 true_val = true_vals[i].item()
#                 pred_val = preds[i].item()
#                 error = abs(true_val - pred_val)
#                 print(f"{date_str:<12} {time_str:<10} {true_val:>12.2f}     {pred_val:>12.2f}     {error:>12.2f}")

#     os.makedirs('logs/txt2', exist_ok=True)
#     filename_suffix = f"{start_date}_{args.model_name}_{plot_days}days_{args.output_size}length_{args.weather_type}.txt" if plot_days > 1 else f"{start_date}_{args.model_name}_{args.output_size}length_{args.weather_type}.txt"
#     output_txt_path = f'logs/txt2/forecast_data_{filename_suffix}'

#     with open(output_txt_path, 'w', encoding='utf-8') as f:
#         # 写入表头
#         if plot_days == 1:

#             f.write(f"{'时刻':<10} {'真实功率(MW)':<15} {'预测功率(MW)':<15}\n")

#         else:


#             f.write(f"{'日期':<12} {'时刻':<10} {'真实功率(MW)':<15} {'预测功率(MW)':<15}\n")

        
#         # 写入数据行
#         for i in range(len(time_labels)):
#             if plot_days == 1:
#                 f.write(f"{time_labels.iloc[i]:<10} "
#                     f"{true_vals[i].item():>12.2f}     {preds[i].item():>12.2f}\n")
#             else:
#                 date_str = times_sorted.iloc[i].strftime('%Y-%m-%d')
#                 time_str = times_sorted.iloc[i].strftime('%H:%M')
#                 true_val = true_vals[i].item()
#                 pred_val = preds[i].item()
#                 f.write(f"{date_str:<12} {time_str:<10} {true_val:>12.2f}     {pred_val:>12.2f}\n")

#         print(f"📄 数据表格已保存至: {output_txt_path}")        
#         # 计算整体指标
#         total_mae = mean_absolute_error(preds, true_vals).item()
#         total_rmse = np.sqrt(mse_loss(preds, true_vals).item())
#         total_r2 = r2_score(preds, true_vals)
#         total_nmae = total_mae / (max_value['Power'] - min_value['Power']) * 100
        
#         # 输出指标
#         if plot_days == 1:
#             print(f"{'-'*60}")
#             print(f"📈 当日评估指标:")
#             print(f"  NMAE: {total_nmae:.2f}%")
#             print(f"  MAE:  {total_mae:.2f} MW")
#             print(f"  RMSE: {total_rmse:.2f} MW")
#             print(f"  R²:   {total_r2:.4f}")
#             print(f"{'='*60}\n")
#         else:
#             # 计算逐日指标
#             print(f"\n📈 逐日评估指标:")
#             print(f"{'='*80}")
#             print(f"{'日期':<12} {'MAE (MW)':<12} {'RMSE (MW)':<12} {'R²':<8} {'NMAE (%)':<10}")
#             print(f"{'-'*80}")
            
#             unique_dates = times_sorted.dt.normalize().unique()
#             for date in unique_dates:
#                 day_mask = times_sorted.dt.normalize() == date
#                 day_preds = preds[day_mask]
#                 day_true = true_vals[day_mask]
                
#                 day_mae = mean_absolute_error(day_preds, day_true).item()
#                 day_rmse = np.sqrt(mse_loss(day_preds, day_true).item())
#                 day_r2 = r2_score(day_preds, day_true)
#                 day_nmae = day_mae / (max_value['Power'] - min_value['Power']) * 100
                
#                 print(f"{date.strftime('%Y-%m-%d'):<12} {day_mae:>10.2f}   {day_rmse:>10.2f}   {day_r2:>6.4f}   {day_nmae:>8.2f}")
            
#             print(f"{'='*80}")
#             print(f"📊 {plot_days}天整体评估指标:")
#             print(f"  NMAE: {total_nmae:.2f}%")
#             print(f"  MAE:  {total_mae:.2f} MW")
#             print(f"  RMSE: {total_rmse:.2f} MW")
#             print(f"  R²:   {total_r2:.4f}")
#             print(f"{'='*80}\n")
    



