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


# 设置随机种子
seed = 44
# seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True #将cudnn框架中的随机数生成器设为确定性模式
torch.backends.cudnn.benchmark = False #关闭CuDNN框架的自动寻找最优卷积算法的功能，以避免不同的算法对结果产生影响
np.random.seed(seed)




dataset = 'data/Solar2_processed_3.csv'
in_channels = 1
input_size = 39
output_size = 26
y_step = 1



# PV1数据集
# file_path = 'data/Solar1_processed.csv'
file_path = dataset
dtype_dict = {'TSI': np.float32, 'DNI': np.float32, 'GHI': np.float32, 
              'Temp': np.float32, 'Humidity': np.float32, 'Atmos': np.float32, 
              'Power': np.float32}
data = pd.read_csv(file_path, dtype=dtype_dict)
# data = pd.read_csv(file_path)
print('Solar_processed_shape:{}\n'.format(data.shape))


feature = ['Power', 'TSI', 'DNI', 'GHI', 'Temp', 'Humidity', 'Atmos']
feat_names = feature[:in_channels]


# max_value是data的前1000行的最大值, min_value是data的前1000行的最小值
train_len = int((len(data)-input_size-output_size+1-y_step+1)*0.6)+input_size+output_size-1+y_step-1
print('train_len:{}\n'.format(train_len))
max_value = data.head(train_len)[feature].max()
min_value = data.head(train_len)[feature].min()
print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))


# Normalize the dataset
data[feature] = (data[feature] - min_value) / (max_value - min_value)


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


# MAE的GPU版本
def mean_absolute_error(pred, target):
    return torch.abs(pred - target).mean()

# r2的GPU版本
def r2_score(pred, target):
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ss_res / ss_tot


data_x, data_y = create_dataset2(data, feat_names, input_size, output_size, y_step)
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


