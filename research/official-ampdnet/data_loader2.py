import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time, datetime
from sklearn.preprocessing import MinMaxScaler
import torch



# Load the dataset
file_path = 'data/dataset_hour.csv'
data = pd.read_csv(file_path)
print('dataset_hour_shape:{}\n'.format(data.shape))


# feature = ['Power', 'WS', 'Temp', 'Humidity', 'GHR', 'DHR', 'WD', 'Rainfall', 'RGT', 'RDT']
feature = ['Power', 'GHR', 'RGT', 'DHR', 'RDT', 'Humidity', 'WS', 'Temp', 'WD', 'Rainfall']
feat_names = ['Power', 'GHR', 'RGT', 'DHR', 'RDT']


# 归一化数据集的时间步以外的所有变量
# 获得数据集的时间步2014-01-01 06:00:00到2016-12-31 18:00:00的所有数据,建立一个子数据集
data1 = data[(data['timestamp'] >= '2014-01-01 06:00:00') & (data['timestamp'] <= '2016-12-31 18:00:00')]
print('dataset1_shape:{}\n'.format(data1.shape))
# Get the maximum and minimum values for normalization
max_value = data1[feature].max()
min_value = data1[feature].min()
# print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))
# Normalize the dataset
data[feature] = (data[feature] - min_value) / (max_value - min_value)
# print('data:{}\n'.format(data))
print('data_shape:{}\n'.format(data.shape))


# 将子数据集data1去除掉得到data2
data2 = data.drop(data1.index)


# 用滑动窗口的方法来获得时间序列的样本
# Create the dataset with a sliding window
def create_dataset(dataset, feat_names, x_length, y_length, y_step):
    data_x, data_y = [], []
    for start_id in range(0, len(dataset)-x_length-y_length+1-y_step+1, 1):
        # iloc取切片操作是前闭后开
        data_x.append(data[feat_names].iloc[start_id: start_id+x_length])
        data_y.append(data['Power'].iloc[start_id+x_length+y_step-1: start_id+x_length+y_length+y_step-1])
    return np.array(data_x), np.array(data_y)

# 样本量: len(self.data) - self.x_length - self.y_length + 1 - self.y_step + 1


# 根据lookbacks,horizons,特征数来分割形成样本,这里生成的是训练集验证集的样本
# 仅预测 y_t+1
# data1_x, data1_y = create_dataset(data1, feat_names, 39, 1, 1)
# 直接多步预测 y_t+1 到 y_t+13
data1_x, data1_y = create_dataset(data1, feat_names, 39, 1, 6)
print('data1_x_shape: {}  data1_y_shape: {}'.format(data1_x.shape, data1_y.shape))


# 2014年到2016年的数据来得到训练集验证集
# 将data1_x和data1_y的前80%作为训练集,后20%作为验证集
# split_index = int(len(data1_x) * 0.8)
# x_train = data1_x[:split_index]
# y_train = data1_y[:split_index]
# x_val = data1_x[split_index:]
# y_val = data1_y[split_index:]
# print('x_train_shape: {}  y_train_shape: {}'.format(x_train.shape, y_train.shape))
# print('x_val_shape: {}  y_val_shape: {}'.format(x_val.shape, y_val.shape))


# 2014年到2016年的数据来得到训练集验证集
# 设置固定随机种子,每次运行都是选取一样的20%作为验证集
np.random.seed(44)
# 将data1_x和data1_y的随机80%作为训练集,剩余20%作为验证集
split_index = int(len(data1_x) * 0.8)
index = np.arange(len(data1_x))
np.random.shuffle(index)
x_train = data1_x[index[:split_index]]
y_train = data1_y[index[:split_index]]
x_valid = data1_x[index[split_index:]]
y_valid = data1_y[index[split_index:]]
print('x_train_shape: {}  y_train_shape: {}'.format(x_train.shape, y_train.shape))
print('x_valid_shape: {}  y_valid_shape: {}'.format(x_valid.shape, y_valid.shape))


# 2017年的数据来得到测试集
# 根据lookbacks,horizons,特征数来分割形成样本,这里生成的是测试集的样本
# 仅预测 y_t+1
# x_test, y_test = create_dataset(data2, feat_names, 39, 1, 1)
# 直接多步预测 y_t+1 到 y_t+13
x_test, y_test = create_dataset(data2, feat_names, 39, 1, 6)
print('x_test_shape: {}  y_test_shape: {}'.format(x_test.shape, y_test.shape))

