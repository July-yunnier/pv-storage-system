# 工具箱
# coding:utf-8

import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing


def save_pickle(filename, data):
    with open(filename, 'wb') as f:
        # 将一个字典对象data序列化为bytes类型，然后将其存储到文件filename中
        pickle.dump(data, f)


def load_pickle(filename):
    data = None
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


# 


# 获取训练集、验证集、测试集的index
# 前80%作为训练集,中间20%作为验证集,最后20%作为测试集
# x_shape: (18980, 13, 3)  y_shape: (18980,)
# 18980*0.6=11388, 18980*0.2=3796
# def get_ids_for_tvt():
def load_data(f_x, f_y):
    x = load_pickle(f_x)
    y = load_pickle(f_y)
    # HLL：在列方向上增加新维度,使得y的shape由(8736,)变成(8736,1)
    y = np.array(y[:, np.newaxis])

    # 确定划分比例
    train_ratio = 0.6
    valid_ratio = 0.2
    test_ratio = 0.2
    # 计算划分的索引
    num_samples = x.shape[0]
    num_train = int(train_ratio * num_samples)
    num_valid = int(valid_ratio * num_samples)
    # 划分数据集
    x_train = x[:num_train, :]
    y_train = y[:num_train, :]
    x_valid = x[num_train:num_train + num_valid, :]
    y_valid = y[num_train:num_train + num_valid, :]
    x_test = x[num_train + num_valid:, :]
    y_test = y[num_train + num_valid:, :]

    print('x_shape: {}  y_shape: {}\nx_train_shape: {}  y_train_shape: {}  x_valid_shape: {}  y_valid_shape: {}  x_test_shape: {}  y_test_shape: {}\n'
          .format(x.shape, y.shape, x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))
    return x_train, y_train, x_valid, y_valid, x_test, y_test


# numel()函数：返回数组中元素的个数
def get_param_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


