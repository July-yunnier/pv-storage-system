import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


file_path = 'data/87-Site_DKA-M9_A+C-Phases.csv'
# data = pd.read_csv(file_path, header=0, low_memory=False, index_col=0)
data = pd.read_csv(file_path)
data = data.rename(columns={
    u'Active_Energy_Delivered_Received': 'AE_Power', #有用功
    u'Current_Phase_Average': 'Current', #电流
    u'Active_Power': 'Power',   #有用功率
    u'Wind_Speed': 'WS',   #风速
    u'Weather_Temperature_Celsius': 'Temp',    #气温
    u'Weather_Relative_Humidity': 'Humidity',    #湿度
    u'Global_Horizontal_Radiation': 'GHR',   #全球水平辐照度
    u'Diffuse_Horizontal_Radiation': 'DHR',   #扩散水平辐照度
    u'Wind_Direction': 'WD',  #风向
    u'Weather_Daily_Rainfall': 'Rainfall',   #降雨
    u'Radiation_Global_Tilted': 'RGT',   #全局倾斜辐照度
    u'Radiation_Diffuse_Tilted': 'RDT'   #扩散倾斜辐照度
})

data = data.drop(columns='AE_Power')
data = data.drop(columns='Current')

# feature = ['Current', 'Power', 'WS', 'Temp', 'Humidity', 'GHR', 'DHR', 'WD', 'Rainfall', 'RGT', 'RDT']
feature = ['Power', 'WS', 'Temp', 'Humidity', 'GHR', 'DHR', 'WD', 'Rainfall', 'RGT', 'RDT']
# 设定输入特征
# input_feature = ['Wind_speed', 'Humidity', 'Temp', 'GHI']
# input_feature_num = 4
# 设定目标特征
target_feature = ['Power']
# 判断功率是不是为0,仅保留功率不为0的值
# dataset = data[~data['Power'].isin([0])].dropna(axis=0)

# 删除功率为空的数据组
# data = data.dropna(subset=['Power'])


# 转换日期列的类型
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 生成期望的时间范围
expected_range = pd.date_range(start=data['timestamp'].min(), end=data['timestamp'].max(), freq='5min')
# 查找缺失的时间步
missing_steps = expected_range[~expected_range.isin(data['timestamp'])]
# 如果有时间步缺失,则补充缺失的时间步
if not missing_steps.empty:
    missing_df = pd.DataFrame({'timestamp': missing_steps})
    data = pd.concat([data, missing_df]).sort_values('timestamp').reset_index(drop=True)


# 检查功率值有无负数,对负数功率值用0代替
# 通过检测可知,Power,GHR,RGT,DHR,RDT,Humidity,Rainfall的最小值为0,WS的最小值为-7.422060e+01,Temp的最小值为-1.137545e+01,WD的最小值为-1.086476e+03
data.loc[data['Power'] < 0, 'Power'] = 0
data.loc[data['GHR'] < 0, 'GHR'] = 0
data.loc[data['RGT'] < 0, 'RGT'] = 0
data.loc[data['DHR'] < 0, 'DHR'] = 0
data.loc[data['RDT'] < 0, 'RDT'] = 0
data.loc[data['Humidity'] < 0, 'Humidity'] = 0
data.loc[data['WS'] < 0, 'WS'] = 0
data.loc[data['Rainfall'] < 0, 'Rainfall'] = 0


# 使用线性插值对所有列进行缺失值的补全
data.interpolate(method='linear', inplace=True)
# 重新设置索引(可选)
# data.reset_index(inplace=True)
# data.to_csv('data/dataset1.csv', index=False)


# 仅保留每天上午6点到晚上7点的数据,2014年到2017年
# 过滤日期范围
start_date = '2014-01-01 00:00:00'
end_date = '2017-12-31 23:55:00'
data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
# 过滤时间范围
data = data[(data['timestamp'].dt.hour >= 6) & (data['timestamp'].dt.hour < 19)]
# 原始的数据集输出shape,(227674, 11),说明有时间步缺失
print(data.shape)
data.to_csv('data/dataset.csv', index=False)

# 转换日期列的类型
data['timestamp'] = pd.to_datetime(data['timestamp'])
# 将时间戳列设置为索引
data.set_index('timestamp', inplace=True)
# 使用groupby和resample进行每小时平均
hourly_data = data.groupby(data.index.floor('H')).mean()
# 重新设置索引
hourly_data.reset_index(inplace=True)
# 输出shape,(227674, 11)
print(hourly_data.shape)
# 保存结果
hourly_data.to_csv('data/dataset_hour.csv', index=False)
# 处理后的dataset,dataset_hour没有缺失时间步,没有缺失值,没有NAN值,没有负数Power值



# # 归一化
# scaler = MinMaxScaler()
# data[feature] = scaler.fit_transform(data[feature].to_numpy())
# # print(data['Power'].describe())
# # 输出shape
# # print(data.shape)

# # 设置因素特征数据集
# # data_x = data[feature]
# # 设置目标特征数据集
# # data_y = data[target_feature]
# # data.to_csv('./dataset.csv')

# # 加上index=False,防止多出来一列index
# # 此时的dataset没有缺失时间步,没有缺失值,没有NAN值
# data.to_csv('data/dataset.csv', index=False)






