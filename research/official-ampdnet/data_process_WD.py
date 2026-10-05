import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


file_path = 'data/WD_offshore_f5.csv'
data = pd.read_csv(file_path)
# 删除data的第一列
data = data.drop(data.columns[0], axis=1)


feature = ['Atmos', 'Humidity', 'Cloud', 'WS10', 'WD10', 'Temp', 'Radiation', 'Rain', 'WS100', 'WD100', 'Power']


# 转换日期列的类型
data['timestamp'] = pd.to_datetime(data['timestamp'])

# 生成期望的时间范围,粒度为15分钟
# 2019年为365天,2020年为366天,整个数据集有731天,有731*24*4=70176个时间步
expected_range = pd.date_range(start=data['timestamp'].min(), end=data['timestamp'].max(), freq='15min')
# 查找缺失的时间步
missing_steps = expected_range[~expected_range.isin(data['timestamp'])]
# 如果有时间步缺失,则补充缺失的时间步
if not missing_steps.empty:
    missing_df = pd.DataFrame({'timestamp': missing_steps})
    data = pd.concat([data, missing_df]).sort_values('timestamp').reset_index(drop=True)


# 打印原始数据集data的feature对应的所有变量的最大值最小值
print('max_value of raw dataset:')
print(data[feature].max())
print('min_value of raw dataset:')
print(data[feature].min())

# 打印原始数据集的shape
print('data_original_shape:',data.shape)
# 统计数据集的'Power'有多少负值
data['Power'] = pd.to_numeric(data['Power'], errors='coerce')
print('Power<0:', data[data['Power'] < 0].shape[0])


# 检查数据集的各列有无缺失值
print('各列的缺失值数量: ')
print(data.isnull().sum())
# 计算数据集的各列的缺失率
print('各列的缺失率: ')
print(data.isnull().sum() / len(data))


# 检查功率值有无负数,对负数功率值用0代替
# data.loc[data['Power'] < 0, 'Power'] = 0

# 使用线性插值对所有列进行缺失值的补全
# data = data.interpolate(method='linear', limit_direction='both', axis=0)
# 检测数据集的各列有无'<NULL>',有的话对其进行线性插值代替
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].replace('<NULL>', np.nan)
        data[col] = data[col].astype(float)
        data[col] = data[col].interpolate(method='linear', limit_direction='both', axis=0)


# 对数据集的feature对应的所有变量进行检测,看是否有负数,有的话用线性插值代替
# for col in feature:
#     if data[col].min() < 0:
#         data.loc[data[col] < 0, col] = 0
#         data[col] = data[col].interpolate(method='linear', limit_direction='both', axis=0)

# 对数据集的除了温度以外的所有其余变量进行检测,看是否有负数,有的话先对其赋值为'NaN',再用线性插值将'NaN'去除
# data.loc[(data['Temp'] > 50) | (data['Temp'] < -10), 'Temp'] = np.nan
for col in ['Atmos', 'Humidity', 'Cloud', 'WS10', 'WD10', 'Temp', 'Radiation', 'Rain', 'WS100', 'WD100', 'Power']:
    # if data[col].min() < 0:
    #     data.loc[data[col] < 0, col] = np.nan
    # 对于'TSI', 'DNI', 'GHI'这3个变量,如果有取值大于2000的,则用'NaN'代替
    if col in ['WS10', 'WD10', 'WS100', 'WD100', 'Radiation', 'Rain']:
        data.loc[data[col] < 0, col] = np.nan
    # 对于'Temp'这个变量,如果有取值大于50或者小于-10的,则用'NaN'代替
    if col == 'Temp':
        data.loc[data[col] < 0, col] = np.nan
    # 对于'Atmos'这个变量,如果有取值大于1100或者小于400的,则用'NaN'代替
    if col == 'Atmos':
        data.loc[(data[col] > 120000) | (data[col] < 400), col] = np.nan
    # 对于'Humidity'这个变量,如果有取值大于100的,则用'NaN'代替
    if col == 'Humidity':
        data.loc[(data[col] > 100) | (data[col] < 0), col] = np.nan
    if col == 'Cloud':
        data.loc[(data[col] > 1) | (data[col] < 0), col] = np.nan
    # 对于'Power'这个变量,如果有取值大于100的,则用'NaN'代替
    if col == 'Power':
        data.loc[data[col] < 0, col] = np.nan
    
    data[col] = data[col].interpolate(method='linear', limit_direction='both', inplace=False)



# 仅保留每天上午6点到晚上7点的数据,一天13个小时
# 731*13*4=38012
# data2 = data[(data['timestamp'].dt.hour >= 6) & (data['timestamp'].dt.hour < 19)]
# # 打印数据集的shape
# print('data_processed_shape:',data2.shape)

# 对data2的feature对应的所有变量进行每小时的平均值计算
# data3 = data2.resample('H', on='timestamp').mean().reset_index(drop=False)
# data3 = data3.dropna()
# data3.to_csv('data/Solar7_processed_hour.csv', index=False)


# 打印data的feature对应的所有变量的最大值最小值
# print('max_value of processed dataset:')
# print(data2[feature].max())
# print('min_value of processed dataset:')
# print(data2[feature].min())


# 用皮尔逊相关系数计算除了第一列时间步以外的所有变量之间的相关性
# correlation = data2[feature].corr()
# print(correlation)
# # 画出相关性热力图
# plt.figure(figsize=(10, 10))
# plt.matshow(correlation, cmap='coolwarm', fignum=1)
# plt.colorbar()
# plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
# plt.yticks(range(len(correlation.columns)), correlation.columns)
# plt.savefig('pictures/correlation_heatmap.pdf')

# 检查数据集的各列有无缺失值
print('异常处理后各列的缺失值数量: ')
print(data.isnull().sum())
# 计算数据集的各列的缺失率
print('异常处理后各列的缺失率: ')
print(data.isnull().sum() / len(data))
print('data_processed_shape', data.shape)

# 打印处理后的数据集data的feature对应的所有变量的最大值最小值
print('max_value of processed dataset:')
print(data[feature].max())
print('min_value of processed dataset:')
print(data[feature].min())


# 打印各个变量之间的皮尔逊相关系数大小关系
correlation = data[feature].corr()
print('correlation:')
print(correlation)
# 按照降序排列，打印出'Power'与其他变量之间的相关性和对应的变量
print('correlation with Power:')
print(correlation['Power'].sort_values(ascending=False))




# 保存结果
data.to_csv('data/WD5_processed.csv', index=False)
# 处理后的dataset,dataset_hour没有缺失时间步,没有缺失值,没有NAN值,没有负数Power值

