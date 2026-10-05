import pandas as pd




# PV1数据集
file_path = 'data/Solar7_processed.csv'
# file_path = args.dataset
data = pd.read_csv(file_path)
print('Solar_processed_shape:{}\n'.format(data.shape))
feature = ['Power', 'TSI', 'DNI', 'GHI', 'Temp', 'Humidity', 'Atmos']




# 得到data数据集的时间步2019-01-01 06:00:00到2020-06-30 18:45:00的feature列对应的最大值和最小值
max_value = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:45:00')][feature].max()
min_value = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:45:00')][feature].min()
print('max_value:{}\nmin_value:{}\n'.format(max_value, min_value))
# Normalize the dataset
data[feature] = (data[feature] - min_value) / (max_value - min_value)
data.to_csv('data/Solar7_normalized.csv', index=False)
data1 = data[(data['timestamp'] >= '2019-01-01 06:00:00') & (data['timestamp'] <= '2020-06-30 18:45:00')]
print('dataset1_shape:{}\n'.format(data1.shape))
data2 = data.drop(data1.index)
print('dataset2_shape:{}\n'.format(data2.shape))





