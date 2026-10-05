# coding:utf-8

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import metrics
# from sklearn.metrics import mean_absolute_error # 评分 MAE 的计算函数
import models
import models2
import ModernTCN
import TSMixer
import PatchMixer
import iTrans_test
import PatchTST_test
from Transformer import myTrans
import utils
import config as cfg
# from data_sample import min_power, max_power



# Load the best model for testing
# best_model = models.SimpleMLP(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size)
# best_model = models.BiGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
# best_model = models.SimpleGRU(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
# best_model = models.BiLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
# best_model = models.SimpleLSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)


net = None
if cfg.model_name == 'MLP':
    net = models.SimpleMLP(input_size=cfg.input_size*cfg.in_channels, output_size=cfg.output_size)
elif cfg.model_name == 'CNN':
    net = models.CNN(input_size=cfg.input_size, in_channels=cfg.in_channels, output_size=cfg.output_size)
elif cfg.model_name == 'LSTM':
    net = models.SimpleLSTM(input_size=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'GRU':
    net = models.SimpleGRU(input_size=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'BiLSTM':
    net = models.BiLSTM(input_size=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'BiGRU':
    net = models.BiGRU(input_size=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'CNN-LSTM':
    net = models.CNNLSTM(in_channels=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'CNN-GRU':
    net = models.CNNGRU(in_channels=cfg.in_channels, hidden_size=cfg.hidden_size, output_size=cfg.output_size, num_layers=cfg.num_layers)
elif cfg.model_name == 'TCN':
    net = models.TCN(input_size=cfg.in_channels, output_size=cfg.output_size, num_channels=[32]*cfg.levels, kernel_size=cfg.kernel_size, dropout=cfg.dropout)
elif cfg.model_name == 'DLinear':
    net = models2.DLinear(input_size=cfg.input_size*cfg.in_channels, output_size=cfg.output_size)
elif cfg.model_name == 'Linear':
    net = models2.Linear(input_size=cfg.input_size*cfg.in_channels, output_size=cfg.output_size)
# elif cfg.model_name == 'NLinear':
#     net = models2.NLinear(input_size=cfg.input_size*cfg.in_channels, output_size=cfg.output_size)
elif cfg.model_name == 'ModernTCN':
    net = ModernTCN.ModernTCN(M=cfg.in_channels, L=cfg.input_size, T=cfg.output_size)
elif cfg.model_name == 'TSMixer':
    net = TSMixer.TSMixer(input_size=cfg.input_size, output_size=cfg.output_size, in_channels=cfg.in_channels)
elif cfg.model_name == 'PatchMixer':
    net = PatchMixer.PatchMixer(in_channels=cfg.in_channels, input_size=cfg.input_size, output_size=cfg.output_size, patch_len=5, stride=3)
elif cfg.model_name == 'iTransformer':
    net = iTrans_test.iTransformer(input_size=cfg.input_size, output_size=cfg.output_size, d_model=128, enc_in=cfg.in_channels, factor=5, dropout=0.01, output_attention=False, n_heads=4, d_ff=256, activation=F.gelu, e_layers=2)
elif cfg.model_name == 'PatchTST':
    net = PatchTST_test.PatchTST(input_size=cfg.input_size, output_size=cfg.output_size, d_model=128, dropout=0.01, factor=5, output_attention=False, n_heads=4, d_ff=256, activation=F.gelu, e_layers=5, enc_in=cfg.in_channels, patch_len=4, stride=2)
elif cfg.model_name == 'Transformer':
    net = myTrans





x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_data(f_x=cfg.f_x, f_y=cfg.f_y)
min_power = 0.0
max_power = 21.36063830
y_test = y_test * (max_power - min_power) + min_power

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# if cfg.model_name == 'CNN' or cfg.model_name == 'CNN-LSTM' or cfg.model_name == 'CNN-GRU' or cfg.model_name == 'ModernTCN':
#     x_test_tensor = x_test_tensor.permute(0,2,1)



best_model_path = cfg.model_save_pth
net.load_state_dict(torch.load(best_model_path))
criterion = nn.MSELoss()

# Testing the model
net.eval()
with torch.no_grad():
    test_loss = 0.0
    if cfg.model_name == 'Transformer':
        dec_input = torch.zeros_like(x_test_tensor).float()
        pred = net(x_test_tensor, x_mark_enc=None, x_dec=dec_input, x_mark_dec=None)
    else:
        pred = net(x_test_tensor)
    # pred = scaler.inverse_transform(pred)
    # 预测值反归一化
    pred = pred * (max_power - min_power) + min_power
    # y_test_tensor = y_test_tensor * (max_power - min_power) + min_power
    test_loss += criterion(pred, y_test_tensor).item()


# print(pred.shape) # 报错:'list' object has no attribute 'shape'
pred = np.array(pred).reshape(-1,1)
print(pred.shape) # pred的shape为(3796,1)
# y_test = y_test * (max_power - min_power) + min_power
print(y_test.shape) # y_test的shape为(3796,1)
# y_test = y_test_tensor.numpy().flatten()
# print(y_test.shape) # y_test的shape为(3796,)


# 画图函数,真实值和预测值之间的对比
def plot_real_and_predicted(y_test, pred, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Real')
    plt.plot(pred, label='Predicted')
    plt.title('Real vs Predicted_ModernTCN')
    plt.xlabel('Time(h)')
    plt.ylabel('Active Power(kW)')
    plt.legend()
    plt.savefig(filename, format='pdf')

week_length = 91
plot_real_and_predicted(y_test[10:week_length+1+10], pred[10:week_length+1+10], 'real_vs_predicted_ModernTCN.pdf')



# 4个评价指标
mse_test = metrics.mean_squared_error(y_test, pred)
rmse_test = np.sqrt(mse_test)
mae_test = metrics.mean_absolute_error(y_test, pred)
# mape_test = np.mean(np.abs((y_test - pred) / (y_test + 1e-8))) * 100
r2_test = metrics.r2_score(y_test, pred)

# print(f'Test Loss: {test_loss:.8f}')
# print('\nMSE_test: {:.8f}  RMSE_test: {:.8f}  MAE_test: {:.8f}  MAPE_test: {:.8f}  R2_test: {:.8f}\n'.format(mse_test, rmse_test, mae_test, mape_test, r2_test))
print('\nMSE_test: {:.8f}  RMSE_test: {:.8f}  MAE_test: {:.8f}  R2_test: {:.8f}\n'.format(mse_test, rmse_test, mae_test, r2_test))




