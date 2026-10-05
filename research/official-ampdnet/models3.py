# 这里是各个模型baseline
# 具体有 MLP,CNN,LSTM,GRU,BiLSTM,BiGRU,CNN-LSTM,CNN-GRU,TCN,LSTM-AM,CNN-LSTM-AM,
# coding:utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import weight_norm




# LSTM
# batch_first=True时,nn.LSTM输入的数据格式为(batch, seq, feature)
class SimpleLSTM2(nn.Module): # [B, L, M] -> [B, 1, M]
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self, input, hidden_cell):
        # hidden_cell = (torch.zeros(self.num_layers, input.size(0), self.hidden_dim), torch.zeros(self.num_layers, input.size(0), self.hidden_dim))
        out, hidden_cell = self.lstm(input, hidden_cell) # [B, L, M] -> [B, L, hidden_dim]
        output = self.linear(out[:,-1,:]) # [B, L, hidden_dim] -> [B, 1, hidden_dim] -> [B, output_dim]
        output = output.unsqueeze(1)  # [B, M] -> [B, 1, M]
        return output, hidden_cell


class SimpleGRU2(nn.Module): # [B, L, M] -> [B, 1, M]
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

    # GRU的前向传播
    def forward(self, input, hidden_cell):
        # hidden_cell = torch.zeros(self.num_layers, input.size(0), self.hidden_dim)
        out, hidden_cell = self.gru(input, hidden_cell) # [B, L, M] -> [B, L, hidden_dim]
        output = self.linear(out[:,-1,:]) # [B, L, hidden_dim] -> [B, hidden_dim] -> [B, output_dim]
        output = output.unsqueeze(1)  # [B, M] -> [B, 1, M]
        return output, hidden_cell


class CNNLSTM2(nn.Module): # [B, L, M] -> [B, 1, M]
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=8*input_dim, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=8*input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self, input, hidden_cell):
        x = input.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(x) # [B, M, L] -> [B, 64, L]
        x = self.relu1(x)
        x = self.conv2(x) # [B, 64, L] -> [B, M, L]
        x = x.permute(0,2,1) # [B, M, L] -> [B, L, M]
        # hidden_cell = (torch.zeros(self.num_layers, input.size(0), self.hidden_dim), torch.zeros(self.num_layers, input.size(0), self.hidden_dim))
        out, hidden_cell = self.lstm(x, hidden_cell) # [B, L, M] -> [B, L, hidden_dim]
        output = self.linear(out[:,-1,:]) # [B, L, hidden_dim] -> [B, hidden_dim] -> [B, output_dim]
        output = output.unsqueeze(1)  # [B, M] -> [B, 1, M]
        return output, hidden_cell


class CNNLSTM3(nn.Module): # [B, L, M] -> [B, 1, M]
    def __init__(self, input_size, input_dim=4, hidden_dim=64, output_dim=4, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=8*input_size, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=8*input_size, out_channels=input_size, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self, input, hidden_cell):
        # x = input.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(input) # [B, L, M] -> [B, 64*L, M]
        x = self.relu1(x)
        x = self.conv2(x) # [B, 64*L, M] -> [B, L, >]
        # x = x.permute(0,2,1) # [B, M, L] -> [B, L, M]
        # hidden_cell = (torch.zeros(self.num_layers, input.size(0), self.hidden_dim), torch.zeros(self.num_layers, input.size(0), self.hidden_dim))
        out, hidden_cell = self.lstm(x, hidden_cell) # [B, L, M] -> [B, L, hidden_dim]
        output = self.linear(out[:,-1,:]) # [B, L, hidden_dim] -> [B, hidden_dim] -> [B, output_dim]
        output = output.unsqueeze(1)  # [B, M] -> [B, 1, M]
        return output, hidden_cell


class CNNGRU2(nn.Module): # [B, L, M] -> [B, 1, M]
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=8*input_dim, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=8*input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

        # self.hidden_cell = (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self, input, hidden_cell):
        x = input.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(x) # [B, M, L] -> [B, 8*M, L]
        x = self.relu1(x)
        x = self.conv2(x) # [B, 8*M, L] -> [B, M, L]
        x = x.permute(0,2,1) # [B, M, L] -> [B, L, M]
        # hidden_cell = (torch.zeros(self.num_layers, input.size(0), self.hidden_dim), torch.zeros(self.num_layers, input.size(0), self.hidden_dim))
        out, hidden_cell = self.gru(x, hidden_cell) # [B, L, M] -> [B, L, hidden_dim]
        output = self.linear(out[:,-1,:]) # [B, L, hidden_dim] -> [B, hidden_dim] -> [B, output_dim]
        output = output.unsqueeze(1)  # [B, M] -> [B, 1, M]
        return output, hidden_cell


# TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         """
#         构成TCN的核心Block, 原作者在图中成为Residual block, 是因为它存在残差连接.
#         但注意, 这个模块包含了2个Conv1d.

#         :param n_inputs: int, 输入通道数或者特征数
#         :param n_outputs: int, 输出通道数或者特征数
#         :param kernel_size: int, 卷积核尺寸
#         :param stride: int, 步长, 在TCN固定为1
#         :param dilation: int, 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系. 
#                                 例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
#                                       如果这个Residual block在第2层, dilation = 2**1 = 2;
#                                       如果这个Residual block在第3层, dilation = 2**2 = 4;
#                                       如果这个Residual block在第4层, dilation = 2**3 = 8 ......
#         :param padding: int, 填充系数. 与kernel_size和dilation有关. 
#         :param dropout: float, dropout比率
#         """
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))

#         # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
#         self.chomp1 = Chomp1d(padding)
#         self.relu1 = nn.ReLU()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
#                                            stride=stride, padding=padding, dilation=dilation))
#         self.chomp2 = Chomp1d(padding)
#         self.relu2 = nn.ReLU()
#         self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)

#         # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
#         # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
#         self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

#         # 在整个Residual block中有非线性的激活. 这个容易忽略!
#         self.relu = nn.ReLU()
#         self.init_weights()

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

#     def forward(self, x):
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
#         return self.relu(out + res)

# class TemporalConvNet(nn.Module):
#     def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
#         super(TemporalConvNet, self).__init__()
#         """
#         :param num_inputs: int,  输入通道数或者特征数
#         :param num_channels: list, 每层的hidden_channel数. 例如[5,12,3], 代表有3个block, 
#                                 block1的输出channel数量为5; 
#                                 block2的输出channel数量为12;
#                                 block3的输出channel数量为3.
#         :param kernel_size: int, 卷积核尺寸
#         :param dropout: float, drop_out比率
#         """
#         layers = []
#         num_levels = len(num_channels)
#         # 可见,如果num_channels=[5,12,3],那么
#         # block1的dilation_size=1
#         # block2的dilation_size=2
#         # block3的dilation_size=4
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = num_inputs if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
#                                      padding=(kernel_size-1) * dilation_size, dropout=dropout)]

#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         输入x的结构不同于RNN,一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels),
#         这里把seq_len放在channels后面,把所有时间步的数据拼起来,当做Conv1d的输入尺寸,实现卷积跨时间步的操作,
#         很巧妙的设计
        
#         :param x: size of (Batch, input_channel, seq_len)
#         :return: size of (Batch, output_channel, seq_len)
#         """
#         return self.network(x)

# class TCN(nn.Module):
#     def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
#         super(TCN, self).__init__()
#         self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
#         self.linear = nn.Linear(num_channels[-1], output_size)

#     def forward(self, x):
#         output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
#         pred = self.linear(output[:, -1, :])
#         return pred

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """
        构成TCN的核心Block, 原作者在图中称为Residual block, 是因为它存在残差连接.
        但注意, 这个模块包含了2个Conv1d.

        :param n_inputs: int, 输入通道数或者特征数
        :param n_outputs: int, 输出通道数或者特征数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长, 在TCN固定为1
        :param dilation: int, 膨胀系数. 与这个Residual block(或者说, 隐藏层)所在的层数有关系. 
                                例如, 如果这个Residual block在第1层, dilation = 2**0 = 1;
                                      如果这个Residual block在第2层, dilation = 2**1 = 2;
                                      如果这个Residual block在第3层, dilation = 2**2 = 4;
                                      如果这个Residual block在第4层, dilation = 2**3 = 8 ......
        :param padding: int, 填充系数. 与kernel_size和dilation有关. 
        :param dropout: float, dropout比率
        """
        # 即所谓的因果卷积？
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 因为 padding 的时候, 在序列的左边和右边都有填充, 所以要裁剪
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 1×1的卷积. 只有在进入Residual block的通道数与出Residual block的通道数不一样时使用.
        # 一般都会不一样, 除非num_channels这个里面的数, 与num_inputs相等. 例如[5,5,5], 并且num_inputs也是5
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        # 在整个Residual block中有非线性的激活. 这个容易忽略!
        # 下面这两行代码什么意思？
        self.relu = nn.ReLU()
        self.init_weights()

    # 对每个block的两个卷积层和残差连接层的卷积核参数初始化
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # 网络的输出加上残差
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        """
        :param num_inputs: int,  输入通道数或者特征数
        :param num_channels: list, 每层的hidden_channel数. 例如[5,12,3], 代表有3个block, 
                                block1的输出channel数量为5; 
                                block2的输出channel数量为12;
                                block3的输出channel数量为3.
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        layers = []
        # TCN的层数
        num_levels = len(num_channels)
        # 可见，如果num_channels=[5,12,3]，那么
        # block1的dilation_size=1
        # block2的dilation_size=2
        # block3的dilation_size=4
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN2(nn.Module): # [B, L, N] -> [B, 1, output_dim]
    def __init__(self, input_dim, output_dim, num_channels=[32,16,4], kernel_size=4, dropout=0.05):
        super(TCN2, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, L, M] -> [B, M, L]
        output = self.tcn(x) # [B, M, L] -> [B, num_channels[-1], L]=[B, 4, L]
        output = output.permute(0, 2, 1) # [B, 4, L] -> [B, L, 4]
        pred = self.linear(output[:, -1, :]) # [B, L, 4] -> [B, 4] -> [B, output_dim]
        pred = pred.unsqueeze(1)  # [B, output_dim] -> [B, 1, output_dim]
        return pred


class TCN3(nn.Module): # [B, L, N] -> [B, 1, output_dim]
    def __init__(self, input_dim, output_dim, num_channels=[64], kernel_size=4, dropout=0.05):
        super(TCN3, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, L, M] -> [B, M, L]
        output = self.tcn(x) # [B, M, L] -> [B, num_channels[-1], L]=[B, 64, L]
        output = output.permute(0, 2, 1) # [B, 64, L] -> [B, L, 64]
        pred = self.linear(output[:, -1, :]) # [B, L, 64] -> [B, 64] -> [B, output_dim]
        pred = pred.unsqueeze(1)  # [B, output_dim] -> [B, 1, output_dim]
        return pred


# 测试CNNLSTM模型
# input = torch.randn(128, 104, 4)
# hidden_cell = (torch.zeros(1, input.size(0), 64), torch.zeros(1, input.size(0), 64))
# net = CNNLSTM2(input_dim=4, hidden_dim=64, output_dim=4, num_layers=1)
# output, hidden_cell = net(input, hidden_cell)
# print(output.shape)  
# # torch.Size([128, 1, 4]) 


# 测试CNNGRU2模型
# input = torch.randn(128, 104, 4)
# hidden_cell = torch.zeros(1, input.size(0), 64)
# net = CNNGRU2(input_dim=4, hidden_dim=64, output_dim=4, num_layers=1)
# output, hidden_cell = net(input, hidden_cell)
# print(output.shape)
# # torch.Size([128, 1, 4])



# 测试TCN模型的输入输出的shape,TCN的输入为[B, L, N],输出为[B, H]
# input_dim = 4
# output_dim = 4
# num_channels = [64, 64, 64]
# kernel_size = 3
# dropout = 0.01
# tcn = TCN(input_dim, output_dim, num_channels, kernel_size, dropout)
# x = torch.randn(128, 104, 4)
# output = tcn(x)
# print(output.shape)  
# torch.Size([128, 1, 4])


# CNN-LSTM-AM












