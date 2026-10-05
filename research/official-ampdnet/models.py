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


# MLP
# MLP的input_size为L*M=104*4=416
class SimpleMLP(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # nn.Linear将输入张量的最后一个维度进行线性映射
        # self.fc1 = nn.Linear(input_size, hidden_size*input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size*input_size, output_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # flatten操作的作用是将多维输入的除了第一维后面的其它所有维度展平
        x = self.flatten(x)
        # 第一层全连接层
        out = self.fc1(x)
        # 非线性激活函数(ReLU)
        out = self.relu(out)
        # 第二层全连接层
        out = self.fc2(out)
        return out



# 一维CNN
# Conv1d的输入的shape为(batchsize,特征数,时间窗口长度)
# nn.Conv1d的输入的shape是(B, C_in, L), 输出的shape是(B, C_out, L_out), 对输入的batch后面的第2个维度进行操作
class CNN(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, input_size, output_size, in_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64*input_size, output_size)
    
    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(x)
        x = self.relu(x)
        # 经过下面的view变换后,x的shape为(batchsize,Linear的输入特征数)
        x = x.view(x.size(0), -1)
        pred = self.fc1(x)
        return pred





# LSTM
# batch_first=True时,nn.LSTM输入的数据格式为(batch, seq, feature)
class SimpleLSTM(nn.Module):   # [B, L, M] -> [B, H]
    def __init__(self, input_size, hidden_size=16, num_layers=1, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM层
        output, _ = self.lstm(x) # [B, L, M] -> [B, L, hidden_size]
        # 取最后一个时间步的输出
        output = output[:, -1, :]   # [B, L, hidden_size] -> [B, hidden_size]
        # 全连接层
        pred = self.fc(output)  # [B, hidden_size] -> [B, output_size]
        return pred


# 基于自回归的LSTM
class SimpleLSTM2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, output_size=1, num_layers=1):
        super(SimpleLSTM2, self).__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True) # [B, L, M] -> [B, L, hidden_dim]
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        predictions = []
        h, c = None, None

        for _ in range(self.output_size):
            out, (h, c) = self.lstm(x, (h, c)) if h is not None else self.lstm(x)
            out = self.linear(out[:, -1, :])  # [B,L,hidden_dim] -> [B,hidden_dim] -> [B, output_dim]
            out = out.unsqueeze(1)  # final_output shape: [B, 1, M]
            predictions.append(out)

            # 使用当前预测作为下一步的输入
            x = out

        return torch.cat(predictions, dim=1)[:,:,0]  # [B, output_size, M]


# GRU
class SimpleGRU(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # GRU层
        output, _ = self.gru(x)  
        # 取最后一个时间步的输出
        output = output[:, -1, :]   
        # 全连接层
        pred = self.fc(output)   
        return pred


class SimpleGRU2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, output_size=1, num_layers=1):
        super(SimpleGRU2, self).__init__()
        self.output_size = output_size
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        predictions = []
        h, c = None, None

        for _ in range(self.output_size):
            out, (h, c) = self.gru(x, (h, c)) if h is not None else self.gru(x)
            out = self.linear(out[:, -1, :])  # [B,L,hidden_dim] -> [B,hidden_dim] -> [B, output_dim]
            out = out.unsqueeze(1)  # final_output shape: [B, 1, M]
            predictions.append(out)

            # 使用当前预测作为下一步的输入
            x = out

        return torch.cat(predictions, dim=1)  # [B, output_size, M]


# BiLSTM
class BiLSTM(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向的,所以乘以2

    def forward(self, x):
        # LSTM层
        out, _ = self.lstm(x)   
        # 取最后一个时间步的输出
        out = out[:, -1, :]    
        # 全连接层
        out = self.fc(out)   
        return out
    


# BiGRU
class BiGRU(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 因为是双向的,所以乘以2

    def forward(self, x):
        # GRU层
        out, _ = self.gru(x)    
        # 取最后一个时间步的输出
        out = out[:, -1, :]   
        # 全连接层
        out = self.fc(out)   
        return out


class CNNLSTM(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, in_channels, output_size, hidden_size=64, num_layers=1):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        # 这里相当于LSTM的输入的shape为(11388, 64, 13)
        # 经permute后变成(11388, 13, 64)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(x) # [B, M, L] -> [B, 64, L]
        x = self.relu1(x)
        x = self.conv2(x) # [B, 64, L] -> [B, M, L]
        
        x = x.permute(0,2,1) # [B, M, L] -> [B, L, M]
        # LSTM层
        out, _ = self.lstm(x)        
        # 取最后一个时间步的输出
        out = out[:, -1, :]       
        # 全连接层
        out = self.fc(out)       
        return out


class CNNGRU(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, in_channels, output_size, hidden_size=64, num_layers=1):
        super(CNNGRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=in_channels, kernel_size=3, stride=1, padding=1) 
        # 这里相当于LSTM的输入的shape为(11388, 64, 13)
        # 经permute后变成(11388, 13, 64)
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        
        # 将特征图转换为序列
        # x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0,2,1) # [B, M, L] -> [B, L, M]
        # LSTM层
        out, _ = self.gru(x)        
        # 取最后一个时间步的输出
        out = out[:, -1, :]       
        # 全连接层
        out = self.fc(out)       
        return out


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

class TCN(nn.Module): # [B, L, N] -> [B, output_size]
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        output = self.tcn(x.transpose(1, 2))
        output = output.transpose(1, 2)
        pred = self.linear(output[:, -1, :])
        return pred


# 测试TCN模型的输入输出的shape,TCN的输入为[B, L, N],输出为[B, H]
# input_size = 4
# output_size = 52
# num_channels = [64, 64, 64]
# kernel_size = 3
# dropout = 0.01
# tcn = TCN(input_size, output_size, num_channels, kernel_size, dropout)
# x = torch.randn(128, 156, 4)
# output = tcn(x)
# print(output.shape)  
# torch.Size([128, 52])


# CNN-LSTM-AM












