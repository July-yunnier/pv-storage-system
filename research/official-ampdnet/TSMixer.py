# TSMixer: An All-MLP Architecture for Time Series Forecasting 2023
import torch
import torch.nn as nn
import torch.fft


class Mlp_feat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_feat, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x): # [B, L, M] -> [B, L, M]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_time, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x): # [B, M, L] -> [B, M, L]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Mixer_Layer(nn.Module): # [B, L, M] -> [B, L, M]
    def __init__(self, time_dim, feat_dim):
        super(Mixer_Layer, self).__init__()

        # nn.BatchNorm: axis is Integer
        # nn.LayerNorm: axis is Integer or List[Integer]
        # given [B, L, D] nn.BatchNorm1d(B) is equal to nn.LayerNorm([L, D])

        # self.batchNorm2D = nn.LayerNorm([time_dim, feat_dim]) # the norm of the paper, seems bad
        # self.bn1 = nn.BatchNorm1d(time_dim)
        # self.bn2 = nn.BatchNorm1d(feat_dim)
        self.MLP_time = Mlp_time(time_dim, time_dim)
        self.MLP_feat = Mlp_feat(feat_dim, feat_dim)

    def forward(self, x): # [B, L, M] -> [B, L, M]
        res1 = x
        # x = self.bn1(x)
        x = self.MLP_time(x.permute(0, 2, 1)).permute(0, 2, 1) # [B, L, M] -> [B, M, L] -> [B, M, L] -> [B, L, M]
        x = x + res1

        res2 = x
        # x = self.bn2(x)
        x = self.MLP_feat(x) # [B, L, M] -> [B, L, M]
        x = x + res2
        return x

class TSMixer(nn.Module):
    def __init__(self, input_size, output_size, in_channels, layer_num = 3):
        super(TSMixer, self).__init__()

        self.layer_num = layer_num
        self.flatten = nn.Flatten()
        self.backbone = nn.ModuleList([Mixer_Layer(input_size, in_channels) for _ in range(layer_num)])
        # self.mix_layer = Mixer_Layer(input_size, in_channels)
        # self.temp_proj = nn.Linear(input_size*in_channels, output_size)
        self.temp_proj = nn.Linear(input_size, output_size)

    def forward(self, x): # [B, L, M] -> [B, H]
        # x = self.backbone(x) # [B, L, M] -> [B, L, M]
        # x = self.temp_proj(x.permute(0, 2, 1)).permute(0, 2, 1) # [B, L, M] -> [B, H, M] -> [B, M, H]
        for i in range(self.layer_num):
            x = self.backbone[i](x) # [B, L, M] -> [B, L, M]
        # x = self.flatten(x)
        x = x.permute(0, 2, 1) # [B, L, M] -> [B, M, L]
        x = self.temp_proj(x) # [B, M, L] -> [B, M, H]
        x = x[:,0,:] # [B, M, H] -> [B, H]
        return x



# TSMixer使用了两个MLP,一个MLP对时间维度进行处理,一个MLP对变量维度进行处理
# 测试TSMixer的输入输出
# input = torch.randn(128, 104, 4)
# model = TSMixer(104, 16, 4)
# output = model(input)
# print(output.shape) 
# torch.Size([128, 16])


