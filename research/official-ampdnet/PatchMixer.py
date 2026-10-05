import torch
import torch.nn as nn
import torch.fft



# 此MLP即为Transformer中常用的FFN
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# PatchMixer的d_model设置为patch_len * patch_len,比较局限,后面可以按照自己的想法进行改造
# PatchMixer本意上是通道共享的,不区分通道,后面可以自己改造
class Backbone(nn.Module): # [B, L, M] -> [B, H]
    def __init__(self, in_channels, input_size, output_size, patch_len, stride, padding_patch = 'end'):
        super(Backbone, self).__init__()

        # Patching
        # self.d_model = d_model # 64
        self.patch_len = patch_len # 16
        self.stride = stride  # 8
        self.patch_num = patch_num = int((input_size - patch_len) / stride + 1)
        self.padding_patch = padding_patch
        if self.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num = patch_num = patch_num + 1

        # 1
        d_model = patch_len * patch_len
        self.embed = nn.Linear(patch_len, d_model)
        self.dropout_embed = nn.Dropout(0.01)

        # 2
        # self.lin_res = nn.Linear(seq_len, pred_len) # direct res, seems bad
        self.lin_res = nn.Linear(patch_num * d_model * in_channels, output_size)
        self.dropout_res = nn.Dropout(0.01)

        # 3.1
        self.depth_conv = nn.Conv1d(patch_num, patch_num, kernel_size=patch_len, stride=patch_len, groups=patch_num) 
        self.depth_activation = nn.GELU()
        self.depth_norm = nn.BatchNorm1d(patch_num)
        self.depth_res = nn.Linear(d_model, patch_len)

        # 3.2
        # self.point_conv = nn.Conv1d(patch_len,patch_len,kernel_size=1, stride=1)
        # self.point_activation = nn.GELU()
        # self.point_norm = nn.BatchNorm1d(patch_len)
        self.point_conv = nn.Conv1d(patch_num, patch_num, kernel_size=1, stride=1)
        self.point_activation = nn.GELU()
        self.point_norm = nn.BatchNorm1d(patch_num)

        # 4
        self.flatten = nn.Flatten()
        self.mlp = Mlp(patch_len * patch_num * in_channels, 256, output_size)

    def forward(self, x): # [B, L, M] -> [B, H]
        B, _, M = x.shape # B为batch_size, L为时间维度, M为变量维度
        N = self.patch_num # N = (L - P) / S + 1, N为patch的数量
        P = self.patch_len # P为patch的长度

        # z_res = self.lin_res(x.permute(0, 2, 1)) # [B, L, M] -> [B, H, M]
        # z_res = self.dropout_res(z_res)

        # 1
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(x.permute(0, 2, 1))  # [B, L, M] -> [B, M, L] -> [B, M, L2], L2为在尾部填充后的时间维度长度
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B, M, L2] -> [B, M, N, P], N为patch的数量
        z = z.reshape(B * M, N, P, 1).squeeze(-1) # [B, M, N, P] -> [B * M, N, P]
        z = self.embed(z) # [B*M, N, P] -> [B*M, N, d_model]
        z = self.dropout_embed(z)

        # 2
        z_res = self.lin_res(z.reshape(B, -1)) # [B*M, N, d_model] -> [B, M*N*d_model] -> [B, H]
        z_res = self.dropout_res(z_res)

        # 3.1
        res = self.depth_res(z) # [B*M, N, d_model] -> [B*M, N, P]
        z_depth = self.depth_conv(z) # [B*M, N, d_model] -> [B*M, N, P],本文的d_model设置为 patch_len * patch_len
        z_depth = self.depth_activation(z_depth)
        z_depth = self.depth_norm(z_depth)
        z_depth = z_depth + res
        # 3.2
        z_point = self.point_conv(z_depth) # [B*M, N, P] -> [B*M, N, P]
        z_point = self.point_activation(z_point)
        z_point = self.point_norm(z_point)
        z_point = z_point.reshape(B, M, -1) # [B*M, N, P] -> [B, M, N*P]

        # 4
        z_point = self.flatten(z_point)
        z_mlp = self.mlp(z_point) # [B, M*N*P] -> [B, H]

        return (z_res + z_mlp)   # [B, H]


class PatchMixer(nn.Module): # [B, L, M] -> [B, H]

    def __init__(self, in_channels, input_size, output_size, patch_len, stride, padding_patch = 'end'):
        super(PatchMixer, self).__init__()

        self.backbone = Backbone(in_channels, input_size, output_size, patch_len, stride, padding_patch)


    def forward(self, x):
        x = self.backbone(x) # [B, L, M] -> [B, H]
        return x



# 测试PatchMixer的输入输出
# input = torch.randn(128, 104, 4)
# model = PatchMixer(4, 104, 16, 4, 2)
# output = model(input)
# print(output.shape) 
# torch.Size([128, 16])

