import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# 此代码非论文官方实现的代码,是的泼墨佛给克呢的知乎代码
# B：batch size
# M：多变量序列的变量数
# L：过去序列的长度
# T: 预测序列的长度
# N: 分Patch后Patch的个数
# D：每个变量的通道数/特征数,这里的通道数/特征数不是指变量数
# P：kernel size of embedding layer,即patch化中的卷积核大小
# S：stride of embedding layer

class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1, 
            out_channels=D, 
            kernel_size=P, 
            stride=S
            )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
            )  # [B*M, 1, L] -> [B*M, 1, L+P-S]
        
        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]

        return x_emb  # x_emb: [B, M, D, N]


class MultiDWConv(nn.Module):
    def __init__(self, M, D, kernel_sizes=[5,51]):
        super(MultiDWConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                padding='same'
                ) for kernel_size in kernel_sizes
            ])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [conv(x) for conv in self.convs]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


class MultiDWConvBN(nn.Module):
    def __init__(self, M, D, kernel_sizes=[5,51]):
        super(MultiDWConvBN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=M*D, 
                out_channels=M*D, 
                kernel_size=kernel_size, 
                groups=M*D, 
                padding='same'
                ) for kernel_size in kernel_sizes
            ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(M*D) for _ in kernel_sizes])

    def forward(self, x):
        # x: [B, M*D, N]
        x = [bn(conv(x)) for conv, bn in zip(self.convs, self.bns)]  # [B, M*D, N] -> [B, M*D, N] * len(kernel_sizes)
        x = torch.stack(x, dim=1)  # [B, M*D, N] * len(kernel_sizes) -> [B, len(kernel_sizes), M*D, N]
        x = x.sum(dim=1)  # [B, len(kernel_sizes), M*D, N] -> [B, M*D, N]
        return x  # x: [B, M*D, N]


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, dropout=0, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = M if one else D
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D, 
            out_channels=r*M*D, 
            kernel_size=1,
            groups=groups_num
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=r*M*D, 
            out_channels=M*D, 
            kernel_size=1,
            groups=groups_num
            )

    def forward(self, x):
        # x: [B, M*D, N]
        # 先对M维度做逐点卷积,再对D维度做逐点卷积
        # x = self.pw_con2(F.gelu(self.pw_con1(x)))
        x = self.pw_con1(x)
        x = self.dropout1(x)
        x = F.gelu(x)
        x = self.pw_con2(x)
        x = self.dropout2(x)
        return x  # x: [B, M*D, N]


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 大核逐深度可分离卷积负责捕获时域关系,卷积核大小为51
        self.dw_conv = nn.Conv1d(
            in_channels=M*D, 
            out_channels=M*D, 
            kernel_size=kernel_size,
            groups=M*D,
            padding='same'
            )  
        self.bn = nn.BatchNorm1d(M*D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x)                                   # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_ffn1(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_ffn2(x)                                 # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]


class ModernTCNBlock2(nn.Module):
    def __init__(self, M, D, kernel_sizes, r):
        super(ModernTCNBlock2, self).__init__()
        # 大核逐深度可分离卷积负责捕获时域关系,卷积核大小为51
        # self.dw_conv = MultiDWConvBN(M, D, kernel_sizes) 
        self.dw_conv = MultiDWConv(M, D, kernel_sizes) 
        self.bn = nn.BatchNorm1d(M*D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N]
        D = x_emb.shape[-2]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')          # [B, M, D, N] -> [B, M*D, N]
        x = self.dw_conv(x)                                   # [B, M*D, N] -> [B, M*D, N]
        x = self.bn(x)                                        # [B, M*D, N] -> [B, M*D, N]
        # 变量维度M的逐点卷积ConvFFN1
        x = self.conv_ffn1(x)                                 # [B, M*D, N] -> [B, M*D, N]

        x = rearrange(x, 'b (m d) n -> b m d n', d=D)         # [B, M*D, N] -> [B, M, D, N]
        x = x.permute(0,2,1,3)                                # [B, M, D, N] -> [B, D, M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')              # [B, D, M, N] -> [B, D*M, N]
        
        # 特征维度D的逐点卷积ConvFFN2
        x = self.conv_ffn2(x)                                 # [B, D*M, N] -> [B, D*M, N]

        x = rearrange(x, 'b (d m) n -> b d m n', d=D)         # [B, D*M, N] -> [B, D, M, N]
        x = x.permute(0,2,1,3)                                # [B, D, M, N] -> [B, M, D, N]

        out = x + x_emb

        return out  # out: [B, M, D, N]


class ModernTCN(nn.Module):
    # 在顶会论文中, kernel_size=51, 这里我改成了5, 原来r=8,num_layers=3
    # 对于大数据集,r=8;对于ETTh1,ETTh2,Exchange,ILI等小数据集,r=1;r在1,2,4,8中选取
    # 对于大数据集,num_layers=3;对于小数据集,num_layers=1,num_layers在1,2,3中选取
    # D默认等于64,D在32,64,128中选取
    def __init__(self, M, L, T, D=64, P=4, S=2, kernel_size=51, r=4, num_layers=1):
        super(ModernTCN, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([ModernTCNBlock(M, D, kernel_size, r) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)

    def forward(self, x):
        # x: [B, L, M]
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度M中取出第一个
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        # Flatten
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]

        return pred  # out: [B, T]


class ModernTCN2(nn.Module):
    # 在顶会论文中, kernel_size=51, 这里我改成了5, 原来r=8,num_layers=3
    # 对于大数据集,r=8;对于ETTh1,ETTh2,Exchange,ILI等小数据集,r=1;r在1,2,4,8中选取
    # 对于大数据集,num_layers=3;对于小数据集,num_layers=1,num_layers在1,2,3中选取
    # D默认等于64,D在32,64,128中选取
    def __init__(self, M, L, T, D=64, P=4, S=2, kernel_sizes=[5,51], r=4, num_layers=1):
        super(ModernTCN2, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        N = L // S
        self.embed_layer = Embedding(P, S, D)
        self.backbone = nn.ModuleList([ModernTCNBlock2(M, D, kernel_sizes, r) for _ in range(num_layers)])
        self.head = nn.Linear(D*N, T)
        # self.head = nn.Linear(M*D*N, T)

    def forward(self, x):
        # x: [B, L, M]
        x = x.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_emb = self.embed_layer(x)  # [B, M, L] -> [B, M, D, N]

        for i in range(self.num_layers):
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]
        
        # 从x_emb中的第2维度M中取出第一个
        x_emb = x_emb[:,0,:,:]  # [B, M, D, N] -> [B, D, N]
        # Flatten
        z = rearrange(x_emb, 'b d n -> b (d n)')  # [B, D, N] -> [B, D*N]
        # z = rearrange(x_emb, 'b m d n -> b (m d n)')  # [B, D, N] -> [B, D*N]
        pred = self.head(z)  # [B, D*N] -> [B, T]
        # pred = self.head(z)  # [B, M*D*N] -> [B, T]

        return pred  # out: [B, T]

# 输入为[B, L, M], 输出为[B, T]
# past_series = torch.rand(128, 104, 4)
# mymodel = ModernTCN2(4, 104, 52)
# pred_series = mymodel(past_series)
# print(pred_series.shape)
# # torch.Size([128, 52])
    

# 测试ConvFFN
# x = torch.rand(128, 4*64, 78)
# convffn = ConvFFN(4, 64, 2, one=True)
# out = convffn(x)
# print(out.shape)
# torch.Size([128, 256, 78])


# 测试一下Embedding的输入输出
# input = torch.rand(128, 4, 26)
# embedding = Embedding(4, 4, 64)
# output = embedding(input)
# print(output.shape)
# torch.Size([128, 4, 64, 6])