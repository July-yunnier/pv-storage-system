import argparse
import datetime
import os

# import lightning as L
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from einops import rearrange
# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_
# from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

# from data_factory import data_provider
# from utils import save_copy_of_files, random_masking_3D, str2bool


# class ICB(L.LightningModule):
# 由代码可知,ICB实现的是patch_num维度的交互
class ICB(nn.Module): # [B*M, N, E] -> [B*M, N, E]
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x): # [B*M, N, E] -> [B*M, N, E]
        x = x.transpose(1, 2) # [B*M, N, E] -> [B*M, E, N]
        x1 = self.conv1(x) # [B*M, E, N] -> [B*M, 3*E, N]
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x) # [B*M, E, N] -> [B*M, 3*E, N]
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2) # [B*M, 3*E, N] -> [B*M, E, N]
        x = x.transpose(1, 2) # [B*M, E, N] -> [B*M, N, E]
        return x



# torch.randn(dim, 2, dtype=torch.float32) * 0.02这部分代码创建了一个形状为(dim, 2)的张量,
# 其中的元素是从均值为0, 标准差为1的正态分布中随机抽取的, 然后乘以0.02, 这是一种常见的权重初始化方法
# 最后一个维度是2表示实部和虚部,即此时complex_weight_high张量的元素全部是实数
# trunc_normal_函数的作用是将输入张量中的元素截断到指定的标准差范围内
class Adaptive_Spectral_Block(nn.Module): # [B*M, N, E] -> [B*M, N, E]
    def __init__(self, dim):
        super().__init__()
        # 此为局部权重
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        # 此为全局权重
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        
        # 这段代码的作用是使用标准差为0.02的截断正态分布来初始化self.complex_weight_high这个参数
        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        # 自适应阈值
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
    
    # 该函数的作用是创建一个自适应的高频掩码,用于在频域中筛选出主要的频率成分
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1) # [B, N//2+1, E] -> [B, N//2+1, E] -> [B, N//2+1]

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension, # [B, N//2+1] -> [B, N//2+1]
        # median函数返回一个元组，其中第一个元素是中位数，第二个元素是中位数在原张量中的索引。在这个例子中，我们只关心中位数，所以使用[0]来获取元组的第一个元素
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median, # [B, N//2+1] -> [B, 1]
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions, # [B, 1] -> [B, 1]

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)
        
        # 计算normalized_energy张量的self.threshold_param分位数,并将结果存储在threshold中
        threshold = torch.quantile(normalized_energy, self.threshold_param) # [B, N//2+1, E] -> [B, 1],即每个样本的阈值,是一个标量
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device) # [B, N//2+1, E]
        # 将dominant_frequencies张量中的值为True,即大于阈值的归一化能量处的位置设为1,其余位置保持为0
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask # [B, N//2+1, E]

    def forward(self, x_in, adaptive_filter=True): # [B*M, N, E] -> [B*M, N, E]
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension,即在N这个维度上使用FFT
        # 如果x的形状是[BM, N, E],那么x_fft的形状将是[BM, N//2+1, E],
        # 这是因为对于实数输入,傅立叶变换的结果是对称的,所以只需要保留一半的结果就足够了
        # N//2+1是因为还包括了0频率(直流分量)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho') # [B*M, N, E] -> [B*M, N//2+1, E]
        # 这段代码的作用是将self.complex_weight转换为复数张量,并将结果存储在weight中
        weight = torch.view_as_complex(self.complex_weight) # [dim, 2] -> [dim], 这里的dim即为E
        x_weighted = x_fft * weight # 全局权重

        if adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft) # [B, N//2+1, E]
            x_masked = x_fft * freq_mask.to(x.device) # [B*M, N//2+1, E]

            weight_high = torch.view_as_complex(self.complex_weight_high) # [dim, 2] -> [dim], 这里的dim即为E
            x_weighted2 = x_masked * weight_high # 局部权重, [B*M, N//2+1, E]

            x_weighted += x_weighted2 # 全局权重和局部权重之和, [B*M, N//2+1, E]

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho') # [B*M, N//2+1, E] -> [B*M, N, E]

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


# class TSLANet_layer(L.LightningModule):
class TSLANet_layer(nn.Module): # [B*M, N, E] -> [B*M, N, E]
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # Check if both ASB and ICB are true
        # if args.ICB and args.ASB:
        #     x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # # If only ICB is true
        # elif args.ICB:
        #     x = x + self.drop_path(self.icb(self.norm2(x)))
        # # If only ASB is true
        # elif args.ASB:
        #     x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class TSLANet(nn.Module): # [B, L, M] -> [B, H]

    def __init__(self, input_size, patch_size, emb_dim, num_layers, output_size, dropout=0.):   
        super(TSLANet, self).__init__()

        self.patch_size = patch_size
        self.stride = self.patch_size // 2
        num_patches = int((input_size - self.patch_size) / self.stride + 1)

        # 嵌入层,就一个单纯的线性层,都没有位置嵌入,感觉就是论文说一套,代码又是另一套
        self.input_layer = nn.Linear(self.patch_size, emb_dim)
        
        # 生成一个从0到dropout的等差数列, 这个数列包含num_layers哥元素, 并将这个数列的每个元素转换为Python的标量值, 然后将这些标量值存储在dpr列表中
        dpr = [x.item() for x in torch.linspace(0, dropout, num_layers)]  # stochastic depth decay rule

        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=emb_dim, drop=dropout, drop_path=dpr[i])
            for i in range(num_layers)]
        )

        # 输出层,此为单层线性层
        self.out_layer = nn.Linear(emb_dim * num_patches, output_size)

    # def pretrain(self, x_in):
    #     x = rearrange(x_in, 'b l m -> b m l')
    #     x_patched = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    #     x_patched = rearrange(x_patched, 'b m n p -> (b m) n p')

    #     xb_mask, _, self.mask, _ = random_masking_3D(x_patched, mask_ratio=args.mask_ratio)
    #     self.mask = self.mask.bool()  # mask: [bs x num_patch]
    #     xb_mask = self.input_layer(xb_mask)

    #     for tsla_blk in self.tsla_blocks:
    #         xb_mask = tsla_blk(xb_mask)

    #     return xb_mask, self.input_layer(x_patched)


    def forward(self, x): # [B, L, M] -> [B, H, M] 或 [B, H]
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride) # [B, M, L] -> [B, M, N, P]
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x) # [B*M, N, P] -> [B*M, N, E]

        for tsla_blk in self.tsla_blocks: # [B*M, N, E] -> [B*M, N, E]
            x = tsla_blk(x)

        outputs = self.out_layer(x.reshape(B * M, -1)) # [B*M, N, E] -> [B*M, N*E] -> [B*M, H]
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B) # [B*M, H] -> [B, M, H]

        outputs = outputs * stdev
        outputs = outputs + means

        # return outputs # [B, H, M]
        return outputs[:, :, 0] # [B, H, M] -> [B, H]
    



# 测试Adaptive_Spectral_Block的输入输出
# input = torch.randn(128*4, 52, 64)
# ASB = Adaptive_Spectral_Block(64)
# output = ASB(input)
# print(output.shape)
# torch.Size([512, 52, 64])


# 测试ICB的输入输出
# input = torch.randn(128*4, 52, 64)
# ICB = ICB(64, 3*64)
# output = ICB(input)
# print(output.shape)
# # torch.Size([512, 52, 64])
    

# 测试TSLANet_layer的输入输出
# input = torch.randn(128*4, 52, 64)
# TSLANet_layer = TSLANet_layer(64)
# output = TSLANet_layer(input)
# print(output.shape)
# # torch.Size([512, 52, 64])
    

# 测试TSLANet的输入输出
# input = torch.randn(128, 104, 4)
# TSLANet = TSLANet(104, 4, 64, 1, 16, 0.02)
# output = TSLANet(input)
# print(output.shape)
# # torch.Size([128, 16])

