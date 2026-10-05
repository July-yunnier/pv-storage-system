import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math



# 固定位置嵌入
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class LD(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True) 
        # Define the parameters for Gaussian initialization
        kernel_size_half = kernel_size // 2
        sigma = 1.0  # 1 for variance
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = math.exp(-((i - kernel_size_half) / (2 * sigma)) ** 2)

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights,dim=-1)
        self.conv.bias.data.fill_(0.0)
        
    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)
        
        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out
    

# 模仿上面的LD,但是卷积核的初始化为均匀分布(相等分布)的权重
class LD_uni(nn.Module):
    def __init__(self, kernel_size=25):
        super(LD_uni, self).__init__()
        # Define a shared convolution layers for all channels
        self.conv=nn.Conv1d(1, 1, kernel_size=kernel_size, stride=1, padding=int(kernel_size//2), padding_mode='replicate', bias=True) 
        weights = torch.zeros(1, 1, kernel_size)
        for i in range(kernel_size):
            weights[0, 0, i] = 1.0

        # Set the weights of the convolution layer
        self.conv.weight.data = F.softmax(weights,dim=-1)
        self.conv.bias.data.fill_(0.0)
    
    def forward(self, inp):
        # Permute the input tensor to match the expected shape for 1D convolution (B, N, T)
        inp = inp.permute(0, 2, 1)
        # Split the input tensor into separate channels
        input_channels = torch.split(inp, 1, dim=1)
        
        # Apply convolution to each channel
        conv_outputs = [self.conv(input_channel) for input_channel in input_channels]
        
        # Concatenate the channel outputs
        out = torch.cat(conv_outputs, dim=1)
        out = out.permute(0, 2, 1)
        return out


class FlattenHead(nn.Module):
    def __init__(self, D, num_patch, output_size, dp_head=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(D*num_patch, output_size)
        self.dropout = nn.Dropout(dp_head)

    def forward(self, x):  # x: [B, D, N2]
        x = self.flatten(x) # [B, D, N2] -> [B, D*N2]
        x = self.linear(x) # [B, D*N2] -> [B, H]
        x = self.dropout(x)
        return x



# 前馈神经网络
class FFN(nn.Module): # [B, D, N] -> [B, r*D, N] -> [B, D, N]
    def __init__(self, D, r, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=D, out_channels=r*D, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=r*D, out_channels=D, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        return x
    

# 非组卷积的ConvGLU, 即DAMSNet中的Undecoupled ConvGLU
class ConvGLU(nn.Module): # [B*M, D, N] -> [B*M, r*D, N] -> [B*M, D, N]
    def __init__(self, M, D, r=2, dropout=0.02):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.pw_con1 = nn.Conv1d(
            in_channels=D, 
            out_channels=r*D, 
            kernel_size=1,
            )
        self.pw_con2 = nn.Conv1d(
            in_channels=D, 
            out_channels=r*D, 
            kernel_size=1,
            )
        self.pw_con3 = nn.Conv1d(
            in_channels=r*D, 
            out_channels=D, 
            kernel_size=1,
            )

    def forward(self, x):
        # x: [B, M*D, N]
        # 上面的代码即为以下3行
        x1 = self.pw_con1(x) # [B*M, D, N] -> [B*M, r*D, N]
        x2 = self.pw_con2(x) # [B*M, D, N] -> [B*M, r*D, N]
        # x2经过sigmoid激活函数,然后与x1相乘
        x3 = x1 * torch.sigmoid(x2) # [B*M, r*D, N] -> [B*M, r*D, N]
        x3 = self.dropout1(x3) # [B*M, r*D, N] -> [B*M, r*D, N]
        x4 = self.pw_con3(x3) # [B*M, r*D, N] -> [B*M, D, N]
        x4 = self.dropout2(x4) # [B*M, D, N] -> [B*M, D, N]
        return x4


# 基于FFN的Transformer编码器
# [B*N1, N2, D] -> [B*N1, N2, D], Encoder在N2这个维度上进行自注意力计算
# FFN在D这个维度上进行信息交互
class Encoder(nn.Module): 
    def __init__(self, in_channels, input_size, output_size, n_head, D, P, S, r, dropout):
        super().__init__()
        self.att = nn.MultiheadAttention(embed_dim=D, num_heads=n_head, dropout=dropout, batch_first=True)
        self.attn_weights = None
        self.norm1 = nn.LayerNorm(D)
        self.ffn1 = FFN(D, r, dropout)
        # self.glu = ConvGLU(in_channels, D, r, dropout)
        self.norm2 = nn.LayerNorm(D)
    
    def forward(self, x_emb):
        new_x, self.attn_weights = self.att(x_emb, x_emb, x_emb)
        # self.attn_weights = attn_output_weights
        x_emb = x_emb + new_x
        x_emb = self.norm1(x_emb)
        new_x = self.ffn1(x_emb.transpose(1, 2)).transpose(1, 2)
        # new_x = self.glu(x_emb.transpose(1, 2)).transpose(1, 2)
        x_emb = x_emb + new_x
        x_emb = self.norm2(x_emb)
        return x_emb



# 2024.12.13
# 具体来说, PVformer4考虑了未来气象变量信息, 这是我的第2个研究点的模型
# PVformer4模型即为我的第2个研究点的模型
# PVformer4模型用run_PV9.py文件来跑
class PVformer4(nn.Module):
    def __init__(self, in_channels, input_size, output_size, n_head, D1, D2, P, S, r, dp1, dp2, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # self.ld = LD(kernel_size=9)
        # self.amsp = AMSP(kernel_sizes=[3, 5, 7, 9])
        self.patch_embedding = PatchEmbedding(D1, P, S, padding=S, dropout=0)
        self.variate_embedding = nn.Linear(input_size+output_size, D2)
        self.patch_encoder = nn.ModuleList([Encoder(in_channels, input_size, output_size, n_head, D1, P, S, r, dp1) for _ in range(num_layers)])
        self.variate_encoder = nn.ModuleList([Encoder(in_channels, input_size, output_size, n_head, D2, P, S, r, dp2) for _ in range(num_layers)])
        self.num_patch = int((input_size - P) / S) + 2
        self.head1 = nn.Linear(D1*self.num_patch, output_size)
        self.head2 = nn.Linear(D2, output_size)
        # self.projection = nn.Linear(input_size, output_size)
    
    # 获取最后一层的注意力权重, 即判断哪个变量对预测结果的影响最大
    def get_last_attn_weights(self):
        # return self.variate_encoder[-1].attn_weights
        # 返回的是最后一个Encoder的attn_weights,并将其类型转换为Tensor
        return self.variate_encoder[-1].attn_weights
    
    
    # 注: 这里x1是过去信息,具体包括历史光伏功率和历史气象变量信息; 
    # x2是未来气象变量信息,仅包括未来气象变量信息
    # x1的shape是[B, L, M], x2的shape是[B, H, M-1]; N1是变量个数, M也是变量个数, N1和M是一个意思, N2是patch块的个数
    def forward(self, x1, x2, signal=1): # [B, L, M] 和 [B, H, M-1] -> [B, H]
        x_patch = x1
        x_patch = x_patch.permute(0,2,1) # [B, L, M] -> [B, M, L]
        x_patch, N1 = self.patch_embedding(x_patch) # [B, M, L] -> [B*N1, N2, D]

        for i in range(self.num_layers):
            x_patch = self.patch_encoder[i](x_patch)  # [B*N1, N2, D] -> [B*N1, N2, D]
        
        x_patch = rearrange(x_patch, '(b n1) n2 d -> b n1 n2 d', n1=N1)  # [B*N1, N2, D] -> [B, N1, N2, D]
        x_patch = x_patch[:, 0, :, :]  # [B, N1, N2, D] -> [B, N2, D]
        x_patch = rearrange(x_patch, 'b n2 d -> b (d n2)', n2=self.num_patch)  # [B, N2, D] -> [B, D*N2]
        x_patch = self.head1(x_patch) # [B, N2*D] -> [B, H]

        # 将x_patch和x2进行拼接
        x_patch_2 = x_patch.unsqueeze(2)  # 将x_patch的形状从[B, H]改为[B, H, 1]
        # x_variate = torch.cat((x2, x_patch_2), dim=2) # 这行是原先的错误代码,已经修改
        x_variate = torch.cat((x_patch_2, x2), dim=2)  # 在最后一个维度(即变量维度)上拼接x2和x_patch_2, 拼接后的x_variate的shape是[B, H, M], 且功率变量在第一个位置
        # 将x1和x_variate进行拼接
        x_variate = torch.cat((x1, x_variate), dim=1) # 在第二个维度(即时间维度)上拼接x1和x_variate, 拼接后的x_variate的shape是[B, L+H, M], 且L在前, H在后

        x_variate = x_variate.permute(0,2,1) # [B, L+H, M] -> [B, M, L+H]
        x_variate = self.variate_embedding(x_variate)  # [B, N1, L+H] -> [B, N1, D]

        for i in range(self.num_layers):
            x_variate = self.variate_encoder[i](x_variate) # [B, N1, D] -> [B, N1, D]
        
        x_variate = x_variate[:, 0, :] # [B, N1, D] -> [B, D]
        x_variate = self.head2(x_variate) # [B, D] -> [B, H]


        # 子组件的消融实验
        if signal == 1:
            x_out = x_patch + x_variate # [B, H] + [B, H] -> [B, H]
        # 序列内子模型, 仅使用了历史光伏功率进行预测, 即History-only
        elif signal == 2:
            x_out = x_patch # [B, H]
        # 序列间子模型
        elif signal == 3:
            x_out = x_variate # [B, H]
        
        return x_out # [B, H]
  
    

# PVformer4的消融实验 仅使用气象变量未来信息进行预测, 即Future-only
class PVformer4_Future(nn.Module):
    def __init__(self, in_channels, input_size, output_size, n_head, D1, D2, P, S, r, dp1, dp2, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # self.ld = LD(kernel_size=9)
        # self.amsp = AMSP(kernel_sizes=[3, 5, 7, 9])
        # self.patch_embedding = PatchEmbedding(D1, P, S, padding=S, dropout=0)
        self.variate_embedding = nn.Linear(output_size, D2)
        # self.patch_encoder = nn.ModuleList([Encoder(in_channels, input_size, output_size, n_head, D1, P, S, r, dp1) for _ in range(num_layers)])
        self.variate_encoder = nn.ModuleList([Encoder(in_channels-1, input_size, output_size, n_head, D2, P, S, r, dp2) for _ in range(num_layers)])
        # self.num_patch = int((input_size - P) / S) + 2
        # self.head1 = nn.Linear(D1*self.num_patch, output_size)
        self.head2 = nn.Linear(D2, output_size)
        # self.projection = nn.Linear(input_size, output_size)
    
    
    # 注: 这里x1是过去信息,具体包括历史光伏功率和历史气象变量信息; 
    # x2是未来气象变量信息,仅包括未来气象变量信息
    # x1的shape是[B, L, M], x2的shape是[B, H, M-1]; N1是变量个数, M也是变量个数, N1和M是一个意思, N2是patch块的个数
    def forward(self, x1, x2): # [B, L, M] 和 [B, H, M-1] -> [B, H]
        # x_patch = x1
        # x_patch = x_patch.permute(0,2,1) # [B, L, M] -> [B, M, L]
        # x_patch, N1 = self.patch_embedding(x_patch) # [B, M, L] -> [B*N1, N2, D]

        # for i in range(self.num_layers):
        #     x_patch = self.patch_encoder[i](x_patch)  # [B*N1, N2, D] -> [B*N1, N2, D]
        
        # x_patch = rearrange(x_patch, '(b n1) n2 d -> b n1 n2 d', n1=N1)  # [B*N1, N2, D] -> [B, N1, N2, D]
        # x_patch = x_patch[:, 0, :, :]  # [B, N1, N2, D] -> [B, N2, D]
        # x_patch = rearrange(x_patch, 'b n2 d -> b (d n2)', n2=self.num_patch)  # [B, N2, D] -> [B, D*N2]
        # x_patch = self.head1(x_patch) # [B, N2*D] -> [B, H]

        # 将x_patch和x2进行拼接
        # x_patch_2 = x_patch.unsqueeze(2)  # 将x_patch的形状从[B, H]改为[B, H, 1]
        # x_variate = torch.cat((x2, x_patch_2), dim=2)  # 在最后一个维度上拼接x2和x_patch_2, x_variate的shape是[B, H, M]
        # 将x1和x_variate进行拼接
        # x_variate = torch.cat((x1, x_variate), dim=1) # 在第二个维度上拼接x1和x_variate, x_variate的shape是[B, L+H, M]

        x_variate = x2
        x_variate = x_variate.permute(0,2,1) # [B, H, M-1] -> [B, M-1, H]
        x_variate = self.variate_embedding(x_variate)  # [B, M-1, H] -> [B, M-1, D2]

        for i in range(self.num_layers):
            x_variate = self.variate_encoder[i](x_variate) # [B, M-1, D2] -> [B, M-1, D2]
        
        x_variate = x_variate[:, 0, :] # [B, M-1, D2] -> [B, D2]
        x_variate = self.head2(x_variate) # [B, D2] -> [B, H]


        x_out = x_variate # [B, H]
        
        return x_out # [B, H]




# 测试PVformer4的输入输出
# 包括power的话总共4个变量, 输入长度为104, 输出长度为16
# 历史信息input1, 未来信息input2, input2为不包括power的未来信息, 所以变量数为3
# input1 = torch.randn(128, 104, 4)
# input2 = torch.randn(128, 16, 3)
# mymodel = PVformer4(4, 104, 16, 4, 64, 128, 4, 2, 2, 0.04, 0.04, 1)
# output = mymodel(input1, input2)
# print(output.shape)
# torch.Size([128, 16])
    
# 测试PVformer4_Future的输入输出
# input1 = torch.randn(128, 104, 4)
# input2 = torch.randn(128, 16, 3)
# mymodel = PVformer4_Future(4, 104, 16, 4, 64, 128, 4, 2, 2, 0.04, 0.04, 1)
# output = mymodel(input1, input2)
# print(output.shape)
# torch.Size([128, 16])