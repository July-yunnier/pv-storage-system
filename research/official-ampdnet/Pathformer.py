import math
import torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul


class Pathformer(nn.Module):
    def __init__(self, configs):
        super(Pathformer, self).__init__()
        self.layer_nums = configs.layer_nums  # 设置pathway的层数
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        # x = x.permute(0, 2, 1)

        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))


        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        # return out, balance_loss
        return out[:,:,0]
    


# 写一个config类,用于存储模型的参数
class Config:
    def __init__(self):
        self.layer_nums = 2
        self.num_nodes = 4 # 变量个数
        self.pred_len = 16
        self.seq_len = 104
        self.k = 2
        self.num_experts_list = [2, 2, 2]
        # self.patch_size_list = [8, 6, 4, 2]
        self.patch_size_list = [[8, 4, 2], [8, 4, 2], [8, 4, 2], [8, 4, 2]]
        self.d_model = 4
        self.d_ff = 64
        self.residual_connection = 1
        self.revin = True
        self.gpu = 3


# 测试Pathformer的输入输出
# 这个模型代码挺傻逼的,输入长度取104的时候就会报错,分patch的时候分不尽
config = Config()
# input = torch.randn(128, 104, 4)
mymodel = Pathformer(config)
# output = mymodel(input)
# print(output.shape)
# torch.Size([128, 16, 4])
