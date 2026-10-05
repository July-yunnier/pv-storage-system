import torch
import torch.nn as nn
# from layers.RevIN import RevIN
from layers.Leddam import Leddam



# pe_type为位置嵌入的类型
class Leddam2024(nn.Module):
    def __init__(self, in_channels, input_size, output_size, d_model, dropout, pe_type, num_layers):
        super(Leddam2024, self).__init__()
        # self.revin=configs.revin
        # self.revin_layer=RevIN(channel=configs.enc_in,output_dim=configs.pred_len)
        self.leddam = Leddam(in_channels, input_size, d_model, 
                       dropout, pe_type, kernel_size=5, n_layers=num_layers)
        
        self.Linear_main = nn.Linear(d_model, output_size) 
        self.Linear_res = nn.Linear(d_model, output_size)
        self.Linear_main.weight = nn.Parameter(
                (1 / d_model) * torch.ones([output_size, d_model])) 
        self.Linear_res.weight = nn.Parameter(
                (1 / d_model) * torch.ones([output_size, d_model])) 
    def forward(self, inp):

        # if self.revin:
        #     inp = self.revin_layer(inp)
        res, main = self.leddam(inp)
        main_out = self.Linear_main(main.permute(0,2,1)).permute(0,2,1)
        res_out = self.Linear_res(res.permute(0,2,1)).permute(0,2,1)
        pred = main_out + res_out
        # if self.revin:
        #     pred = self.revin_layer.inverse_normalize(pred)
        # return pred
        return pred[:,:,0]
    

# # 写一个Config类,用于保存配置信息
# class Config:
#     def __init__(self):
#         # self.revin = False
#         self.enc_in = 4
#         self.pred_len = 16
#         self.seq_len = 104
#         self.d_model = 64
#         self.dropout = 0.02
#         self.pe_type = 'gauss'
#         self.n_layers = 2


# # 实例化一个Config类
# configs = Config()
# # 对Leddam模型进行输入输出测试
# model = Leddam2024(configs)
# input = torch.randn(128, 104, 4)
# output = model(input)
# print(output.shape)
# # torch.Size([128, 16])
    

# 测试Leddam模型的输入输出
# input = torch.randn(128, 104, 4)
# model = Leddam2024(4, 104, 16, 64, 0.02, 'gauss', 2)
# output = model(input)
# print(output.shape)
# torch.Size([128, 16])