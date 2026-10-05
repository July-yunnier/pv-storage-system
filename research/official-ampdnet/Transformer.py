import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np
import argparse



# 相关参数解释
# output_attention:  # whether to output attention in encoder, 默认为False
# channel_independence: 是否通道独立; 
# pred_len: 预测序列的长度;
# enc_in: encoder的输入维度; dec_in: decoder的输入维度;
# c_out: 输出序列的维度; d_model: 一般为512,是嵌入的隐空间维度;
# embed: time features encoding, options:[timeF, fixed, learned]; 
# freq: # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h;
# dropout:;
# factor:;
# n_heads: 多头注意力的头数;
# e_layers: encoder的层数; d_layers: decoder的层数;
# d_ff: FFN中的维度;
# features: forecasting task, options:[M, S, MS], 分别为多对多,单对单,多对单;


class Transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    # def __init__(self, output_size, output_attention, channel_independence, enc_in, dec_in, c_out, d_model, embed, freq, dropout, n_heads, d_ff, activation, e_layers, factor, d_layers):
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # Embedding
        # enc_in,d_model等是用来初始化这个类的,用来创建一个对象
        self.enc_embedding = DataEmbedding(self.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.dec_embedding = DataEmbedding(self.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec): # [B, L, N1] -> [B, H, c_out]
        # Embedding
        # x_enc, x_mark_enc参数传进了DataEmbedding的forward函数
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [B, L, N1] -> [B, L, d]
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # [B, L, d] -> [B, L, d] #自注意力机制对L维度进行操作

        dec_out = self.dec_embedding(x_dec, x_mark_dec) # [B, L2, N1] -> [B, L2, d]
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None) #[B, L2, d] -> [B, L2, d] #交叉注意力对L2维度进行操作
        return dec_out
    
    # x_mark_enc,x_mark_dec分别是用来表明encoder,decoder之前的输入要不要加入时序嵌入
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None): # [B, L, N1] -> [B, H, c_out]
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        # dec_out = torch.squeeze(dec_out, dim=-1)
        return dec_out[:, -self.pred_len:, :].squeeze(-1)  # [B, H, c_out]



# 源代码的Transformer的输入是(B, L, D)
# Transformer里面没有factor这个参数,只有Informer里面有这个参数
parser = argparse.ArgumentParser(description='Transformer')
args = parser.parse_args()
# 模型的配置config
parser.add_argument('--pred_len', type=int, default=1)
parser.add_argument('--output_attention', type=bool, default=False)
parser.add_argument('--channel_independence', type=bool, default=False)
parser.add_argument('--enc_in', type=int, default=13)
parser.add_argument('--dec_in', type=int, default=13)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--d_model', type=int, default=128)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--dropout', type=int, default=0.01)
parser.add_argument('--factor', type=int, default=5)
parser.add_argument('--n_heads', type=int, default=4)
parser.add_argument('--d_ff', type=int, default=256)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_layers', type=int, default=3)


# 模型的具体参数
args.pred_len = 1
args.output_attention = False
args.channel_independence = False
# 编码器的输入变量数
args.enc_in = 5
# 解码器的输入变量数
args.dec_in = 5
# 输出变量数
args.c_out = 1
args.d_model = 128
args.embed = 'timeF'
args.freq = 'h'
args.dropout = 0.01
args.factor = 5
args.n_heads = 4
args.d_ff = 256
args.activation = F.gelu
args.e_layers = 2
args.d_layers = 2



# 源代码的Transformer的输入是(B, L, N1), N1是变量数
# Transformer里面没有factor这个参数,只有Informer里面有这个参数
# mymodel = Transformer(13,1,128,3,5,0.01,False,4,256,'relu',3)
myTrans = Transformer(args)
# enc_input = torch.rand((64, 13, 3))
# dec_input = torch.rand((64, 13, 3))
# 这里x_mark_enc=None和x_mark_dec=None,表明的encoder和decoder之前的输入都不用时序嵌入
# out = mymodel(x_enc=enc_input, x_mark_enc=None, x_dec=dec_input, x_mark_dec=None)
# print(out.shape)
# torch.Size([64, 1])