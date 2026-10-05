# 此为从清华大学时间序列预测库Time-Series-Library中下载的PatchTST代码


import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


# nf即为head_nf,即为d_model * (int((input_size - patch_len) / stride) + 2),即为d*N2
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, target_window)
        # self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(nf*n_vars, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num],即[B, N1, d, N2]
        x = self.flatten(x) # [B, N1, d, N2] -> [B, N1*d*N2]
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FlattenHead2(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.linear = nn.Linear(nf, target_window)
        # self.dropout = nn.Dropout(head_dropout)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [B, d, N2]
        x = self.flatten(x) # [B, d, N2] -> [B, d*N2]
        x = self.linear(x) # [B, d*N2] -> [B, H]
        x = self.dropout(x)
        return x


# N1是变量数, N2是patch的个数
class PatchTST(nn.Module):
    def __init__(self, input_size, enc_in, output_size, d_model, dropout, factor, output_attention, n_heads, d_ff, activation, e_layers, patch_len=4, stride=2):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super(PatchTST, self).__init__()
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.head_nf = d_model * \
                       (int((input_size - patch_len) / stride) + 2)
        # self.head = FlattenHead(enc_in, self.head_nf, output_size,
        #                             head_dropout=dropout)
        self.head = FlattenHead2(self.head_nf, output_size,
                                    head_dropout=dropout)

    def forward(self, x_enc): # [B, L, N1] -> [B, H]

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1) # [B, L, N1] -> [B, N1, L]
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc) # [B, N1, L] -> [B*N1, N2, d]

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out) # [B*N1, N2, d] -> [B*N1, N2, d]
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [B*N1, N2, d] -> [B, N1, N2, d]
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2) # [B, N1, N2, d] -> [B, N1, d, N2]
        enc_out = enc_out[:, 0, :, :] # [B, N1, d, N2] -> [B, d, N2]

        # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = self.head(enc_out)  # [B, N1, d, N2] -> [B, N1*d*N2] -> [B, H]
        dec_out = self.head(enc_out)  # [B, d, N2] -> [B, d*N2] -> [B, H]
        # dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    # def forward(self, x_enc):
    #     dec_out = self.forecast(x_enc)
    #     return dec_out[:, -self.pred_len:, :]  # [B, H, N1]




# 源代码的PatchTST的输入是(B, L, N1), N1是变量数
# PatchTST里面没有factor这个参数,只有Informer里面有这个参数
# mymodel = PatchTST(104,4,16,128,0.01,5,False,4,256,'relu',1,4,2)
# src = torch.rand(128, 104, 4)
# out = mymodel(src)
# print(out.shape)
# torch.Size([128, 16])