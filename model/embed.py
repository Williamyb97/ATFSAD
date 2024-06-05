import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np



class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,kernel_size=3, padding=padding, padding_mode='circular', bias=False)
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(100.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=1000):
#         super(PositionalEmbedding, self).__init__()
#         # pe = torch.zeros(max_len, d_model).float()
#         # pe.require_grad = False
#         # position = torch.arange(0, max_len).float().unsqueeze(1)
#         # div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10.0) / d_model)).exp()
#         # pe[:, 0::2] = torch.sin(position * div_term)
#         # pe[:, 1::2] = torch.cos(position * div_term)
#         # pe = pe.unsqueeze(0)
#         # self.register_buffer('pe', pe)
#     def forward(self, x):
#         # out= self.pe[:, :x.size(1)]
#
#         d_model=512
#         seq_len=288
#         pos_table = np.array([
#             [pos / np.power(10, 2 * i / d_model) for i in range(d_model)]
#             for pos in range(seq_len)])
#         pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
#         pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
#         return torch.FloatTensor(pos_table)
#
#         # return out


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TemporalEmbedding, self).__init__()
        dayOfYear_size =  365+2
        dayOfMonth_size = 31+2
        dayOfWeek_size =  7+2
        hourOfDay_size = 24+2
        minuteOfHour_size = 60+2
        self.dayOfYear_embed = nn.Embedding(dayOfYear_size, d_model)
        self.dayOfMonth_embed = nn.Embedding(dayOfMonth_size, d_model)
        self.dayOfWeek_embed = nn.Embedding(dayOfWeek_size, d_model)
        self.hourOfDay_embed = nn.Embedding(hourOfDay_size, d_model)
        self.minuteOfHour_embed = nn.Embedding(minuteOfHour_size, d_model)
    def forward(self, x):
        x=x.int()
        dayOfYear=self.dayOfYear_embed(x[:, :, 0])
        dayOfMonth=self.dayOfMonth_embed(x[:, :, 1])
        dayOfWeek=self.dayOfWeek_embed(x[:, :, 2])
        hourOfDay=self.hourOfDay_embed(x[:, :, 3])
        minuteOfHour=self.minuteOfHour_embed(x[:, :, 4])
        return dayOfYear + dayOfMonth + dayOfWeek + hourOfDay + minuteOfHour


class BeginEmbded(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args=args
        #enc
        self.enc_value_embedding = TokenEmbedding(c_in=args.c_in, d_model=args.d_model)
        self.enc_position_embedding = PositionalEmbedding(d_model=args.d_model)
        self.enc_time_embedding = TemporalEmbedding(d_model=args.d_model)
        #denc_season
        self.denc_value_embedding = TokenEmbedding(c_in=args.c_in, d_model=args.d_model)
        self.denc_position_embedding = PositionalEmbedding(d_model=args.d_model)
        self.denc_time_embedding = TemporalEmbedding(d_model=args.d_model)
        #dropout
        self.dropout = nn.Dropout(p=args.dropout)
    def forward(self, batch_en_x,batch_en_time,seasonal_init,batch_den_time,trend_init):
        #数值编码
        enc_value_embed=self.enc_value_embedding(batch_en_x)
        denc_value_embed=self.denc_value_embedding(seasonal_init)
        #位置信息编码
        if self.args.usePositionInfo==True:
            enc_position_embed=self.enc_position_embedding(batch_en_x).to(self.args.device)
            denc_position_embed=self.denc_position_embedding(seasonal_init).to(self.args.device)
        else:
            enc_position_embed=0
            denc_position_embed=0
        #时间信息编码
        if self.args.useTimeInfo==True:
            enc_time_embed=self.enc_time_embedding(batch_en_time)
            denc_time_embed=self.denc_time_embedding(batch_den_time)
        else:
            enc_time_embed=0
            denc_time_embed=0
        #结合
        # batch_enc_x=self.dropout(enc_value_embed+enc_position_embed+enc_time_embed)
        # batch_denc_season=self.dropout(denc_value_embed+denc_position_embed+denc_time_embed)
        batch_enc_x=enc_value_embed+enc_position_embed+enc_time_embed
        batch_denc_season=denc_value_embed+denc_position_embed+denc_time_embed
        batch_denc_trend=trend_init[:,:,self.args.target_dim:self.args.target_dim+1]
        if self.args.weatherSTList == True:
                self.args.STList.append(batch_denc_trend.clone().cpu().detach().numpy()[:,-144:,:])
        return batch_enc_x,batch_denc_season,batch_denc_trend


