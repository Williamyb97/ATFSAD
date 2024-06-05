import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from units.untils import Series_decomp_multi
from model.attention import AttentionLayer,CrossAttentionLayer

class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.args=args
        self.dropout = nn.Dropout(args.dropout)
        self.attentionLayer = AttentionLayer(args)
        self.crossAttentionLayer = CrossAttentionLayer(args)
        self.autoSTL1 = Series_decomp_multi(args.moving_avg)
        self.autoSTL2 = Series_decomp_multi(args.moving_avg)
        self.autoSTL3 = Series_decomp_multi(args.moving_avg)
        self.activation = F.relu if args.activation == "relu" else F.gelu
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1, bias=False)
        #t1
        self.dec_Linear1_1=nn.Linear(in_features=args.d_model, out_features=args.d_ff)
        self.dec_Linear1_2 = nn.Linear(in_features=args.d_ff, out_features=1)
        #t2
        self.dec_Linear2_1=nn.Linear(in_features=args.d_model, out_features=args.d_ff)
        self.dec_Linear2_2 = nn.Linear(in_features=args.d_ff, out_features=1)
        #t3
        self.dec_Linear3_1=nn.Linear(in_features=args.d_model, out_features=args.d_ff)
        self.dec_Linear3_2 = nn.Linear(in_features=args.d_ff, out_features=1)
        self.projection = nn.Conv1d(in_channels=args.d_model, out_channels=1, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
    def forward(self,cross_info,dec_season,dec_trend):
        x  = self.attentionLayer(dec_season, dec_season, dec_season)
        x  = dec_season + self.dropout(x)
        t1,s1 = self.autoSTL1(x)

        x = s1+ self.crossAttentionLayer(s1,cross_info,cross_info)
        t2, s2 = self.autoSTL2(x)

        x = self.dropout(self.activation(self.conv1(s2.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1)) + s2
        t3, s3 = self.autoSTL3(x)
        dec_season = s3
        #cutdown
        trend1 = t1[:,-self.args.pred_len:,:]
        trend1=self.dec_Linear1_2(self.dec_Linear1_1(trend1))

        trend2 = t2[:, -self.args.pred_len:, :]
        trend2=self.dec_Linear2_2(self.dec_Linear2_1(trend2))

        trend3 = t3[:, -self.args.pred_len:, :]
        trend3=self.dec_Linear3_2(self.dec_Linear3_1(trend3))
        if self.args.weatherSTList == True:
                self.args.STList.append(trend1.clone().cpu().detach().numpy())
                self.args.STList.append(trend2.clone().cpu().detach().numpy())
                self.args.STList.append(trend3.clone().cpu().detach().numpy())
        residual_trend = trend1 + trend2 + trend3
        #dec_trend
        dec_trend=dec_trend[:,-self.args.pred_len:,:]
        dec_trend=dec_trend + residual_trend
        return dec_season,dec_trend
class DecoderStack(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.decoderList = [Decoder(args).to(args.device) for _ in range(args.decoder_layers)]
    def forward(self, cross_info,dec_season,dec_trend):
        for decoder in self.decoderList:
            dec_season,dec_trend=decoder(cross_info,dec_season,dec_trend)
        return dec_season,dec_trend
