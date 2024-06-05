import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from model.embed import BeginEmbded
from model.encoder import EncoderStack
from model.decoder import DecoderStack
from units.untils import Creat_inps

class Myformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args=args
        self.creat_inputs = Creat_inps(args)
        self.beginEmbded=BeginEmbded(args)
        self.encoderstack=EncoderStack(args)
        self.decoderstack = DecoderStack(args)
        self.Linear1=nn.Linear(in_features=args.d_model, out_features=args.d_ff)
        self.activation = F.relu if args.activation == "relu" else F.gelu
        self.Linear2 = nn.Linear(in_features=args.d_ff, out_features=1)
    def forward(self,batch_en_x,batch_den_x,batch_en_time,batch_den_time):
        # 在模型初始阶段形成: batch_en_x,batch_en_time,seasonal_init,batch_den_time,trend_init
        batch_en_x,batch_en_time,seasonal_init,batch_den_time,trend_init=self.creat_inputs(batch_en_x,batch_den_x,batch_en_time,batch_den_time)
        batch_enc_x,batch_denc_season,batch_denc_trend=self.beginEmbded(batch_en_x,batch_en_time,seasonal_init,batch_den_time,trend_init)
        enc_out = self.encoderstack(batch_enc_x)
        if self.args.useTrend_init==True:
            pass
        else:
            batch_denc_trend=torch.zeros([batch_denc_trend.shape[0], batch_denc_trend.shape[1], batch_denc_trend.shape[2]]).to(self.args.device)

        dec_season,dec_trend=self.decoderstack(enc_out,batch_denc_season,batch_denc_trend)
        x1=dec_trend
        x2= self.Linear2(self.activation(self.Linear1(dec_season)))[:,-self.args.pred_len:,:]
        if self.args.weatherSTList == True:
                self.args.STList.append(x2.clone().cpu().detach().numpy())
        predict_value=x1+x2
        if self.args.weatherSTList == True:
                self.args.STList.append(predict_value.clone().cpu().detach().numpy())
        #温度在维度0
        return predict_value