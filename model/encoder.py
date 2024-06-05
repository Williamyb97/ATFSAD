import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from units.untils import Series_decomp_multi
from model.attention import AttentionLayer

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.attentionLayer = AttentionLayer(args)
        self.autoSTL1 = Series_decomp_multi(args.moving_avg)
        self.autoSTL2 = Series_decomp_multi(args.moving_avg)
        self.activation = F.relu if args.activation == "relu" else F.gelu
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=args.d_ff, out_channels=args.d_model, kernel_size=1, bias=False)
    def forward(self, x):
        x  = self.attentionLayer(x, x, x)
        x  = x + self.dropout(x)
        x,_= self.autoSTL1(x)
        x2 = x
        x2 = self.dropout(self.activation(self.conv1(x2.transpose(-1, 1))))
        x2 = self.dropout(self.conv2(x2).transpose(-1, 1))
        x,_= self.autoSTL2(x + x2)
        return x


class EncoderStack(nn.Module):
    def __init__(self,args):
        super(EncoderStack, self).__init__()
        self.args=args
        self.encoderList = [Encoder(args).to(args.device) for _ in range(args.encoder_layers)]

    def forward(self, x):
        for encoder in self.encoderList:
            x=encoder(x)
        return x
