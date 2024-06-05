import argparse
import os
import sys
import torch
import random
import numpy as np
from model.exp_main import Exp_Main

def main():
    parser = argparse.ArgumentParser(description='ProbSparse_STL')
    #是否进行调试
    parser.add_argument('--debug', type=bool , default=True, help='是否进行调试')
    parser.add_argument('--att_name', type=str, default='ProbSparseAttention', help='注意力模型选择')
    parser.add_argument('--factor', type=int, default=10, help='factor')
    parser.add_argument('--moving_avg', default=[4], help='moving_avg')
    parser.add_argument('--batchsize', type=int, default=6, help='batchsize')
    # useInfo define
    parser.add_argument('--usePositionInfo', type=bool, default=True, help='使用位置信息')
    parser.add_argument('--useTimeInfo', type=bool, default=True, help='使用时间信息')
    parser.add_argument('--useTrend_init', type=bool, default=True, help='使用过去的趋势信息加入预测中')
    parser.add_argument('--useSample_k', type=bool, default=False, help='随机选取k的列')
    # model define
    parser.add_argument('--encoder_layers', type=int, default=2, help='layer of encoder ')
    parser.add_argument('--decoder_layers', type=int, default=2, help='layer of decoder ')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of hidden size')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--c_in', type=int, default=10, help='输入的特征维度')
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数')
    # forecasting task
    parser.add_argument('--target_dim', type=int, default=0,help='目标特征的列号')
    parser.add_argument('--seq_len', type=int, default=288, help='input sequence length')
    parser.add_argument('--help_len', type=int, default=288//2, help='decoder辅助信息长度')
    parser.add_argument('--pred_len', type=int, default=144, help='prediction sequence length')
    #数据地址
    parser.add_argument('--input_dataPath', type=str, default='.\\data\\jena_climate\\input_std_data.npy', help='输入数据文件地址')
    parser.add_argument('--timePath', type=str, default='.\\data\\jena_climate\\times_integer_mark.npy', help='时间文件地址')
    # optimization
    # GPU
    parser.add_argument('--device', type=str, default='cuda', help='使用训练设备')
    #均值和std
    parser.add_argument('--mean', type=float, default=9.450147354304178, help='均值')
    parser.add_argument('--std', type=float, default=8.423365210386098, help='std')
    #others
    parser.add_argument('--weatherSTList', type=bool, default=False)
    parser.add_argument('--STList', type=list, default=[])
####################################################################################################
    args = parser.parse_args()
    exp_main=Exp_Main(args)
    exp_main.start()
if __name__ == "__main__":
    main()
