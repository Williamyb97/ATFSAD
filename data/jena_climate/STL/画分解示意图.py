from torch.utils.data import Dataset, DataLoader
import pandas as pd
import  numpy as np
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import torch

import matplotlib.pyplot as plt
import numpy as np
data_path='jena_climate_2009_2016.csv'
import math


seq_len=144*2
help_len=seq_len//2
pred_len=144





def check_data(df):
    #step1:找到00：00：00的坐标
    # --->+144*3查看是否符合常识
    # -->若符合常识则保留坐标
    df_date=df['Date Time']
    str_date = df_date.values.astype(np.str_)
    # 获得所有时分秒为000的坐标
    date=pd.to_datetime(str_date,dayfirst=True)
    dateSum=(date.hour+date.minute).values
    condition_index=np.where(dateSum==0)[0]

    where=np.where( condition_index+(seq_len+pred_len) <  len(str_date)-(seq_len+pred_len) )[0]
    condition_index=condition_index[where]
    temp_index=np.where(   date[condition_index+(seq_len+pred_len)].hour+date[condition_index+(seq_len+pred_len)].minute == 0     )
    condition_index=condition_index[temp_index]
    return condition_index



class Moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(Moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2),1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2),1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

def main():
    # 读取csv文件
    df = pd.read_csv(data_path)[:4000]
    #初始化
    traningSet=[]
    time_str=[]
    #获取合法index
    valid_index=list(check_data(df))
    column=df.columns.values
    column=np.delete(column,[list(column).index('Date Time'),list(column).index('p (mbar)'),list(column).index('wv (m/s)'),list(column).index('max. wv (m/s)'),list(column).index('wd (deg)')])
    #获取mean和std
    #获得特征样本和标签样本
    i=5
    index=valid_index[i]
    traningSet.append(df.loc[index:index+seq_len-1,column].values)
    traningSet=np.array(traningSet)

    trend= Moving_avg(kernel_size=8, stride=1)(torch.from_numpy(traningSet)).numpy()[0]
    traningSet=traningSet[0]
    seasonal=traningSet-trend


    traningSet=traningSet[:144,:]
    trend=trend[:144,:]
    seasonal=seasonal[:144,:]

    for i in range(1):
        plt.figure()
        plt.plot(traningSet[:,i],'black')
        plt.figure()
        plt.plot(trend[:,i],'black')
        plt.title(column[i])
        plt.figure()
        plt.plot(seasonal[:,i],'black')
    plt.show()
    return


if __name__ == "__main__":
    main()


