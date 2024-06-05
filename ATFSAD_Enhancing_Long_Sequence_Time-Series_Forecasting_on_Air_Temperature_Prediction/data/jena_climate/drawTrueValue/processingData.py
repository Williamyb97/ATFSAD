from torch.utils.data import Dataset, DataLoader
import pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import torch


data_path='..//jena_climate_2009_2016.csv'

import matplotlib.pyplot as plt

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




def save2npy(data,str_path):
    np.save(str_path, data)
    return




def main():
    # 读取csv文件
    df = pd.read_csv(data_path)
    #初始化
    true=[]
    time_str=[]
    #获取合法index
    valid_index=list(check_data(df))
    #获取mean和std

    #获得特征样本和标签样本
    for i in valid_index:
        # true.append(df.loc[i+seq_len:i+seq_len+pred_len-1,'T (degC)'].values)
        # time_str.append(df.loc[i+seq_len:i+seq_len+pred_len-1,'Date Time'].values[0])
        plt.plot(df.loc[i+seq_len:i+seq_len+pred_len-1,'T (degC)'].values, label=df.loc[i+seq_len:i+seq_len+pred_len-1,'Date Time'].values[0])
        plt.legend()

        plt.show()


    true=np.array(true)

    return


if __name__ == "__main__":
    main()


