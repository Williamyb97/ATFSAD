from torch.utils.data import Dataset, DataLoader
import pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import torch


data_path='jena_climate_2009_2016.csv'


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



def standardization(data,mean,sigma):
    std_data=(data-mean)/sigma
    return std_data


def save2npy(data,str_path):
    np.save(str_path, data)
    return

def getMeanStd(data_df,column):
    data_df=data_df.loc[:,column]
    mean=data_df.mean().values
    sigma=data_df.std().values
    return mean,sigma

def timemark(time_str):
    timeList=time_str.tolist()
    times=[]
    for i in timeList:
        time = pd.to_datetime(i,dayfirst=True)
        DayOfYear = (time.dayofyear - 1)/ 365.0 - 0.5
        DayOfMonth = (time.day - 1)/30.0 - 0.5
        DayOfWeek = time.dayofweek   / 6.0 - 0.5
        HourOfDay = time.hour   / 23.0 - 0.5
        MinuteOfHour = time.minute / 59.0 - 0.5
        stack_time = np.vstack((DayOfYear,DayOfMonth,DayOfWeek,HourOfDay,MinuteOfHour)).transpose((1,0))
        times.append(stack_time)
    times_mark=np.array(times)
    return times_mark

def time_integer_mark(time_str):
    timeList=time_str.tolist()
    times=[]
    for i in timeList:
        time = pd.to_datetime(i,dayfirst=True)
        DayOfYear = time.dayofyear
        DayOfMonth =time.day
        DayOfWeek = time.dayofweek
        HourOfDay = time.hour
        MinuteOfHour = time.minute
        stack_time = np.vstack((DayOfYear,DayOfMonth,DayOfWeek,HourOfDay,MinuteOfHour)).transpose((1,0))
        times.append(stack_time)
    times_mark=np.array(times)
    return times_mark


def main():
    # 读取csv文件
    df = pd.read_csv(data_path)
    #初始化
    traningSet=[]
    time_str=[]
    #获取合法index
    valid_index=list(check_data(df))
    column=df.columns.values
    column=np.delete(column,[list(column).index('Date Time'),list(column).index('p (mbar)'),list(column).index('wv (m/s)'),list(column).index('max. wv (m/s)'),list(column).index('wd (deg)')])
    #获取mean和std
    mean,sigma=getMeanStd(df,column)
    #获得特征样本和标签样本
    for i in valid_index:
        traningSet.append(df.loc[i:i+seq_len+pred_len-1,column].values)
        time_str.append(df.loc[ i:i+seq_len+pred_len-1,'Date Time'].values)
    traningSet=np.array(traningSet)
    time_str = np.array(time_str)
    times_mark=timemark(time_str)
    times_integer_mark = time_integer_mark(time_str)
    x,y,z=traningSet.shape[0],traningSet.shape[1],traningSet.shape[2]
    input_std_data=standardization(traningSet,mean,sigma)

    # save2npy(data=input_std_data,str_path='input_std_data.npy')
    # save2npy(data=times_mark,str_path='times_mark.npy')
    save2npy(data=times_integer_mark, str_path='times_integer_mark.npy')

    return


if __name__ == "__main__":
    main()


