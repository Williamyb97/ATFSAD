

import torch
import torch.nn as nn
import torch.nn.functional as F
import math







def rmse(x,y):
    mse = nn.MSELoss()
    rmse_loss=torch.sqrt(mse(x,y))
    return rmse_loss





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
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class Series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(Series_decomp_multi, self).__init__()
        self.moving_avg = [Moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))
    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean

class  Creat_inps(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.series_decomp_multi=Series_decomp_multi(args.moving_avg)
    def forward(self, batch_en_x,batch_den_x,batch_en_time,batch_den_time):
        #encoder输入
        batch_en_x=batch_en_x
        batch_en_time=batch_en_time
        #decoder输入
        batch_den_time=batch_den_time
        season,trend=self.series_decomp_multi(batch_den_x)
        mean = torch.mean(batch_en_x, dim=1).unsqueeze(1).repeat(1, self.args.pred_len, 1).to(self.args.device)
        zeros = torch.zeros([batch_en_x.shape[0], self.args.pred_len, batch_en_x.shape[2]]).to(self.args.device)
        seasonal_init=torch.cat([season[:, -self.args.help_len:, :], zeros], dim=1).to(self.args.device)
        trend_init = torch.cat([trend[:, -self.args.help_len:, :], mean], dim=1).to(self.args.device)
        return batch_en_x,batch_en_time,seasonal_init,batch_den_time,trend_init
