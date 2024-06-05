import argparse
import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import os
from torchinfo import summary
import datetime
import math
import pickle
from torch.utils.data import DataLoader, Dataset
from units.untils import rmse
import time
from data.data_loader import TrainDataset
from model.myformer import Myformer

class Exp_Main():
    def __init__(self, args):
        self.args=args
        self.device = args.device
        self.model=Myformer(args).to(args.device)
        self.model_optim = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss = nn.MSELoss()
        #loss function
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.min_cost=5
        self.trainCurve = []
        self.valiCurve = []
        print(self.model)
        print("####################################################################################################")
        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())
    def train(self):
        self.model.train()
        mae_list = []
        mse_list = []
        rmse_list = []

        trainDataset = TrainDataset(state='train',args=self.args)
        train_loader = DataLoader(dataset=trainDataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        self.args.STList = []
        for batch_index, (batch_en_x,batch_den_x,batch_en_time,batch_den_time,batch_label) in enumerate(train_loader):

            batch_index+=1
            pred=self.model(batch_en_x,batch_den_x,batch_en_time,batch_den_time)
            # Backpropagation
            self.model_optim.zero_grad()
            loss=self.loss(pred,batch_label)
            loss.backward()
            self.model_optim.step()
            if self.args.weatherSTList == True:
                self.args.STList.append(batch_label.clone().cpu().detach().numpy())
                STLnumpy = np.array(self.args.STList)

            mae_list.append(      self.mae(pred*self.args.std+self.args.mean,batch_label*self.args.std+self.args.mean).item()           )
            mse_list.append(      self.mse(pred*self.args.std+self.args.mean,batch_label*self.args.std+self.args.mean).item()           )
            rmse_list.append(      rmse(pred*self.args.std+self.args.mean,batch_label*self.args.std+self.args.mean).item()           )

        mae_avg=np.average(mae_list)
        mse_avg=np.average(mse_list)
        rmse_avg=np.average(rmse_list)

        print('....................................................MAE_Trainset_loss:',mae_avg,'')
        print('....................................................MSE_Trainset_loss:',mse_avg,'')
        print('....................................................RMSE_Trainset_loss:',rmse_avg,'')
        self.trainCurve.append(mae_avg)
        np.save(file='.//curve//train.npy', arr=self.trainCurve)
        if self.min_cost > mae_avg :
            self.min_cost = mae_avg
            if self.args.debug==True:
                pass
            else:
                torch.save(self.model.state_dict(), ".//model_weight//ProbSparse_STL.pth")
        if self.args.weatherSTList == True:
            np.save('.\\curve\\draw_S_T\\S_T_Label.npy', STLnumpy)
        return


    def vali(self):
        self.model.eval()
        valiDataset = TrainDataset(state='vali',args=self.args)
        train_loader = DataLoader(dataset=valiDataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True)
        mae_list = []
        mse_list = []
        rmse_list = []
        with torch.no_grad():
            for batch_index, (batch_en_x,batch_den_x,batch_en_time,batch_den_time,batch_label) in enumerate(train_loader):
                batch_index+=1
                pred = self.model(batch_en_x, batch_den_x, batch_en_time, batch_den_time)
                mae_list.append(self.mae(pred * self.args.std + self.args.mean,batch_label * self.args.std + self.args.mean).item())
                mse_list.append(self.mse(pred * self.args.std + self.args.mean,batch_label * self.args.std + self.args.mean).item())
                rmse_list.append(rmse(pred * self.args.std + self.args.mean, batch_label * self.args.std + self.args.mean).item())
        mae_avg=np.average(mae_list)
        mse_avg=np.average(mse_list)
        rmse_avg=np.average(rmse_list)
        print('......................................................................MAE_valiset_loss:',mae_avg,'')
        print('......................................................................MSE_valiset_loss:',mse_avg,'')
        print('......................................................................RMSE_valiset_loss:',rmse_avg,'')
        self.valiCurve.append(np.average(mae_list))
        np.save(file='.//curve//vali.npy', arr=self.valiCurve)
        self.model.train()
        return

    def start(self):
        epochtimes = 0
        while(1):
            epochtimes += 1
            print(f"Epoch {epochtimes}\n---------------------------------------------------------------------------------------------------------------------------------------")
            start = time.time()
            self.train()
            # self.vali()
            end = time.time()
            print('Running time: %s Seconds' % (end - start))
            print('\n')
