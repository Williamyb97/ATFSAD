import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import  numpy as np



class TrainDataset(Dataset):
    #输入全是npy
    def __init__(self,state,args):
        input_data = np.load(file=args.input_dataPath,allow_pickle=True).astype(np.float)
        timemark = np.load(file=args.timePath).astype(np.float)
        if args.debug==True:
            input_data=input_data[:25,:,:]
            timemark=timemark[:25,:]
        else:
            input_data=input_data[:,:,:]
            timemark=timemark[:,:,:]


        device=args.device
        input_data=torch.from_numpy(input_data).to(device)
        timemark = torch.from_numpy(timemark).to(device)
        input_data=input_data.type(torch.FloatTensor).to(device)
        timemark = timemark.type(torch.FloatTensor).to(device)

        len=input_data.shape[0]
        train_len=int(len*0.7)
        vali_len = int(len * 0.3)

        if state=='train':
            self.en_x=input_data[:train_len,:args.seq_len,:]
            self.en_time = timemark[:train_len, :args.seq_len, :]
            self.den_x=input_data[:train_len,args.seq_len-args.help_len:args.seq_len,:]
            self.den_time = timemark[:train_len, args.seq_len-args.help_len:args.seq_len+args.pred_len, :]
            self.label = input_data[:train_len, -args.pred_len:, args.target_dim:args.target_dim+1]
            self.len=train_len
        elif state == 'vali':
            self.en_x=input_data[-vali_len:,:args.seq_len,:]
            self.en_time = timemark[-vali_len:, :args.seq_len, :]
            self.den_x=input_data[-vali_len:,args.seq_len-args.help_len:args.seq_len,:]
            self.den_time = timemark[-vali_len:, args.seq_len-args.help_len:args.seq_len+args.pred_len, :]
            self.label = input_data[-vali_len:, -args.pred_len:, args.target_dim:args.target_dim+1]
            self.len=vali_len
        else:
            raise NameError('no catched state')
    def __getitem__(self, index):
        return self.en_x[index],self.den_x[index],self.en_time[index],self.den_time[index],self.label[index]
    def __len__(self):
        return self.len


