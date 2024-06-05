



import numpy as np
import matplotlib.pyplot as plt
import torch

def SinusoidalEncoding(seq_len, d_model):
    pos_table = np.array([
        [pos / np.power(10, 2 * i / d_model) for i in range(d_model)]
        for pos in range(seq_len)])
    pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
    pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
    return torch.FloatTensor(pos_table)





def get_absolute_position_encoding(max_length, d_model):
    position = np.arange(max_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 1) * -(np.log(10.0) / d_model))
    sin = np.sin(position * div_term)
    cos = np.cos(position * div_term)
    encoding = np.concatenate([sin, cos], axis=1)
    return encoding

def plot_absolute_position_encoding(max_length, d_model):
    encoding = get_absolute_position_encoding(max_length, d_model)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(encoding, cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.xlim((0, d_model))
    plt.ylabel('Position')
    plt.ylim((0, max_length))
    plt.colorbar()
    plt.show()

plot_absolute_position_encoding(max_length=288, d_model=512)