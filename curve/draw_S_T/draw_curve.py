
import matplotlib.pyplot as plt
import numpy as np

std=8.4233
mean=9.45014

#临时
path='S_T_Label.npy'
my_array = np.load(path)

for i in range(6):
    plt.figure()
    meanall = my_array[0, i, :, 0]
    season=my_array[7,i,:,0]
    plt.subplot(3, 3, 1)
    plt.plot(meanall*std+mean, label='mean',color='blue')
    plt.plot(my_array[9, i, :, 0]*std+mean, label='label',linestyle = 'solid',color='green')
    plt.legend()
    sum=meanall+season
    for j in range(6):
        plt.subplot(3, 3, j+2)
        sum=sum+my_array[j+1,i,:,0]
        plt.plot(sum*std+mean, label=j,color='blue')
        plt.plot(my_array[9,i,:,0]*std+mean, label='label',color='green')
        # plt.plot(my_array[8, i, :, 0]*std+mean, label='predict',linestyle = '--')
        plt.legend()


plt.show()

