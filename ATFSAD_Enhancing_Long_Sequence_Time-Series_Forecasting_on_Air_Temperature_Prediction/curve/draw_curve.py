
import matplotlib.pyplot as plt
import numpy as np


time=60*3 #单位：秒
startindex=50


#临时
model1_train_path='train.npy'
model1_vali_path='vali.npy'


plt.ion()
while 1:
    plt.clf()
    try :
        model1_trainNpy = np.load(model1_train_path)[startindex:]
        model1_valiNpy = np.load(model1_vali_path)[startindex:]
        plt.subplot(3, 1, 1)
        plt.text(model1_trainNpy.shape[-1], model1_trainNpy[-1], round(model1_trainNpy[-1], 3))
        plt.plot(model1_trainNpy, label='train')
        plt.legend()
    except :
        model1_trainNpy = np.load(model1_train_path)
        model1_valiNpy = np.load(model1_vali_path)
        plt.subplot(3, 1, 1)
        plt.plot(model1_trainNpy, label='train')
        plt.text(model1_trainNpy.shape[-1], model1_trainNpy[-1], round(model1_trainNpy[-1], 3))
        plt.legend()





    plt.subplot(3, 1, 2)
    plt.plot(model1_valiNpy,label='vali')
    plt.text(model1_valiNpy.shape[-1],model1_valiNpy[-1],     round( model1_valiNpy[-1], 3) )
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(model1_trainNpy, label='train')
    plt.plot(model1_valiNpy,label='vali')
    plt.legend()
    plt.savefig('C:\\Users\\a1067\\OneDrive\\train_curve\\jena')
    plt.pause(time)
plt.ioff()
plt.show()
