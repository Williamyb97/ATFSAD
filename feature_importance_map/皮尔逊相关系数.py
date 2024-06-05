
import pandas as pd
import  numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns



data_path='..\data\jena_climate\jena_climate_2009_2016.csv'
df = pd.read_csv(data_path)
train_corr = df.corr().abs()



plt.figure(1)
sns.heatmap(train_corr,annot=True, vmax=1, square=True,cmap='rocket',linewidths=1,linecolor='grey')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
# plt.savefig('figure.png', format='png', compression=0)
