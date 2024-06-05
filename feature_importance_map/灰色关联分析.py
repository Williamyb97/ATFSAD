
import pandas as pd
import  numpy as np
from numpy import amin, amax, zeros, mean
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import rankdata

# 灰色关联结果矩阵可视化
# 灰色关联结果矩阵可视化
import seaborn as sns







# 无量纲化
def dimensionlessProcessing(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        MEAN = d.mean()
        newDataFrame[c] = ((d - MEAN) / (MAX - MIN)).tolist()
    return newDataFrame

def GRA_ONE(gray, m=0):
    # 读取为df格式
    gray = dimensionlessProcessing(gray)
    # 标准化
    std = gray.iloc[:, m]  # 为标准要素
    gray.drop(str(m),axis=1,inplace=True)
    ce = gray.iloc[:, 0:]  # 为比较要素
    shape_n, shape_m = ce.shape[0], ce.shape[1]  # 计算行列

    # 与标准要素比较，相减
    a = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            a[i, j] = abs(ce.iloc[j, i] - std[j])

    # 取出矩阵中最大值与最小值
    c, d = amax(a), amin(a)

    # 计算值
    result = zeros([shape_m, shape_n])
    for i in range(shape_m):
        for j in range(shape_n):
            result[i, j] = (d + 0.5 * c) / (a[i, j] + 0.5 * c)

    # 求均值，得到灰色关联值,并返回
    result_list = [mean(result[i, :]) for i in range(shape_m)]
    result_list.insert(m,1)
    return pd.DataFrame(result_list)


def GRA(DataFrame):
    df = DataFrame.copy()
    list_columns = [
        str(s) for s in range(len(df.columns)) if s not in [None]
    ]
    df_local = pd.DataFrame(columns=list_columns)
    df.columns=list_columns
    for i in range(len(df.columns)):
        df_local.iloc[:, i] = GRA_ONE(df, m=i)[0]
    return df_local



def ShowGRAHeatMap(DataFrame,feature_names):
    colormap = plt.cm.RdBu
    f, ax = plt.subplots(figsize=(14, 14))

    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="rocket",
                    annot=True,
                    # mask=mask,
                    xticklabels=feature_names, yticklabels=feature_names
                    )
    plt.show()



data_path='..\data\jena_climate\jena_climate_2009_2016.csv'
df = pd.read_csv(data_path).iloc[:, 1:]
feature_names = df.columns.tolist()
feature_names = [col.split('(')[0].strip() for col in feature_names]


data_wine_gra = GRA(df)
ShowGRAHeatMap(data_wine_gra,feature_names)
