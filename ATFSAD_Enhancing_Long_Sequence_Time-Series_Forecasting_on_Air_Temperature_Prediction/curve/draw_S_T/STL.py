import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import pandas as pd

# 创建示例时间序列数据
# data = pd.Series(np.random.randn(365*5), index=pd.date_range('20150101', periods=365*5))


path='D:\\D\\python_code\\ProbSparse_STL\\data\\jena_climate\\jena_climate_2009_2016.csv'
data = pd.read_csv(path).loc[:144,'T (degC)']

# 使用 STL 算法进行时间序列分解
stl = STL(data, period=4)
res = stl.fit()

# 绘制分解结果
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].plot(data)
ax[0, 0].set_title('Original time series Y')
ax[0, 0].set_ylabel('Temperature(Celsius)')

ax[0, 1].plot(res.trend)
ax[0, 1].set_title('Trend Component T')
# ax[0, 1].set_ylabel('Temperature(Celsius)')

ax[1, 0].plot(res.seasonal)
ax[1, 0].set_title('Seasonal Component S')
ax[1, 0].set_ylabel('Temperature(Celsius)')

ax[1, 1].plot(res.resid)
ax[1, 1].set_title('Residual Component R')
# ax[1, 1].set_ylabel('Temperature(Celsius)')

plt.tight_layout()
plt.show()