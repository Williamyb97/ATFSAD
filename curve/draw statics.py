import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 12个人的数字
# data = [0.7271, 0.8057, 0.9961, 1.3367, 1.4021, 1.8807, 1.6117, 1.9714, 2.6008, 2.5551, 2.7544, 2.1548]#1.73
data = [0.7557, 0.78547, 0.96356, 1.20216, 1.26356, 1.99517, 1.7151, 2.0778, 2.45927, 2.59057, 2.50477, 2.21626]#1.7108

data = [round(num, 3) for num in data]

print(sum(data)/len(data))
print(data)




Time = ['00:00~02:00',  '02:00~04:00',    '04:00~06:00',   '06:00~08:00',    '08:00~10:00', '10:00~12:00', '12:00~14:00', '14:00~16:00', '16:00~18:00', '18:00~20:00', '20:00~22:00','22:00~24:00']
# 创建一个画布和一个子图
fig, ax = plt.subplots()

# 设置字体为New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


# 画出柱状图
bars =ax.bar(range(len(data)), data, width=0.5)
# 画出曲线
x = np.linspace(0, len(data)-1, 100)
y = np.interp(x, range(len(data)), data)
ax.plot(x, y, color='orange')
# 设置x轴标签和刻度
ax.set_xticks(range(len(data)))
font = {'family': 'serif', 'size': 18}
ax.set_xticklabels( Time, rotation=45, fontsize=18, fontdict=font)

# 设置y轴标签和范围
font = {'family': 'serif', 'size': 18}
ax.set_ylabel('MAE score', fontsize=18, fontdict=font)
ax.set_ylim(0, 3.5)
# ax.set_xlabel('Time Horizon', fontsize=16)
# 设置 y 轴刻度的字体大小为 14
plt.yticks(fontsize=16)
# 在每个柱子上显示指标值
for i, bar in enumerate(bars):
    ax.annotate(str(data[i]), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=18)

# 显示图形
plt.show()