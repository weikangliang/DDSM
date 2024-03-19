# coding=utf-8
import matplotlib.pyplot as plt

# 定义数据
x = [1, 2, 3, 4]
y = [0.5222, 0.5027, 0.5200, 0.5209]

# 创建折线图
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')

# 设置图表标题和坐标轴标签
plt.title('Y Values Over X')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# 设置Y轴的显示范围，略小于最小值，略大于最大值，增加区分度
plt.ylim(min(y) - 0.005, max(y) + 0.005)

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
