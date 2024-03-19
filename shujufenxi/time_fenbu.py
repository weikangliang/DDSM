# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
file_path = '../dataset/Movies_and_TV/jointed-time-new'

# 读取文件内容
with open(file_path, 'r') as file:
    lines = file.readlines()

# 将数据转换为DataFrame以便处理
data = [line.strip().split('\t') for line in lines]
df = pd.DataFrame(data, columns=['Label', 'UserID', 'ItemID', 'Rating', 'Timestamp', 'Category'])

# 将时间戳从字符串转换为整数
df['Timestamp'] = pd.to_numeric(df['Timestamp'])

# 将时间戳转换为日期格式
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')

# 创建每三个月的桶
df['Quarter'] = df['Date'].dt.to_period('Q')

# 统计每个桶中的数量
quarter_counts = df['Quarter'].value_counts().sort_index()

# 绘制直方图
plt.figure(figsize=(10, 6))
quarter_counts.plot(kind='bar')
plt.title('Timestamp Distribution by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
