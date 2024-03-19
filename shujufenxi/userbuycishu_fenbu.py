# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
# 首先读取新上传的文件
new_file_path = '../dataset/Movies_and_TV/item_behavior_sequence.csv'

# 读取文件内容
with open(new_file_path, 'r') as file:
    new_lines = file.readlines()

# 解析数据，计算每个用户购买的序列长度
purchase_lengths = []
for line in new_lines:
    user, purchases = line.strip().split(',', 1)
    items = purchases.split('\x02')  # 使用分隔符分割物品序列
    purchase_lengths.append(len(items))
filtered_purchase_lengths = [length for length in purchase_lengths if length <= 100]
# 转换为DataFrame以方便处理
purchase_df = pd.DataFrame(filtered_purchase_lengths, columns=['PurchaseLength'])

# 绘制直方图，每10个序列长度为一个桶
plt.figure(figsize=(10, 6))
purchase_df['PurchaseLength'].plot(kind='hist', bins=range(0, purchase_df['PurchaseLength'].max() + 5, 5), rwidth=0.8)
plt.title('Distribution of Purchase Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
