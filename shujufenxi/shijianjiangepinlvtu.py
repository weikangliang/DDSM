# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt

# 确保 Matplotlib 使用 Unicode 字符串
plt.rcParams['font.sans-serif'] = ['SimSun']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False

# 文件路径
file_paths = [
    '../dataset/Phones/local_all_sample_sorted_by_time',
    '../dataset/Beauty/local_all_sample_sorted_by_time',
    '../dataset/Video_Games/local_all_sample_sorted_by_time',
    '../dataset/Sports/local_all_sample_sorted_by_time',
    '../dataset/Movies_and_TV/local_all_sample_sorted_by_time'
]

# 类别名称
categories = ['Phones', 'Beauty', 'Video_Games','Sports', 'Movies_and_TV']

# 初始化空的DataFrame列表来存储每个文件的结果
dfs = []

# 处理每个文件
for file_path, category in zip(file_paths, categories):
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # 使用制表符分割每一行
            if len(parts) >= 7:  # 确保行有足够的部分
                # 获取最后一列（购买时间）和历史交互的最后时间
                purchase_time = int(parts[-1])
                last_interaction_time = int(parts[6].split('|')[-1])
                # 计算间隔并添加到列表中
                interval = purchase_time - last_interaction_time
                intervals.append(interval)

    # 创建DataFrame来存储和展示结果
    df_intervals = pd.DataFrame(intervals, columns=['Interval'])
    df_intervals['Category'] = category
    dfs.append(df_intervals)

# 合并所有DataFrame
df_all = pd.concat(dfs)

# 定义桶的边界
max_interval = df_all['Interval'].max()
bucket_edges = [0, 3600, 86400, 86400*7, 86400*30, 86400*90, max_interval]  # 定义为1小时、1天、1周、1月、3月以及最大值的边界
bucket_labels = ['<=1小时', '<=1天', '1天-1周', '1周-1月', '1月-3月', '>3月']

# 使用cut函数分桶
df_all['IntervalBucket'] = pd.cut(df_all['Interval'], bins=bucket_edges, labels=bucket_labels)
# 定义每个类别的颜色
category_colors = {
    'Phones': (59/255,98/255,145/255),
    'Beauty': (148/255,60/255,57/255),
    'Video_Games': (119/255,144/255,67/255),
    'Sports': (98/255,76/255,124/255),
    'Movies_and_TV': (56/255,132/255,152/255)
}
# 绘制柱状图
import seaborn as sns

# 计算每个类别的时间间隔桶的频率百分比
df_grouped = df_all.groupby(['Category', 'IntervalBucket'], observed=True).size().unstack(fill_value=0).apply(lambda x: x / x.sum() * 100, axis=1)

# 重置索引，便于绘图
df_grouped = df_grouped.reset_index()

# 将DataFrame转换为“长格式”，以适应seaborn的需要
df_melted = df_grouped.melt(id_vars='Category', var_name='IntervalBucket', value_name='Percentage')
palette = sns.color_palette([category_colors[cat] for cat in categories])
# 绘制分组条形图
plt.figure(figsize=(15, 10))
sns.barplot(x='IntervalBucket', y='Percentage', hue='Category', data=df_melted, palette=palette)

plt.title('当前与最近交互物品的时间间隔分布', fontsize=35)
plt.xlabel('时间间隔', fontsize=30)
plt.ylabel('占比 (%)', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=30)
plt.tight_layout()
plt.show()