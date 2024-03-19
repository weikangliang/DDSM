# -*- coding: utf-8 -*-
y_true = [1, 0, 1, 1, 0]  # 假设的真实值列表
y_pred = [1, 0, 0, 1, 1]  # 假设的预测值列表

# 写入y_true到文件
with open('y_true.txt', 'w') as file:
    for value in y_true:
        file.write("%s\n" % value)

# 写入y_pred到文件
with open('y_pred.txt', 'w') as file:
    for value in y_pred:
        file.write("%s\n" % value)
