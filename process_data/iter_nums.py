# -*- coding: utf-8 -*-
import math
def count_lines(filename):
    try:
        with open(filename, 'r') as file:
            return sum(1 for _ in file)
    except IOError:
        return "File not found or unable to read file."


# 示例使用，替换 'your_file.txt' 为你的文件名
file_name = '../dataset/Movies_and_TV/local_train_sample_sorted_by_time'
lines=count_lines(file_name)
print(lines)
print(math.ceil(lines / 128.0))
