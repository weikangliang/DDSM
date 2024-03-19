# coding=utf-8
from __future__ import print_function

import math


def count_unique_users_items(file_path):
    user_ids = set()
    item_ids = set()

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('\t')
            if len(data) != 4:
                continue  # Skip any malformed lines
            user_id, item_id = data[0], data[1]
            user_ids.add(user_id)
            item_ids.add(item_id)
    return len(user_ids), len(item_ids)


def count_unique_items_categories(file_path):
    item_ids = set()
    categories = set()

    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('\t')
            if len(data) != 2:
                continue  # Skip any malformed lines
            item_id, category = data[0], data[1]
            item_ids.add(item_id)
            categories.add(category)

    return len(item_ids), len(categories)


def count_lines(filename):
    try:
        with open(filename, 'r') as file:
            return sum(1 for _ in file)
    except IOError:
        return "File not found or unable to read file."


# Similar to the previous task, we'll write a Python function to read the file and count the unique categories.

def count_unique_categories(file_path):
    category_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split('\t')
            if len(data) != 6:
                continue  # Skip any malformed lines
            category_id = data[-1]
            category_ids.add(category_id)
    return len(category_ids)


data_name = "Movies_and_TV"
reviews_file_path = "../dataset/{}/reviews-info".format(data_name)
item_file_path = "../dataset/{}/item-info".format(data_name)
all_sample_file_name = '../dataset/{}/local_all_sample_sorted_by_time'.format(data_name)
file_path_cat = "../dataset/{}/jointed-time-new".format(data_name)
num_cat = count_unique_categories(file_path_cat)

num_user_id, num_item_id = count_unique_users_items(reviews_file_path)
num_all_item_id, num_all_cat_id = count_unique_items_categories(item_file_path)
print("{}-reviews-info的用户数为：".format(data_name), num_user_id)
print("{}-reviews-info的物品数为：".format(data_name), num_item_id)
print("{}-reviews-info的类别数为：".format(data_name), num_cat)
# print("{}-item-info总的物品数为：".format(data_name), num_all_item_id)
# print("{}-item-info总的类别数为：".format(data_name), num_all_cat_id)
lines = count_lines(reviews_file_path)
print("{}总的交互数为：".format(data_name), lines)
# all_sample_lines = count_lines(all_sample_file_name)
# print("{}总的样本数为：".format(data_name), all_sample_lines)
# print("{}总的batch数为：".format(data_name), math.ceil(all_sample_lines / 128.0))

print("{}总的稀疏度为：".format(data_name), float(lines)/(float(num_user_id)*float(num_item_id)))