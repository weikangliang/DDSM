# coding=utf-8
from __future__ import print_function

user_map = {}
file_name = "Movies_and_TV"
# 读取文件并聚合数据
with open("../dataset/{}/reviews-info".format(file_name), "r") as file_review:
    for line in file_review:
        items = line.strip().split("\t")
        user_id = items[0]  # 用户ID
        item_id = items[1]  # 物品ID
        timestamp = items[3]  # 购买时间戳

        # 更新 user_map
        user_behaviors = user_map.get(user_id, [])
        user_behaviors.append((item_id, timestamp))
        user_map[user_id] = user_behaviors
# 排序，并保留item_id和timestamp
sorted_items_in_user_seq = {user_id: sorted(behaviors, key=lambda x: x[1]) for user_id, behaviors in user_map.items()}
# 写入新文件
with open("../dataset/{}/user_behavior_sequence.csv".format(file_name), "w") as output_file:
    for user_id, behaviors in sorted_items_in_user_seq.items():
        formatted_behaviors = [u'{}#{}'.format(item_id, timestamp) for item_id, timestamp in behaviors]
        formatted_line = u"{},".format(user_id) + unichr(2).join(formatted_behaviors)
        print(formatted_line, file=output_file)
