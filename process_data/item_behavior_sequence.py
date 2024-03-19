# coding=utf-8
from __future__ import print_function

item_map = {}
file_name = "Movies_and_TV"
# 读取文件并聚合数据
with open("../dataset/{}/reviews-info".format(file_name), "r") as file_review:
    for line in file_review:
        items = line.strip().split("\t")
        user_id = items[0]  # 用户ID
        item_id = items[1]  # 物品ID
        timestamp = items[3]  # 购买时间戳

        # 更新 item_map
        item_behaviors = item_map.get(item_id, [])
        item_behaviors.append((user_id, timestamp))
        item_map[item_id] = item_behaviors
# 排序，并保留item_id和timestamp
sorted_users_in_item_seq = {item_id: sorted(behaviors, key=lambda x: x[1]) for item_id, behaviors in item_map.items()}
# 写入新文件
with open("../dataset/{}/item_behavior_sequence.csv".format(file_name), "w") as output_file:
    for item_id, behaviors in sorted_users_in_item_seq.items():
        formatted_behaviors = [u'{}#{}'.format(user_id, timestamp) for user_id, timestamp in behaviors]
        formatted_line = u"{},".format(item_id) + unichr(2).join(formatted_behaviors)
        print(formatted_line, file=output_file)
