# coding=utf-8
from __future__ import print_function

import bisect
import cPickle as pkl
import gzip
import json
import os
import random

import numpy as np
import pandas as pd

import shuffle


def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key, value) in d.items())


def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(pkl.load(f))


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


# 传递的参数current_user, current_item, current_timestamp，返回两个列表

def construct_closest_user_id_list(source_dicts, sorted_item_behaviors, current_user, current_item, current_timestamp, max_user_count=5):
    if current_item not in sorted_item_behaviors:
        return None

    records = sorted_item_behaviors[current_item]
    index = bisect.bisect_left(records, (current_user, current_timestamp))
    closest_user_list = []
    closest_user_time_list = []
    # 向左查找最近的用户
    left_index = index - 1
    while left_index >= 0 and len(closest_user_list) < max_user_count:
        user = records[left_index][0]
        # if user == current_user:  # 这个用户可能多次购买过该物品
        #     continue
        closest_user_list.append(source_dicts[0][user] if user in source_dicts[0] else 0)
        closest_user_time_list.append(records[left_index][1])
        left_index -= 1
    closest_user_list.reverse(), closest_user_time_list.reverse()
    return closest_user_list, closest_user_time_list


def build_item_category_dict(file_path):
    item_category_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            item_id, category = parts[0], parts[1]
            item_category_dict[item_id] = category
        return item_category_dict


def generate_behaviors_dict(file_path):
    sorted_item_behaviors = {}
    data = pd.read_csv(file_path, header=None, names=['Item', 'User_Behaviors'], sep=',')
    for _, row in data.iterrows():
        item = row['Item']
        User_Behaviors = row['User_Behaviors'].split('\x02')
        sorted_item_behaviors[item] = [(ut.split('#')[0], ut.split('#')[1]) for ut in User_Behaviors]
    return sorted_item_behaviors


class DataIterator:
    def __init__(self, source, uid_voc_path, mid_voc_path, cat_voc_path, batch_size=128, maxlen=100, skip_empty=False, shuffle_each_epoch=False, max_batch_size=20, minlen=None, his_item_max_his_user_count=5):
        # 读取物品行为列表和用户行为列表，存放的是字符串形式
        self.data_name = os.path.basename(os.path.split(uid_voc_path)[0])
        self.sorted_item_behaviors = generate_behaviors_dict("../dataset/{}/item_behavior_sequence.csv".format(os.path.basename(os.path.split(uid_voc_path)[0])))
        self.sorted_user_behaviors = generate_behaviors_dict("../dataset/{}/user_behavior_sequence.csv".format(os.path.basename(os.path.split(uid_voc_path)[0])))
        self.his_item_max_his_user_count = his_item_max_his_user_count
        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc_path, mid_voc_path, cat_voc_path]:
            self.source_dicts.append(load_dict(source_dict))

        # ------------------------------------- 从item-info构建物品-类别字典 -------------------------------------
        f_meta = open(os.path.dirname(source) + "/item-info", "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        # 将刚刚的物品-类别字典（字符串形式）转换成数字形式
        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx
        # -------------------------------------从 "reviews-info" 读取，根据出现的频次生成辅助损失所需要的负样本所需要的id list，也即是根据出现的频次进行随机采样-------------------------------------
        f_review = open(os.path.dirname(source) + "/reviews-info", "r")
        self.mid_list_for_random = []
        for line in f_review:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.shuffle = shuffle_each_epoch

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False
        self.gap = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.7, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])  # 共有16个取值范围

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()  # 迭代器状态被重置
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))  # 去除末尾的换行符
            # self.source_buffer.reverse()  # 因为下面的pop是从后面弹出来，所以这里先反转一下

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            while True:
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0

                mid_str_list = ss[4].split("|")
                mid_list = [self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0 for fea in mid_str_list]
                cat_str_list = ss[5].split("|")
                cat_list = [self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0 for fea in cat_str_list]
                time_str_list = ss[6].split("|")  # 时间对于接下来构造历史物品的历史行为序列的时候有用
                # 计算历史行为距离目标物品的时间间隔
                his_tiv_list = [int(np.sum((float(ss[8]) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0 + 1.) >= self.gap)) for time in time_str_list]

                item_bhvs_uid_feats = []  # 目标物品的历史购买用户
                item_bhvs_uid_time = []  # 目标物品的历史购买用户购买的时间
                item_bhvs_mid_feats = []  # 目标物品的历史购买用户的历史行为物品序列
                item_bhvs_cat_feats = []  # 目标物品的历史购买用户的历史行为类别序列
                item_bhvs_mid_times = []  # 目标物品的历史购买用户的历史行为时间序列

                all_item_bhvs = ss[7].split(";")  # A27493GW3TEX65_B000674XN2_Rollers & Pens_1099526400_1102982400;A1BHCE2409B5QF_B000C1Z6GU|B00064A5L4_Eau de Toilette|Eau de Parfum_1093046400|1120262400_1120262400
                for bhvs in all_item_bhvs:
                    if bhvs.strip() == "":  # 这说明该目标物品一个历史行为也没有
                        break
                    arr = bhvs.split("_")  # [user,bhvs_items_str,bhvs_cats_str,bhvs_times_str, cur_time]
                    if arr[1].strip() == "":  # 这说明这个用户在购买前没有一个历史行为(不会出现，因为process_data不会将它写进文件)
                        continue
                    bhv_uid = self.source_dicts[0][arr[0]] if arr[0] in self.source_dicts[0] else 0
                    item_bhvs_uid_feats.append(bhv_uid)  # 添加目标物品的一个历史购买用户

                    if arr[1].strip() != "":  # bhvs_items_str
                        id_tmp_list = []
                        for fea in arr[1].split("|"):
                            m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                            id_tmp_list.append(m)
                        item_bhvs_mid_feats.append(id_tmp_list)  # 目标物品的一个历史购买用户的历史行为物品序列

                    if arr[2].strip() != "":  # bhvs_cats_str
                        cat_tmp_list = []
                        for fea in arr[2].split("|"):
                            c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                            cat_tmp_list.append(c)
                        item_bhvs_cat_feats.append(cat_tmp_list)  # 目标物品的一个历史购买用户的历史行为类别序列

                    if arr[3].strip() != "":  # bhvs_times_str
                        time_tmp_list = []
                        for fea in arr[3].split("|"):
                            tmp_gap = float(arr[4]) / 3600.0 / 24.0 - float(fea) / 3600.0 / 24.0 + 1.
                            time_gap = int(np.sum(tmp_gap >= self.gap))
                            time_tmp_list.append(time_gap)
                        item_bhvs_mid_times.append(time_tmp_list)  # 目标物品的一个历史购买用户的历史行为类别序列

                    if arr[4].strip() != "":  # cur_time
                        tmp_gap = float(ss[8]) / 3600.0 / 24.0 - float(arr[4]) / 3600.0 / 24.0 + 1.
                        time_gap = int(np.sum(tmp_gap >= self.gap))
                        item_bhvs_uid_time.append(time_gap)  # 记录历史用户购买目标物品的时间距离当前用户购买的时间

                # if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue
                # 构造历史物品的行为序列和辅助损失函数
                his_item_behaviors_user_list = []
                his_item_behaviors_tiv_list = []
                noclk_mid_list = []
                noclk_cat_list = []

                # 随机选择物品的时候，先排除目标候选物品和历史购买过的物品
                except_mid_list = mid_list[:]
                except_mid_list.append(mid)
                for i, pos_mid in enumerate(mid_list):
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid in except_mid_list:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)
                    # 传进去：当前用户，当前购买的历史物品，购买历史物品的时间，历史物品的行为序列长度
                    user_list, user_time_list = construct_closest_user_id_list(self.source_dicts, self.sorted_item_behaviors, ss[1], mid_str_list[i], time_str_list[i], self.his_item_max_his_user_count)
                    his_item_behaviors_user_list.append(user_list)
                    user_tiv_list = [int(np.sum((float(time_str_list[i]) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0 + 1.) >= self.gap)) for time in user_time_list]
                    his_item_behaviors_tiv_list.append(user_tiv_list)
                if self.data_name == "Phones":
                    loss_tiv_range = float(1397692800) / 3600.0 / 24.0 - float(1074729600) / 3600.0 / 24.0  # Phones
                    loss_tiv_weight =( float(ss[8]) / 3600.0 / 24.0 - float(1074729600) / 3600.0 / 24.0 ) / loss_tiv_range
                elif self.data_name == "Beauty":
                    loss_tiv_range = float(1397779200) / 3600.0 / 24.0 - float(1024185600) / 3600.0 / 24.0   # Beauty
                    loss_tiv_weight = (float(ss[8]) / 3600.0 / 24.0 - float(1024185600) / 3600.0 / 24.0)  / loss_tiv_range
                elif self.data_name == "Video_Games":
                    loss_tiv_range = float(1386028800) / 3600.0 / 24.0  - float(942192000) / 3600.0 / 24.0   # Video_Games
                    loss_tiv_weight =( float(ss[8]) / 3600.0 / 24.0  - float(942192000) / 3600.0 / 24.0)  / loss_tiv_range
                elif self.data_name == "Sports":
                    loss_tiv_range = float(1396051200) / 3600.0 / 24.0 - float(1043366400) / 3600.0 / 24.0 # Sports
                    loss_tiv_weight = (float(ss[8]) / 3600.0 / 24.0 - float(1043366400) / 3600.0 / 24.0) / loss_tiv_range
                else:
                    loss_tiv_range = float(1388620800) / 3600.0 / 24.0 - float(879465600) / 3600.0 / 24.0  # Movies_and_TV
                    loss_tiv_weight = (float(ss[8]) / 3600.0 / 24.0  - float(879465600) / 3600.0 / 24.0) / loss_tiv_range

                source.append([uid, mid, cat, mid_list, cat_list, his_tiv_list, item_bhvs_uid_feats, item_bhvs_mid_feats, item_bhvs_cat_feats, item_bhvs_mid_times, item_bhvs_uid_time, noclk_mid_list, noclk_cat_list, his_item_behaviors_user_list, his_item_behaviors_tiv_list, loss_tiv_weight])
                target.append([float(ss[0]), 1 - float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:  # all sentence pairs in maxibatch filtered out because of length
            source, target = self.next()
        return source, target


if __name__ == '__main__':
    train_file = "local_train_sample_sorted_by_time"
    test_file = "local_test_sample_sorted_by_time"
    uid_voc_path = "uid_voc_path.pkl"
    mid_voc_path = "mid_voc_path.pkl"
    cat_voc_path = "cat_voc_path.pkl"
    train_data = DataIterator(train_file, uid_voc_path, mid_voc_path, cat_voc_path, 128, 100, shuffle_each_epoch=False)
    num = 0
    for src, tgt in train_data:
        if num <= 10:
            print(src)
            print(tgt)
        num += 1
    print(num)
