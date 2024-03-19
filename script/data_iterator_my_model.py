# coding=utf-8
import numpy
import json
import cPickle as pkl
import random
import numpy as np
import gzip
import shuffle
import os
import bisect
import pandas as pd


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
    index = bisect.bisect_left([time for user, time in records], current_timestamp)
    closest_user_list = []
    closest_user_time_list = []
    # 向左查找最近的用户
    left_index = index - 1
    while left_index >= 0 and len(closest_user_list) < max_user_count:
        if records[left_index][1] != current_timestamp:
            user = records[left_index][0]
            if user == current_user:  # 这个用户可能多次购买过该物品
                continue
            closest_user_list.append(source_dicts[0][user] if user in source_dicts[0] else 0)
            closest_user_time_list.append(records[left_index][1])
        left_index -= 1
    return closest_user_list, closest_user_time_list


def build_item_category_dict(file_path):
    item_category_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            item_id, category = parts[0], parts[1]
            item_category_dict[item_id] = category
        return item_category_dict


def generate_behaviors_dict(file):
    sorted_item_behaviors = {}
    # 读取文件
    file_path = file
    # 读取文件
    data = pd.read_csv(file_path, header=None, names=['Item', 'User_Behaviors'], sep=',')
    for _, row in data.iterrows():
        item = row['Item']
        User_Behaviors = row['User_Behaviors'].split('\x02')
        sorted_item_behaviors[item] = [(ut.split('#')[0], ut.split('#')[1]) for ut in User_Behaviors]
    return sorted_item_behaviors


class DataIterator:
    def __init__(self, source, uid_voc_path, mid_voc_path, cat_voc_path, batch_size=128, maxlen=100, skip_empty=False, shuffle_each_epoch=False, sort_by_length=True, max_batch_size=20, minlen=None):

        self.item_category_dict = build_item_category_dict("../dataset/{}/item-info".format(os.path.basename(os.path.split(uid_voc_path)[0])))  # 生成物品到类别的字典
        # 读取物品行为列表和用户行为列表，存放的是字符串形式
        self.sorted_item_behaviors = generate_behaviors_dict("../dataset/{}/item_behavior_sequence.csv".format(os.path.basename(os.path.split(uid_voc_path)[0])))
        self.sorted_user_behaviors = generate_behaviors_dict("../dataset/{}/user_behavior_sequence.csv".format(os.path.basename(os.path.split(uid_voc_path)[0])))

        if shuffle_each_epoch:
            self.source_orig = source
            self.source = shuffle.main(self.source_orig, temporary=True)
        else:
            self.source = fopen(source, 'r')
        self.source_dicts = []
        for source_dict in [uid_voc_path, mid_voc_path, cat_voc_path]:
            self.source_dicts.append(load_dict(source_dict))

        #############################################################################################################
        # 从item-info构建物品-类别字典（目前是字符串形式）
        f_meta = open(os.path.dirname(source) + "/item-info", "r")
        meta_map = {}
        for line in f_meta:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1]
        # 将刚刚的物品-类别字典转换成数字形式
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
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.k = batch_size * max_batch_size

        self.end_of_data = False
        self.gap = np.array([1.1, 1.4, 1.7, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])  # 共有16个取值范围

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
                time_str_list = ss[6].split("|")  # 时间对于接下来构造物品行为序列的时候有用

                # 计算历史行为距离目标物品的时间间隔
                time_gap_list = [int(np.sum((float(ss[8]) / 3600.0 / 24.0 - float(time) / 3600.0 / 24.0 + 1) >= self.gap)) for time in time_str_list]

                item_bhvs_uid_feats = []
                item_bhvs_uid_time = []
                item_bhvs_id_feats = []
                item_bhvs_cat_feats = []

                all_item_bhvs = ss[7].split(";")
                for bhvs in all_item_bhvs:
                    if bhvs.strip() == "":  # 这说明改目标物品一个历史行为也没有
                        break
                    arr = bhvs.split("_")
                    if arr[1].strip() == "":  # 这说明这个用户在购买前没有一个历史行为(不会出现，因为process_data不会将它写进文件)
                        continue
                    bhv_uid = self.source_dicts[0][arr[0]] if arr[0] in self.source_dicts[0] else 0
                    item_bhvs_uid_feats.append(bhv_uid)  # 添加目标物品的一个历史购买用户

                    if arr[1].strip() != "":
                        id_tmp_list = []
                        for fea in arr[1].split("|"):
                            m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                            id_tmp_list.append(m)
                        item_bhvs_id_feats.append(id_tmp_list)  # 目标物品的一个历史购买用户的历史行为物品序列

                    if arr[2].strip() != "":
                        cat_tmp_list = []
                        for fea in arr[2].split("|"):
                            c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                            cat_tmp_list.append(c)
                        item_bhvs_cat_feats.append(cat_tmp_list)  # 目标物品的一个历史购买用户的历史行为类别序列

                    if arr[3].strip() != "":
                        tmp_gap = float(ss[8]) / 3600.0 / 24.0 - float(arr[3]) / 3600.0 / 24.0 + 1.
                        time_gap = int(np.sum(tmp_gap >= self.gap))
                        item_bhvs_uid_time.append(time_gap)  # 记录历史用户购买目标物品的时间

                # read from source file and map to word index

                # if len(mid_list) > self.maxlen:
                #    continue
                if self.minlen != None:
                    if len(mid_list) <= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue
                # 构造历史物品行为序列和辅助损失函数
                item_behaviors_user_list = []
                item_behaviors_time_list = []
                noclk_mid_list = []
                noclk_cat_list = []
                for i, pos_mid in enumerate(mid_list):
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0
                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break
                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)

                    user_list, user_time_list = construct_closest_user_id_list(self.source_dicts, self.sorted_item_behaviors, ss[1], mid_str_list[i], time_str_list[i])
                    item_behaviors_user_list.append(user_list)
                    item_behaviors_time_list.append(user_time_list)

                source.append([uid, mid, cat, mid_list, cat_list, time_gap_list, item_bhvs_uid_feats, item_bhvs_id_feats, item_bhvs_cat_feats, item_bhvs_uid_time, noclk_mid_list, noclk_cat_list, item_behaviors_user_list, item_behaviors_time_list])
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
