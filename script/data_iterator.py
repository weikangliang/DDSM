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


class DataIterator:
    def __init__(self, source, uid_voc_path, mid_voc_path, cat_voc_path, batch_size=128, maxlen=100, skip_empty=False, shuffle_each_epoch=False, sort_by_length=True, max_batch_size=20, minlen=None):
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
            self.source_buffer.reverse()  # 因为下面的pop是从后面弹出来，所以这里先反转一下

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
                # 获得历史行为物品序列
                tmp = []
                for fea in ss[4].split("|"):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp
                # 获得历史行为类别序列
                tmp1 = []
                for fea in ss[5].split("|"):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                item_bhvs_uid_feats = []
                item_bhvs_uid_time = []
                item_bhvs_id_feats = []
                item_bhvs_cat_feats = []

                all_item_bhvs = ss[6].split(";")
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
                        tmp_gap = float(ss[7]) / 3600.0 / 24.0 - float(arr[3]) / 3600.0 / 24.0 + 1.
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

                noclk_mid_list = []
                noclk_cat_list = []
                for pos_mid in mid_list:
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

                source.append([uid, mid, cat, mid_list, cat_list, item_bhvs_uid_feats, item_bhvs_id_feats, item_bhvs_cat_feats, item_bhvs_uid_time, noclk_mid_list, noclk_cat_list])
                target.append([float(ss[0]), 1 - float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
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
