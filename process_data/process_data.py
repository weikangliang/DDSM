# coding=utf-8
from __future__ import print_function
import sys
import random
import time
import numpy

dir_name = 'Clothing'
file_name = 'Clothing'


def process_meta():
    fi = open('../dataset/{}/meta_{}.json'.format(dir_name, file_name), "r")
    fo = open('../dataset/{}/item-info'.format(dir_name), "w")
    for line in fi:
        obj = eval(line)
        cat = obj["categories"][0][-1]
        print(obj["asin"] + "\t" + cat, file=fo)

process_meta()
def process_reviews():
    fi = open('../dataset/{}/reviews_{}_5.json'.format(dir_name, file_name), "r")
    fo = open('../dataset/{}/reviews-info'.format(dir_name), "w")
    for line in fi:
        obj = eval(line)
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        rating = obj["overall"]
        time = obj["unixReviewTime"]
        print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), file=fo)

process_reviews()
def data_stats():
    f_rev = open('../dataset/{}/reviews-info'.format(dir_name), "r")
    user_map = {}
    item_map = {}
    sample_list = []
    num = 0
    for line in f_rev:
        num += 1
        line = line.strip()
        items = line.split("\t")
        sample_list.append((line, float(items[-1])))
        if items[0] not in user_map:
            user_map[items[0]] = []
        user_map[items[0]].append(items[1])  # 每个用户购买了多少物品
        if items[1] not in item_map:
            item_map[items[1]] = []
        item_map[items[1]].append(items[0])  # 每个物品被多少用户购买
    print("review num:{}".format(num))
    print("user num:{}".format(len(user_map)))
    print("item num:{}".format(len(item_map)))

    user_length = [len(user_map[k]) for k in user_map]
    item_length = [len(item_map[k]) for k in item_map]
    user_length_max = numpy.max(user_length)
    item_length_max = numpy.max(item_length)
    user_length_avg = numpy.mean(user_length)
    item_length_avg = numpy.mean(item_length)
    print(user_length_max, item_length_max, user_length_avg, item_length_avg)

data_stats()
def manual_join_as_time():  # 1	A281NPSIMI1C2R	B0000535UX	5.0	1023840000	Hand Soaps
    f_rev = open('../dataset/{}/reviews-info'.format(dir_name), "r")
    item_list = []  # 用来构造负样本
    sample_list = []
    for line in f_rev:
        line = line.strip()
        items = line.split("\t")
        sample_list.append((line, float(items[-1])))  # sample_list就是原来的一行多了时间的元组
        item_list.append(items[1])
    print("sample size:{}".format(len(item_list)))
    sample_list = sorted(sample_list, key=lambda x: x[1])  # 先将sample_list按照时间排序，后面（下一个文件）就不用按照全体的排序了
    # 构建物品-类别字符串形式的词典
    f_meta = open('../dataset/{}/item-info'.format(dir_name), "r")
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]

    fo = open("../dataset/{}/jointed-time-new".format(dir_name), "w")
    for line in sample_list:
        items = line[0].split("\t")
        asin = items[1]
        j = 0
        while True:
            asin_neg_index = random.randint(0, len(item_list) - 1)
            asin_neg = item_list[asin_neg_index]
            if asin_neg == asin:
                continue
            items[1] = asin_neg
            print("0" + "\t" + "\t".join(items) + "\t" + meta_map[asin_neg], file=fo)
            j += 1
            if j == 1:  # negative sampling frequency
                break
        if asin in meta_map:
            print("1" + "\t" + line[0] + "\t" + meta_map[asin], file=fo)
        else:
            print("1" + "\t" + line[0] + "\t" + "default_cat", file=fo)
manual_join_as_time()

maxlen = 20
user_maxlen = 50


def get_all_samples():
    fin = open("../dataset/{}/jointed-time-new".format(dir_name), "r")
    fall = open("../dataset/{}/local_all_sample_sorted_by_time".format(dir_name), "w")
    user_his_items = {}
    user_his_cats = {}
    item_his_users = {}
    line_idx = 0
    for line in fin:
        items = line.strip().split("\t")
        clk = int(items[0])
        user = items[1]
        item_id = items[2]
        timestamp = items[4]
        cat = items[5]
        # 收集用户的历史行为序列
        if user in user_his_items:
            bhvs_items = user_his_items[user][-maxlen:]
        else:
            bhvs_items = []
        if user in user_his_cats:
            bhvs_cats = user_his_cats[user][-maxlen:]
        else:
            bhvs_cats = []
        user_history_clk_num = len(bhvs_items)
        bhvs_items_str = "|".join(bhvs_items)
        bhvs_cats_str = "|".join(bhvs_cats)

        # 构建物品的行为序列
        if item_id in item_his_users:
            item_clk_users = item_his_users[item_id][-user_maxlen:]  # 这里不是单纯的用户，而是用户连带着它的历史行为（购买的物品和类别，以及购买当前物品的时间而不是所有历史行为的时间）
        else:
            item_clk_users = []

        history_users_feats = ";".join(item_clk_users)  # 如果item_clk_users为空这里会变成''而不是' ';"a" + "\t" + '' + "\t" + "a"--->['a', '', 'a']
        if user_history_clk_num >= 1:  # 如果用户的历史行为大于1， 8 is the average length of user behavior
            print(items[0] + "\t" + user + "\t" + item_id + "\t" + cat + "\t" + bhvs_items_str + "\t" + bhvs_cats_str + "\t" + history_users_feats + "\t" + timestamp, file=fall)
        if clk:  # 只有正样本才添加进历史行为
            if user not in user_his_items:
                user_his_items[user] = []
                user_his_cats[user] = []
            user_his_items[user].append(item_id)
            user_his_cats[user].append(cat)

            if item_id not in item_his_users:
                item_his_users[item_id] = []
            if user_history_clk_num >= 1:
                item_bhvs_feat = user + '_' + bhvs_items_str + '_' + bhvs_cats_str + '_' + timestamp
            else:
                item_bhvs_feat = user + '_' + '' + '_' + '' + '_' + timestamp
            if user_history_clk_num >= 1:
                item_his_users[item_id].append(item_bhvs_feat)
        line_idx += 1
get_all_samples()

def get_cut_time(percent=0.85):
    time_list = []
    fin = open("../dataset/{}/local_all_sample_sorted_by_time".format(dir_name), "r")
    for line in fin:
        line = line.strip()
        time = float(line.split("\t")[-1])
        time_list.append(time)
    sample_size = len(time_list)
    print(sample_size)
    train_size = int(sample_size * percent)
    time_list = sorted(time_list, key=lambda x: x)
    cut_time = time_list[train_size]
    return cut_time


def split_test_by_time(cut_time):
    fin = open("../dataset/{}/local_all_sample_sorted_by_time".format(dir_name), "r")
    ftrain = open("../dataset/{}/local_train_sample_sorted_by_time".format(dir_name), "w")
    ftest = open("../dataset/{}/local_test_sample_sorted_by_time".format(dir_name), "w")
    for line in fin:
        line = line.strip()
        time = float(line.split("\t")[-1])
        if time <= cut_time:
            print(line, file=ftrain)
        else:
            print(line, file=ftest)

split_test_by_time(get_cut_time(percent=0.85))
