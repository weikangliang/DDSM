# coding=utf-8
from __future__ import print_function

import argparse
import random
import sys
from datetime import datetime
import os
import numpy

from data_iterator_new import DataIterator
from model_new import *
from model_new_test import myModel_base_test_aux
from utils import *

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
EMBEDDING_DIM = 18
HIDDEN_SIZE = EMBEDDING_DIM * 2
ATTENTION_SIZE = EMBEDDING_DIM * 2
best_auc = 0.0


def prepare_data(input, target, maxlen=None, user_maxlen=None, his_item_user_maxlen=5):  # [128,x]
    # input: a list of sentences
    lengths_x = [len(s[3]) for s in input]  # 历史行为序列的长度
    seqs_mid = [inp[3] for inp in input]  # [128,20]
    seqs_cat = [inp[4] for inp in input]  # [128,20]
    seqs_time = [inp[5] for inp in input]  # [128,20]

    lengths_s_user = [len(s[6]) for s in input]  # [128] 目标物品的历史购买用户的长度，记录的是下面n的具体值
    seqs_user = [inp[6] for inp in input]  # [128,50]目标物品的历史购买用户
    seqs_user_mid = [inp[7] for inp in input]  # [128,50,20]
    seqs_user_cat = [inp[8] for inp in input]  # [128,50,20]
    seqs_user_tiv = [inp[9] for inp in input]  # [128,50,20]
    seqs_user_time = [inp[10] for inp in input]  # [128,50]目标物品的历史购买用户购买目标物品的时间间隔

    noclk_seqs_mid = [inp[11] for inp in input]  # [128,20,5]
    noclk_seqs_cat = [inp[12] for inp in input]  # [128,20,5]
    item_user_mid_length = 0
    # -----------------------------当前用户历史购买物品的历史购买用户 -----------------------------
    his_item_user_list = [inp[13] for inp in input]  # [128,20,k]
    his_item_users_tiv_list = [inp[14] for inp in input]  # [128,20,k]

    # ----------------------------------------------截断行为序列长度（包括当前用户和历史用户的行为序列）-----------------------------------------------------------
    if maxlen is not None:
        new_lengths_x = []
        new_seqs_mid = []
        new_seqs_cat = []
        new_seqs_time = []

        new_seqs_user_mid = []
        new_seqs_user_cat = []
        new_seqs_user_tiv = []

        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []

        new_his_item_user_list = []
        new_his_item_users_tiv_list = []
        # ------------------------------------------------截断与当前用户历史行为有关的长度-----------------------------------------------------------------
        for l_x, inp in zip(lengths_x, input):  # 每次遍历批次的一个用户
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_seqs_time.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[11][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[12][l_x - maxlen:])
                new_his_item_user_list.append(inp[13][l_x - maxlen:])
                new_his_item_users_tiv_list.append(inp[14][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_seqs_time.append(inp[5])
                new_noclk_seqs_mid.append(inp[11])
                new_noclk_seqs_cat.append(inp[12])
                new_his_item_user_list.append(inp[13])
                new_his_item_users_tiv_list.append(inp[14])
                new_lengths_x.append(l_x)
        # ------------------------------------------------------截断目标物品的购买用户的历史行为的长度--------------------------------------------------------------
        for inp in input:  # 遍历目标物品的历史购买用户
            one_sample_user_mid = []
            for user_mids in inp[7]:  # 列表（用户）的列表（行为），因为目标物品的历史购买用户也有历史行为，这里同样限制一下它们的历史行为长度
                len_user_mids = len(user_mids)  # 每个历史用户的历史行为数量
                if len_user_mids > maxlen:
                    item_user_mid_length = maxlen
                    one_sample_user_mid.append(user_mids[len_user_mids - maxlen:])
                else:
                    if len_user_mids > item_user_mid_length:  # 这里记录历史序列最大的长度（小于maxlen的时候才记录，否则item_user_mid_length=maxlen）
                        item_user_mid_length = len_user_mids
                    one_sample_user_mid.append(user_mids)
            new_seqs_user_mid.append(one_sample_user_mid)
            one_sample_user_cat = []
            for user_cat in inp[8]:
                len_user_cat = len(user_cat)
                if len_user_cat > maxlen:
                    one_sample_user_cat.append(user_cat[len_user_cat - maxlen:])
                else:
                    one_sample_user_cat.append(user_cat)
            new_seqs_user_cat.append(one_sample_user_cat)
            one_sample_user_tiv = []
            for user_tiv in inp[9]:
                len_user_tiv = len(user_tiv)
                if len_user_tiv > maxlen:
                    one_sample_user_tiv.append(user_tiv[len_user_tiv - maxlen:])
                else:
                    one_sample_user_tiv.append(user_tiv)
            new_seqs_user_tiv.append(one_sample_user_tiv)
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        lengths_x = new_lengths_x  # 当前用户截断后的历史行为长度

        seqs_mid = new_seqs_mid  # 截断后的历史物品
        seqs_cat = new_seqs_cat  # 截断后的历史物品类别
        seqs_time = new_seqs_time  # 截断后的历史物品类别

        seqs_user_mid = new_seqs_user_mid  # 截断后的目标物品的历史购买用户的历史物品
        seqs_user_cat = new_seqs_user_cat  # 截断后的目标物品的历史购买用户的历史物品类别
        seqs_user_tiv = new_seqs_user_tiv  # 截断后的目标物品的历史购买用户的历史行为之间的时间间隔

        noclk_seqs_mid = new_noclk_seqs_mid  # 截断后的辅助损失物品
        noclk_seqs_cat = new_noclk_seqs_cat  # 截断后的辅助损失物品类别

        his_item_user_list = new_his_item_user_list  # 截断后的历史行为的历史购买用户
        his_item_users_tiv_list = new_his_item_users_tiv_list  # 截断后的历史行为的历史购买用户购买时间

    # ----------------------------------------------截断目标物品的历史购买用户长度（行为长度上面已经截断）-----------------------------------------------------------
    if user_maxlen is not None:
        new_seqs_user = []
        new_lengths_s_user = []
        new_seqs_user_mid = []
        new_seqs_user_cat = []
        new_seqs_user_tiv = []
        new_seqs_user_time = []
        for l_x, inp in zip(lengths_s_user, input):
            if l_x > user_maxlen:
                new_seqs_user.append(inp[6][l_x - user_maxlen:])
                new_lengths_s_user.append(user_maxlen)
                new_seqs_user_time.append(inp[10][l_x - user_maxlen:])
            else:
                new_seqs_user.append(inp[6])
                new_lengths_s_user.append(l_x)
                new_seqs_user_time.append(inp[10])
        for one_sample_user_mid in seqs_user_mid:
            len_one_sample_user_mid = len(one_sample_user_mid)  # 实际上就是等同于l_x
            if len_one_sample_user_mid > user_maxlen:
                new_seqs_user_mid.append(one_sample_user_mid[len_one_sample_user_mid - user_maxlen:])
            else:
                new_seqs_user_mid.append(one_sample_user_mid)
        for one_sample_user_cat in seqs_user_cat:
            len_one_sample_user_cat = len(one_sample_user_cat)
            if len_one_sample_user_cat > user_maxlen:
                new_seqs_user_cat.append(one_sample_user_cat[len_one_sample_user_cat - user_maxlen:])
            else:
                new_seqs_user_cat.append(one_sample_user_cat)
        for one_sample_user_tiv in seqs_user_tiv:
            len_one_sample_user_tiv = len(one_sample_user_tiv)
            if len_one_sample_user_tiv > user_maxlen:
                new_seqs_user_tiv.append(one_sample_user_tiv[len_one_sample_user_tiv - user_maxlen:])
            else:
                new_seqs_user_tiv.append(one_sample_user_tiv)

        seqs_user = new_seqs_user
        seqs_user_mid = new_seqs_user_mid
        seqs_user_cat = new_seqs_user_cat
        seqs_user_tiv = new_seqs_user_tiv
        seqs_user_time = new_seqs_user_time
    # --------------------------------------------------------下面是开始填充的操作，填充的值为0----------------------------------------------------
    n_samples = len(seqs_mid)  # batch_size：128
    # maxlen_x = numpy.max(lengths_x)#这里的话就是不定的了，下面的都填成一定的
    maxlen_x = maxlen
    # user_maxlen_x = numpy.max(lengths_s_user)
    user_maxlen_x = user_maxlen
    user_maxlen_x = user_maxlen_x if user_maxlen_x > 0 else 1
    item_user_mid_length = item_user_mid_length if item_user_mid_length > 0 else 1
    neg_samples = len(noclk_seqs_mid[0][0])  # 一个历史行为多少个辅助损失物品：5
    # print("maxlen_x, user_maxlen_x, item_user_mid_length:", maxlen_x, user_maxlen_x, item_user_mid_length)
    # 当前用户的历史行为
    mid_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    cat_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    time_his = numpy.zeros((n_samples, maxlen_x)).astype('int64')
    # 目标物品的历史购买用户和行为
    item_user_his = numpy.zeros((n_samples, user_maxlen_x)).astype('int64')
    item_user_his_time = numpy.zeros((n_samples, user_maxlen_x)).astype('int64')
    item_user_his_mid = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('int64')
    item_user_his_cat = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('int64')
    item_user_his_tiv = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('int64')
    # 全部的mask
    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    his_item_user_mask = numpy.zeros((n_samples, maxlen_x, his_item_user_maxlen)).astype('float32')
    item_user_his_mask = numpy.zeros((n_samples, user_maxlen_x)).astype('float32')
    item_user_his_mid_mask = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('float32')
    # 辅助损失
    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    # 当前用户历史行为的历史购买用户
    his_item_user_list_pad = numpy.zeros((n_samples, maxlen_x, his_item_user_maxlen)).astype('int64')
    his_item_users_tiv_list_pad = numpy.zeros((n_samples, maxlen_x, his_item_user_maxlen)).astype('int64')

    # 填充历史行为的历史购买用户
    for idx, x in enumerate(his_item_user_list):  # 遍历128 [128,20,10]->[20,10]
        for idy, y in enumerate(x):  # 遍历每个历史用户 [20,10]->[10]，注意这里的n是不一定长的，对于每个目标物品来说购买它的用户长度可能是[1,2,3]，因为seqs_user_mid只是截断了，没有填充
            his_item_user_mask[idx, idy, :len(y)] = 1.0
            his_item_user_list_pad[idx, idy, :len(y)] = y
            his_item_users_tiv_list_pad[idx, idy, :len(y)] = his_item_users_tiv_list[idx][idy]

    for idx, [s_x, s_y, s_t, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, seqs_time, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.
        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        time_his[idx, :lengths_x[idx]] = s_t
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    # 填充目标物品的历史购买用户和购买时间
    for idx, [x, t] in enumerate(zip(seqs_user, seqs_user_time)):
        item_user_his_mask[idx, :len(x)] = 1.
        item_user_his[idx, :len(x)] = x
        item_user_his_time[idx, :len(t)] = t

    # 填充目标物品的历史购买用户的历史购买物品序列
    for idx, x in enumerate(seqs_user_mid):  # 遍历128 [128,n,m]->[n,m]
        for idy, y in enumerate(x):  # 遍历每个历史用户 [n,m]->[m]，注意这里的n是不一定长的对于每个目标物品来说购买它的用户长度可能是[1,2,3]，因为seqs_user_mid只是截断了，没有填充
            # print("len(y)",y)
            item_user_his_mid_mask[idx, idy, :len(y)] = 1.0
            item_user_his_mid[idx, idy, :len(y)] = y
            item_user_his_cat[idx, idy, :len(y)] = seqs_user_cat[idx][idy]
            item_user_his_tiv[idx, idy, :len(y)] = seqs_user_tiv[idx][idy]

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])
    loss_tiv_weight = numpy.array([inp[15] for inp in input])
    return uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_tiv, item_user_his_mid_mask, numpy.array(target), numpy.array(lengths_x), numpy.array(lengths_s_user), noclk_mid_his, noclk_cat_his, his_item_user_list_pad, his_item_user_mask, his_item_users_tiv_list_pad,loss_tiv_weight

count = 1

def should_test(iter, iter_per_epoch=2000, test_iter=100, test_window=200):
    global count
    if iter % test_iter == 0 and count * iter_per_epoch - test_window <= iter <= count * iter_per_epoch + test_window:
        if iter == count * iter_per_epoch + test_window:
            count += 1
        return True
    else:
        return False

def eval(sess, test_data, model, best_model_path, maxlen, user_maxlen):
    from sklearn import metrics
    y_true = []
    y_pred = []
    loss_sum = 0.
    accuracy_sum = 0.
    nums = 0
    sample_num = 0

    for src, tgt in test_data:
        uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_tiv, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his, his_item_user_list_pad, his_item_user_mask, his_item_users_tiv_list_pad,loss_tiv_weight = prepare_data(src, tgt, maxlen, user_maxlen, his_item_user_maxlen=args.his_item_max_his_user_count)
        prob, loss, acc, aux_loss,context_weights_matrices1,context_weights_matrices2 = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat,item_user_his_tiv, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, his_item_user_list_pad, his_item_user_mask, his_item_users_tiv_list_pad, noclk_mid_his, noclk_cat_his,loss_tiv_weight])
        nums += 1
        loss_sum += loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            sample_num += 1
            y_true.append(t)
            y_pred.append(p)

    test_auc = metrics.roc_auc_score(y_true, y_pred)
    test_f1 = metrics.f1_score(numpy.round(y_true), numpy.round(y_pred))
    Logloss = metrics.log_loss(y_true, y_pred)
    accuracy = metrics.accuracy_score(numpy.round(y_true), numpy.round(y_pred))
    global best_auc
    if best_auc < test_auc:
        parts = best_model_path.split('/')
        model_name = parts[-3]
        data_name = parts[-2]

        y_true_path = '../dataset/{}/{}/y_true.txt'.format(data_name,model_name)
        if not os.path.exists(os.path.dirname(y_true_path)):
            os.makedirs(os.path.dirname(y_true_path))
        with open(y_true_path, 'w') as file:
            y_true_reverse = y_true
            y_true_reverse.reverse()
            for value in y_true_reverse:
                file.write("%s\n" % value)
        y_pred_path = '../dataset/{}/{}/y_pred.txt'.format(data_name, model_name)
        if not os.path.exists(os.path.dirname(y_pred_path)):
            os.makedirs(os.path.dirname(y_pred_path))
        with open(y_pred_path, 'w') as file:
            y_pred_reverse = y_pred
            y_pred_reverse.reverse()
            for value in y_pred_reverse:
                file.write("%s\n" % value)
        best_auc = test_auc
        model.save(sess, best_model_path)
        print("----------------------------------best_test_auc: %.4f----------------------------------" % test_auc)
    return test_auc, test_f1, accuracy, Logloss


def train(train_file, test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, user_maxlen, test_iter, model_type, seed):
    parts = uid_voc.split('/')
    data_name = parts[-2]
    best_model_path = "../dnn_best_model/" + model_type +'/'+ data_name + "/ckpt_noshuff"
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, his_item_max_his_user_count=args.his_item_max_his_user_count)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, his_item_max_his_user_count=args.his_item_max_his_user_count)
        n_uid, n_mid, n_cat = train_data.get_n()
        print("n_uid, n_mid, n_cat:", n_uid, n_mid, n_cat)
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'FM':
            model = Model_FM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'LR':
            model = Model_LR(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN_FRNet':
            model = Model_DIN_FRNet(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN_Dynamic_MLP':
            model = Model_DIN_Dynamic_MLP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'CAN':
            model = CAN_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_neg':
            model = DIEN_with_neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_TIEN':
            model = DIEN_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_neg_high':
            model = DIEN_with_neg_high(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_BST':
            model = Model_BST(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN':
            model = Model_DNN_Multi_Head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DNN_Multi_Head_contrasive':
            model = Model_DNN_Multi_Head_contrasive(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_test':
            model = Model_DNN_Multi_Head_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'TIEN':
            model = Model_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DUMN_origin':
            model = Model_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN_test':
            model = Model_DUMN_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_SENET':
            model = Model_DNN_Multi_Head_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN':
            model = Model_DMIN_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN_origin':
            model = Model_DMIN_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN_origin_test':
            model = Model_DMIN_DUMN_origin_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_no_cls':
            model = DRINK_no_cls(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_origin':
            model = DRINK_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_test':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_FRnet':
            model = DRINK_FRnet(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_DIB':
            model = DRINK_time(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_time':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_base':
            model = myModel_base(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_test_comirec':
            model = myModel_test_comirec(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_base_test_aux':
            model = myModel_base_test_aux(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self':
            model = myModel_0_last_Self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self_notiv':
            model = myModel_0_last_Self_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self_test':
            model = myModel_0_last_Self_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head':
            model = myModel_0_last_Context_multi_head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_new':
            model = myModel_0_last_Context_multi_head_new(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_test':
            model = myModel_0_last_Context_multi_head_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_new_DIB':
            model = myModel_0_last_Context_multi_head_new_DIB(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_test_bit_wsie':
            model = myModel_0_last_Context_test_bit_wsie(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_test_vector_wsie':
            model = myModel_0_last_Context_test_vector_wsie(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_nodecouple':
            model = myModel_0_last_Context_multi_head_nodecouple(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_notiv':
            model = myModel_0_last_Context_multi_head_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0':
            model = myModel0(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_DMIN':
            model = myModel0_test_DMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1':
            model = myModel1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2':
            model = myModel2(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test':
            model = myModel0_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_origin':
            model = myModel0_test_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_1user':
            model = myModel0_test_1user(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short':
            model = myModel0_test_only_short(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_old':
            model = myModel0_test_only_short_old(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_1':
            model = myModel0_test_only_short_1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_DIB':
            model = myModel0_test_only_short_DIB(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_user_self':
            model = myModel0_test_user_self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_user_item_self':
            model = myModel0_test_user_item_self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_notiv':
            model = myModel0_test_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_usernotiv':
            model = myModel0_test_usernotiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_item_usernotiv':
            model = myModel0_test_item_usernotiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test':
            model = myModel1_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_SENET':
            model = myModel1_test_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv':
            model = myModel1_test_with_user_tiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_GRU':
            model = myModel1_test_with_user_tiv_GRU(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_DIEN':
            model = myModel1_test_with_user_tiv_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_DMIN':
            model = myModel1_test_with_user_tiv_DMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_target_item':
            model = myModel1_test_with_target_item(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_test':
            model = myModel1_test_with_user_tiv_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_SENET':
            model = myModel1_test_with_user_tiv_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DDSM1_TIEN':
            model = DDSM1_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DDSM1_DRINK':
            model = DDSM1_DRINK(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test':
            model = myModel2_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_no_postiv':
            model = myModel2_test_no_postiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_1':
            model = myModel2_test_1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_2':
            model = myModel2_test_2(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_3':
            model = myModel2_test_3(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_IUI':
            model = Model_IUI(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print("Invalid model_type : %s", model_type)
            return
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        test_auc, test_f1, accuracy_sum, Logloss = eval(sess, test_data, model, best_model_path, maxlen, user_maxlen)
        print('                                                                                   test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % (test_auc, test_f1, accuracy_sum, Logloss))
        sys.stdout.flush()
        print('test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % (test_auc, test_f1, accuracy_sum, Logloss), file=txt_log_test)
        txt_log_test.flush()
        iter = 0
        lr = args.lr
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        for itr in range(args.epochs):
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_tiv, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his, his_item_user_list_pad, his_item_user_mask, his_item_users_tiv_list_pad,loss_tiv_weight = prepare_data(src, tgt, maxlen, user_maxlen, his_item_user_maxlen=args.his_item_max_his_user_count)
                loss, acc, aux_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, time_his, mid_mask,
                                                         item_user_his, item_user_his_mask, item_user_his_time,
                                                         item_user_his_mid, item_user_his_cat,item_user_his_tiv, item_user_his_mid_mask,
                                                         target, current_user_his_item_length, target_item_his_user_length, lr, his_item_user_list_pad, his_item_user_mask, his_item_users_tiv_list_pad, noclk_mid_his, noclk_cat_his,loss_tiv_weight])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += aux_loss
                iter += 1
                sys.stdout.flush()
                if should_test(iter, iter_per_epoch=args.iter_per_epoch, test_iter=args.test_iter, test_window=args.test_window):
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter))
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter), file=txt_log_train)
                    txt_log_train.flush()
                    test_auc, test_f1, test_accuracy_sum, Logloss = eval(sess, test_data, model, best_model_path, maxlen, user_maxlen)
                    print('                                                                                   test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % (test_auc, test_f1, test_accuracy_sum, Logloss))
                    print('iter %d ---- test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % ((iter,) + (test_auc, test_f1, test_accuracy_sum, Logloss)), file=txt_log_test)
                    txt_log_test.flush()
            # 下面的是结束一个epoch之前再进行最后一次测试，一般最好的结果出现在这里
            print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter))
            print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter), file=txt_log_train)
            txt_log_train.flush()
            test_auc, test_f1, test_accuracy_sum, Logloss = eval(sess, test_data, model, best_model_path, maxlen, user_maxlen)
            print('                                                                                   test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % (test_auc, test_f1, test_accuracy_sum, Logloss))
            print('iter %d ---- test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % ((iter,) + (test_auc, test_f1, test_accuracy_sum, Logloss)), file=txt_log_test)
            txt_log_test.flush()
            lr *= 0.5  # 学习率衰减，其实可以改一下方式


def test(train_file, test_file, uid_voc, mid_voc, cat_voc, batch_size, user_maxlen, maxlen, model_type, seed):
    parts = uid_voc.split('/')
    data_name = parts[-2]
    model_path = tf.train.latest_checkpoint("../dnn_best_model/" + model_type  + '/' + data_name)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'LR':
            model = Model_LR(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'FM':
            model = Model_FM(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN_FRNet':
            model = Model_DIN_FRNet(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DIN_Dynamic_MLP':
            model = Model_DIN_Dynamic_MLP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'CAN':
            model = CAN_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)#Model_DIEN_CAN
        elif model_type == 'DIEN_neg':
            model = DIEN_with_neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_TIEN':
            model = DIEN_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_neg_high':
            model = DIEN_with_neg_high(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN':
            model = Model_DNN_Multi_Head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_BST':
            model = Model_BST(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DNN_Multi_Head_contrasive':
            model = Model_DNN_Multi_Head_contrasive(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_SENET':
            model = Model_DNN_Multi_Head_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_test':
            model = Model_DNN_Multi_Head_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'TIEN':
            model = Model_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN':
            model = Model_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DUMN_origin':
            model = Model_DUMN_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN':
            model = Model_DMIN_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN_origin':
            model = Model_DMIN_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_DMIN_DUMN_origin_test':
            model = Model_DMIN_DUMN_origin_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_no_cls':
            model = DRINK_no_cls(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_origin':
            model = DRINK_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_test':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_DIB':
            model = DRINK_time(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_time':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_FRnet':
            model = DRINK_FRnet(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_base':
            model = myModel_base(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_test_comirec':
            model = myModel_test_comirec(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_base_test_aux':
            model = myModel_base_test_aux(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0':
            model = myModel0(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self':
            model = myModel_0_last_Self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self_notiv':
            model = myModel_0_last_Self_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Self_test':
            model = myModel_0_last_Self_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head':
            model = myModel_0_last_Context_multi_head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_new':
            model = myModel_0_last_Context_multi_head_new(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_test':
            model = myModel_0_last_Context_multi_head_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_new_DIB':
            model = myModel_0_last_Context_multi_head_new_DIB(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_test_bit_wsie':
            model = myModel_0_last_Context_test_bit_wsie(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_test_vector_wsie':
            model = myModel_0_last_Context_test_vector_wsie(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_nodecouple':
            model = myModel_0_last_Context_multi_head_nodecouple(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel_0_last_Context_multi_head_notiv':
            model = myModel_0_last_Context_multi_head_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DDSM1_TIEN':
            model = DDSM1_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DDSM1_DRINK':
            model = DDSM1_DRINK(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1':
            model = myModel1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2':
            model = myModel2(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test':
            model = myModel0_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_origin':
            model = myModel0_test_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_1user':
            model = myModel0_test_1user(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short':
            model = myModel0_test_only_short(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_old':
            model = myModel0_test_only_short_old(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_1':
            model = myModel0_test_only_short_1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_only_short_DIB':
            model = myModel0_test_only_short_DIB(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_user_self':
            model = myModel0_test_user_self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_user_item_self':
            model = myModel0_test_user_item_self(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_notiv':
            model = myModel0_test_notiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_usernotiv':
            model = myModel0_test_usernotiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_item_usernotiv':
            model = myModel0_test_item_usernotiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel0_test_DMIN':
            model = myModel0_test_DMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test':
            model = myModel1_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_SENET':
            model = myModel1_test_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv':
            model = myModel1_test_with_user_tiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_GRU':
            model = myModel1_test_with_user_tiv_GRU(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_DIEN':
            model = myModel1_test_with_user_tiv_DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_DMIN':
            model = myModel1_test_with_user_tiv_DMIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_target_item':
            model = myModel1_test_with_target_item(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_test':
            model = myModel1_test_with_user_tiv_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel1_test_with_user_tiv_SENET':
            model = myModel1_test_with_user_tiv_SENET(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test':
            model = myModel2_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_no_postiv':
            model = myModel2_test_no_postiv(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_1':
            model = myModel2_test_1(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_2':
            model = myModel2_test_2(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel2_test_3':
            model = myModel2_test_3(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Model_IUI':
            model = Model_IUI(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print("Invalid model_type : %s", model_type)
            return
        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_F1: %.4f ---- test_accuracy: %.4f ---- aux_loss: %.4f' % eval(sess, test_data, model, model_path, maxlen, user_maxlen))


dirname = "Sports"  # Movies_and_TV20900, Tools,Grocery1800,Baby1900,Toys2000,Phones2200,Beauty2300,Video_Games2800,Sports3400,Clothing3100,Home_and_Kitchen6400,Kindle,Apps8800,Home_and_Kitchen6400
filename = "Sports"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="Model_GRU4REC", choices=["DIN", "DIEN", "DIEN_neg", "DMIN", "myModel_0_last_Self", "myModel_0_last_Context", "DUMN", "DRINK", "myModel0_test", "myModel1_test"])
    parser.add_argument('--train_file', default="../dataset/{}/local_train_sample_sorted_by_time".format(dirname))
    parser.add_argument('--test_file', default="../dataset/{}/local_test_sample_sorted_by_time".format(dirname))
    parser.add_argument('--uid_voc', default="../dataset/{}/uid_voc.pkl".format(dirname))  # myModel_0_last_Self,myModel_0_last_Context
    parser.add_argument('--mid_voc', default="../dataset/{}/mid_voc.pkl".format(dirname))
    parser.add_argument('--cat_voc', default="../dataset/{}/cat_voc.pkl".format(dirname))
    parser.add_argument('--his_beh_maxlen', type=int, default=20)  # 这里是目标物品的历史用户最多带的历史行为数
    parser.add_argument('--user_maxlen', type=int, default=50)  # 目标物品历史用户的最大长度
    parser.add_argument('--train_iter', type=int, default=100)
    parser.add_argument('--his_item_max_his_user_count', type=int, default=20)  # 当前用户每个历史物品的历史购买用户的最大值
    parser.add_argument('--epochs', type=int, default=3)
    # ------------------------------这些训练输出是需要根据数据量调节的------------------------------
    parser.add_argument('--test_iter', type=int, default=100)
    parser.add_argument('--iter_per_epoch', type=int, default=3400)
    parser.add_argument('--test_window', type=int, default=200)
    parser.add_argument('--save_iter', type=int, default=200)
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.1不行，太大了。学习率过大会导致rnn梯度爆炸：Infinity in summary histogram for: rnn_2/GRU_outputs2
    parser.add_argument('--lr_decay_steps', type=int, default=10000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    # ---------------------------注意以下这些维度如果做交互的话，需要统一一下---------------------------
    parser.add_argument('--user_emb_dim', type=int, default=36)
    parser.add_argument('--item_emb_dim', type=int, default=18)
    parser.add_argument('--cat_emb_dim', type=int, default=18)
    parser.add_argument('--embedding_dim', type=int, default=18)
    parser.add_argument('--attention_dim', type=int, default=36)
    parser.add_argument('--rnn_hidden_dim', type=int, default=18)
    args = parser.parse_args()

    print("model:", args.model_type, "file:", filename)
    txt_log_path_train = '../txt_log/' + TIMESTAMP + '_' + args.model_type + '_' + filename + '_train' + ""
    txt_log_path_test = '../txt_log/' + TIMESTAMP + '_' + args.model_type + '_' + filename + '_test' + ""
    txt_log_train = open(txt_log_path_train, 'w')
    txt_log_test = open(txt_log_path_test, 'w')

    tf.set_random_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    type_name = 'train'
    if type_name == 'train':
        train(model_type=args.model_type, train_file=args.train_file, test_file=args.test_file, uid_voc=args.uid_voc, mid_voc=args.mid_voc, cat_voc=args.cat_voc, batch_size=args.batch_size, maxlen=args.his_beh_maxlen, user_maxlen=args.user_maxlen, test_iter=args.test_iter, seed=args.seed)
    elif type_name == 'test':
        test(model_type=args.model_type, train_file=args.train_file, test_file=args.test_file, uid_voc=args.uid_voc, mid_voc=args.mid_voc, cat_voc=args.cat_voc, batch_size=args.batch_size, maxlen=args.his_beh_maxlen, user_maxlen=args.user_maxlen, seed=args.seed)
    else:
        print('do nothing...')
