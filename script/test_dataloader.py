# coding=utf-8
from __future__ import print_function
import numpy
from data_iterator_new import DataIterator
from model_new import *
import time
import random
import sys
import argparse
from utils import *
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0


def prepare_data(input, target, maxlen=None, user_maxlen=None, his_item_user_maxlen=5):  # [128,x]
    # input: a list of sentences
    lengths_x = [len(s[4]) for s in input]  # 历史行为序列的长度
    seqs_mid = [inp[3] for inp in input]  # [128,20]
    seqs_cat = [inp[4] for inp in input]  # [128,20]
    seqs_time = [inp[5] for inp in input]  # [128,20]

    lengths_s_user = [len(s[6]) for s in input]  # [128] 目标物品的历史购买用户的长度，记录的是下面n的具体值
    seqs_user = [inp[6] for inp in input]  # [128,50]目标物品的历史购买用户
    seqs_user_mid = [inp[7] for inp in input]  # [128,50,20]
    seqs_user_cat = [inp[8] for inp in input]  # [128,50,20]
    seqs_user_time = [inp[9] for inp in input]  # [128,50]目标物品的历史购买用户购买目标物品的时间
    noclk_seqs_mid = [inp[10] for inp in input]  # [128,20,5]
    noclk_seqs_cat = [inp[11] for inp in input]  # [128,20,5]
    item_user_mid_length = 0
    # -----------------------------当前用户历史购买物品的历史购买用户 -----------------------------
    his_item_user_list = [inp[12] for inp in input]  # [128,20,k]
    his_item_user_time_list = [inp[13] for inp in input]  # [128,20,k]

    if maxlen is not None:
        new_lengths_x = []
        new_seqs_mid = []
        new_seqs_cat = []
        new_seqs_time = []

        new_seqs_user_mid = []
        new_seqs_user_cat = []

        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []

        new_his_item_user_list = []
        new_his_item_user_time_list = []
        # ------------------------------------------------截断与当前用户历史行为有关的长度-----------------------------------------------------------------
        for l_x, inp in zip(lengths_x, input):  # 每次遍历批次的一个用户
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_seqs_time.append(inp[5][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[10][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[11][l_x - maxlen:])
                new_his_item_user_list.append(inp[12][l_x - maxlen:])
                new_his_item_user_time_list.append(inp[13][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_seqs_time.append(inp[5])
                new_noclk_seqs_mid.append(inp[10])
                new_noclk_seqs_cat.append(inp[11])
                new_his_item_user_list.append(inp[12])
                new_his_item_user_time_list.append(inp[13])
                new_lengths_x.append(l_x)
        # ------------------------------------------------------截断目标物品的购买用户的历史行为的长度--------------------------------------------------------------
        for inp in input:  # 遍历购买目标物品的历史用户
            one_sample_user_mid = []
            one_sample_user_cat = []
            for user_mid in inp[7]:  # 列表（用户）的列表（行为），因为目标物品的历史购买用户也有历史行为，这里同样限制一下它们的历史行为长度
                len_user_mid = len(user_mid)  # 每个历史用户的历史行为数量
                if len_user_mid > maxlen:
                    item_user_mid_length = maxlen
                    one_sample_user_mid.append(user_mid[len_user_mid - maxlen:])
                else:
                    if len_user_mid > item_user_mid_length:  # 这里记录历史序列最大的长度（小于maxlen的时候才记录，否则item_user_mid_length=maxlen）
                        item_user_mid_length = len_user_mid
                    one_sample_user_mid.append(user_mid)
            new_seqs_user_mid.append(one_sample_user_mid)

            for user_cat in inp[8]:
                len_user_cat = len(user_cat)
                if len_user_cat > maxlen:
                    one_sample_user_cat.append(user_cat[len_user_cat - maxlen:])
                else:
                    one_sample_user_cat.append(user_cat)
            new_seqs_user_cat.append(one_sample_user_cat)
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        lengths_x = new_lengths_x  # 当前用户截断后的历史行为长度

        seqs_mid = new_seqs_mid  # 截断后的历史物品
        seqs_cat = new_seqs_cat  # 截断后的历史物品类别
        seqs_time = new_seqs_time  # 截断后的历史物品类别

        seqs_user_mid = new_seqs_user_mid  # 截断后的目标物品的历史购买用户的历史物品
        seqs_user_cat = new_seqs_user_cat  # 截断后的目标物品的历史购买用户的历史物品类别

        noclk_seqs_mid = new_noclk_seqs_mid  # 截断后的辅助损失物品
        noclk_seqs_cat = new_noclk_seqs_cat  # 截断后的辅助损失物品类别

        his_item_user_list = new_his_item_user_list  # 截断后的历史行为的历史购买用户
        his_item_user_time_list = new_his_item_user_time_list  # 截断后的历史行为的历史购买用户购买时间

    # ----------------------------------------------截断目标物品的历史购买用户长度（行为长度上面已经截断）-----------------------------------------------------------
    if user_maxlen is not None:
        new_seqs_user = []
        new_lengths_s_user = []
        new_seqs_user_mid = []
        new_seqs_user_cat = []
        new_seqs_user_time = []
        for l_x, inp in zip(lengths_s_user, input):
            if l_x > user_maxlen:
                new_seqs_user.append(inp[6][l_x - user_maxlen:])
                new_lengths_s_user.append(user_maxlen)
                new_seqs_user_time.append(inp[9][l_x - user_maxlen:])
            else:
                new_seqs_user.append(inp[6])
                new_lengths_s_user.append(l_x)
                new_seqs_user_time.append(inp[9])
        for one_sample_user_mid in seqs_user_mid:
            len_one_sample_user_mid = len(one_sample_user_mid)
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

        seqs_user = new_seqs_user
        seqs_user_mid = new_seqs_user_mid
        seqs_user_cat = new_seqs_user_cat
        seqs_user_time = new_seqs_user_time
    # --------------------------------------------------------下面是开始填充的操作，填充的值为0----------------------------------------------------
    n_samples = len(seqs_mid)
    # maxlen_x = numpy.max(lengths_x)#这里的话就是不定的了，下面的都填成一定的
    maxlen_x = maxlen
    user_maxlen_x = user_maxlen
    # user_maxlen_x = numpy.max(lengths_s_user)
    user_maxlen_x = user_maxlen_x if user_maxlen_x > 0 else 1
    item_user_mid_length = item_user_mid_length if item_user_mid_length > 0 else 1
    neg_samples = len(noclk_seqs_mid[0][0])
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
    his_item_user_time_list_pad = numpy.zeros((n_samples, maxlen_x, his_item_user_maxlen)).astype('int64')

    for idx, x in enumerate(his_item_user_list):  # 遍历128 [128,m,k]->[m,k]
        for idy, y in enumerate(x):  # 遍历每个历史用户 [m,k]->[k]，注意这里的n是不一定长的对于每个目标物品来说购买它的用户长度可能是[1,2,3]，因为seqs_user_mid只是截断了，没有填充
            his_item_user_mask[idx, idy, :len(y)] = 1.0

            his_item_user_list_pad[idx, idy, :len(y)] = y
            his_item_user_time_list_pad[idx, idy, :len(y)] = his_item_user_time_list[idx][idy]

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
            item_user_his_mid_mask[idx, idy, :len(y)] = 1.0

            item_user_his_mid[idx, idy, :len(y)] = y
            item_user_his_cat[idx, idy, :len(y)] = seqs_user_cat[idx][idy]

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    return uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, numpy.array(target), numpy.array(lengths_x), numpy.array(lengths_s_user), noclk_mid_his, noclk_cat_his, his_item_user_list_pad, his_item_user_mask


def eval(sess, test_data, model, model_path, maxlen, user_maxlen):
    import math
    from sklearn import metrics
    y_true = []
    y_pred = []
    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    nums = 0
    logloss = 0.
    sample_num = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his, item_behaviors_user_list, item_behaviors_user_list = prepare_data(src, tgt, maxlen, user_maxlen)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, time_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his])
        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            sample_num += 1
            # logloss += -1.0*(t*math.log(p)+(1-t)*math.log(1-p))
            y_true.append(t)
            y_pred.append(p)
            stored_arr.append([p, t])
    test_auc = metrics.roc_auc_score(y_true, y_pred)
    test_f1 = metrics.f1_score(numpy.round(y_true), numpy.round(y_pred))
    Logloss = metrics.log_loss(y_true, y_pred)
    # test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / nums
    loss_sum = loss_sum / nums
    aux_loss_sum / nums
    global best_auc
    if best_auc < test_auc:
        best_auc = test_auc
        model.save(sess, model_path)
    return test_auc, test_f1, accuracy_sum, Logloss


def train(train_file, test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen, user_maxlen, test_iter, model_type, seed):
    txt_log_test1 = open("../dataset/tesstt11", 'w')
    txt_log_test2 = open("../dataset/tesstt22", 'w')
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        for itr in range(3):
            for src, tgt in train_data:
                print(src,tgt, file=txt_log_test1)
                txt_log_test1.flush()
                print(prepare_data(src, tgt, maxlen, user_maxlen), file=txt_log_test2)
                txt_log_test2.flush()


dirname = "Beauty"
filename = "Beauty"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="DMIN")
    parser.add_argument('--train_file', default="../dataset/{}/local_train_sample_sorted_by_time1".format(dirname))
    parser.add_argument('--test_file', default="../dataset/{}/local_test_sample_sorted_by_time1".format(dirname))
    parser.add_argument('--uid_voc', default="../dataset/{}/uid_voc.pkl".format(dirname))
    parser.add_argument('--mid_voc', default="../dataset/{}/mid_voc.pkl".format(dirname))
    parser.add_argument('--cat_voc', default="../dataset/{}/cat_voc.pkl".format(dirname))
    parser.add_argument('--max_len', type=int, default=20)  # 用户历史行为序列的最大长度，Electronic设置为15，books设置为20
    parser.add_argument('--user_maxlen', type=int, default=50)  # 用户历史行为序列的最大长度，Electronic设置为15，books设置为20
    parser.add_argument('--train_iter', type=int, default=100)
    # ------------------------------这些训练输出是需要根据数据量调节的------------------------------
    parser.add_argument('--test_iter', type=int, default=500)
    parser.add_argument('--save_iter', type=int, default=200)
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.1不行，太大了。学习率过大会导致rnn梯度爆炸：Infinity in summary histogram for: rnn_2/GRU_outputs2
    parser.add_argument('--lr_decay_steps', type=int, default=10000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=3)
    # ---------------------------注意以下这些维度如果做交互的话，需要统一一下---------------------------
    parser.add_argument('--user_emb_dim', type=int, default=36)
    parser.add_argument('--item_emb_dim', type=int, default=18)
    parser.add_argument('--cat_emb_dim', type=int, default=18)
    parser.add_argument('--embedding_dim', type=int, default=18)
    parser.add_argument('--attention_dim', type=int, default=36)
    parser.add_argument('--rnn_hidden_dim', type=int, default=18)
    args = parser.parse_args()

    txt_log_path_train = '../txt_log/' + TIMESTAMP + '_' + args.model_type + '_' + filename + '_train'
    txt_log_path_test = '../txt_log/' + TIMESTAMP + '_' + args.model_type + '_' + filename + '_test'
    txt_log_train = open(txt_log_path_train, 'w')
    txt_log_test = open(txt_log_path_test, 'w')

    tf.set_random_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)

    train(model_type=args.model_type, train_file=args.train_file, test_file=args.test_file, uid_voc=args.uid_voc, mid_voc=args.mid_voc, cat_voc=args.cat_voc, batch_size=args.batch_size, maxlen=args.max_len, user_maxlen=args.user_maxlen, test_iter=args.test_iter, seed=args.seed)
