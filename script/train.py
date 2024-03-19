# coding=utf-8
from __future__ import print_function
import numpy
from data_iterator import DataIterator
import tensorflow as tf
from model import *
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


def prepare_data(input, target, maxlen=None, user_maxlen=None):  # [128,x]
    # input: a list of sentences
    lengths_x = [len(s[4]) for s in input]  # 历史行为序列的长度
    seqs_mid = [inp[3] for inp in input]  # [128,m]
    seqs_cat = [inp[4] for inp in input]  # [128,m]
    lengths_s_user = [len(s[5]) for s in input]  # [128] 目标物品的历史购买用户的长度，记录的是下面n的具体值
    seqs_user = [inp[5] for inp in input]  # [128,n]目标物品的历史购买用户
    seqs_user_mid = [inp[6] for inp in input]  # [128,n,m]
    seqs_user_cat = [inp[7] for inp in input]  # [128,n,m]
    seqs_user_time = [inp[8] for inp in input]  # [128,n]目标物品的历史购买用户购买目标物品的时间
    noclk_seqs_mid = [inp[9] for inp in input]  # [128,m,5]
    noclk_seqs_cat = [inp[10] for inp in input]  # [128,m,5]
    item_user_mid_length = 0

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_lengths_x = []
        new_seqs_user_mid = []
        new_seqs_user_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen:])
                new_seqs_cat.append(inp[4][l_x - maxlen:])
                new_noclk_seqs_mid.append(inp[9][l_x - maxlen:])
                new_noclk_seqs_cat.append(inp[10][l_x - maxlen:])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[9])
                new_noclk_seqs_cat.append(inp[10])
                new_lengths_x.append(l_x)
        # ----------------------------------------------------------------------------------------------------------------------------------------------
        for inp in input:  # 遍历购买目标物品的历史用户
            one_sample_user_mid = []
            one_sample_user_cat = []
            for user_mid in inp[6]:  # 因为目标物品的历史购买用户也有历史行为，这里同样限制一下它们的历史行为长度
                len_user_mid = len(user_mid)
                if len_user_mid > maxlen:
                    item_user_mid_length = maxlen
                    one_sample_user_mid.append(user_mid[len_user_mid - maxlen:])
                else:
                    if len_user_mid > item_user_mid_length:  # 这里记录历史序列最大的长度（小于maxlen的时候才记录，否则item_user_mid_length=maxlen）
                        item_user_mid_length = len_user_mid
                    one_sample_user_mid.append(user_mid)
            new_seqs_user_mid.append(one_sample_user_mid)

            for user_cat in inp[7]:
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
        seqs_user_mid = new_seqs_user_mid  # 截断后的目标物品的历史购买用户的历史物品
        seqs_user_cat = new_seqs_user_cat  # 截断后的目标物品的历史购买用户的历史物品类别
        noclk_seqs_mid = new_noclk_seqs_mid  # 截断后的辅助损失物品
        noclk_seqs_cat = new_noclk_seqs_cat  # 截断后的辅助损失物品类别
        #
        # if len(lengths_x) < 1:
        #     return None, None, None, None
        # ----------------------------------------------截断目标物品的历史购买用户长度（行为长度上面已经截断）-----------------------------------------------------------
    if user_maxlen is not None:
        new_seqs_user = []
        new_lengths_s_user = []
        new_seqs_user_mid = []
        new_seqs_user_cat = []
        new_seqs_user_time = []
        for l_x, inp in zip(lengths_s_user, input):
            if l_x > user_maxlen:
                new_seqs_user.append(inp[5][l_x - user_maxlen:])
                new_lengths_s_user.append(user_maxlen)
                new_seqs_user_time.append(inp[8][l_x - user_maxlen:])
            else:
                new_seqs_user.append(inp[5])
                new_lengths_s_user.append(l_x)
                new_seqs_user_time.append(inp[8])
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
    # 目标物品的历史购买用户和行为
    item_user_his = numpy.zeros((n_samples, user_maxlen_x)).astype('int64')
    item_user_his_time = numpy.zeros((n_samples, user_maxlen_x)).astype('int64')
    item_user_his_mid = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('int64')
    item_user_his_cat = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('int64')

    mid_mask = numpy.zeros((n_samples, maxlen_x)).astype('float32')
    item_user_his_mask = numpy.zeros((n_samples, user_maxlen_x)).astype('float32')
    item_user_his_mid_mask = numpy.zeros((n_samples, user_maxlen_x, maxlen_x)).astype('float32')

    noclk_mid_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')
    noclk_cat_his = numpy.zeros((n_samples, maxlen_x, neg_samples)).astype('int64')

    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, :lengths_x[idx]] = 1.

        mid_his[idx, :lengths_x[idx]] = s_x
        cat_his[idx, :lengths_x[idx]] = s_y
        noclk_mid_his[idx, :lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, :lengths_x[idx], :] = no_sy

    for idx, [x, t] in enumerate(zip(seqs_user, seqs_user_time)):
        item_user_his_mask[idx, :len(x)] = 1.

        item_user_his[idx, :len(x)] = x
        item_user_his_time[idx, :len(t)] = t

    for idx, x in enumerate(seqs_user_mid):  # 遍历128 [128,n,m]->[n,m]
        for idy, y in enumerate(x):  # 遍历每个历史用户 [n,m]->[m]，注意这里的n是不一定长的对于每个目标物品来说购买它的用户长度可能是[1,2,3]，因为seqs_user_mid只是截断了，没有填充
            item_user_his_mid_mask[idx, idy, :len(y)] = 1.0

            item_user_his_mid[idx, idy, :len(y)] = y
            item_user_his_cat[idx, idy, :len(y)] = seqs_user_cat[idx][idy]

    uids = numpy.array([inp[0] for inp in input])
    mids = numpy.array([inp[1] for inp in input])
    cats = numpy.array([inp[2] for inp in input])

    return uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, numpy.array(target), numpy.array(lengths_x), numpy.array(lengths_s_user), noclk_mid_his, noclk_cat_his


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
        uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his = prepare_data(src, tgt, maxlen, user_maxlen)
        prob, loss, acc, aux_loss = model.calculate(sess, [uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his])
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
    best_model_path = "../dnn_best_model/" + model_type + str(seed) + "/ckpt_noshuff"
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        print("n_uid, n_mid, n_cat:", n_uid, n_mid, n_cat)
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_neg':
            model = DIEN_with_neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN':
            model = Model_DNN_Multi_Head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'TIEN':
            model = Model_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN_test':
            model = Model_DUMN_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_DUMN':
            model = Model_DMIN_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN_origin':
            model = Model_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK':
            model = DRINK_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_test':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel':
            model = myModel(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print("Invalid model_type : %s", model_type)
            return
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sys.stdout.flush()
        print('                                                                                   test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % eval(sess, test_data, model, best_model_path, maxlen, user_maxlen))
        sys.stdout.flush()
        print('test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % eval(sess, test_data, model, best_model_path, maxlen, user_maxlen), file=txt_log_test)
        txt_log_test.flush()
        start_time = time.time()
        iter = 0
        lr = args.lr
        for itr in range(3):
            loss_sum = 0.0
            accuracy_sum = 0.
            aux_loss_sum = 0.
            for src, tgt in train_data:
                uids, mids, cats, mid_his, cat_his, mid_mask, item_user_his, item_user_his_mask, item_user_his_time, item_user_his_mid, item_user_his_cat, item_user_his_mid_mask, target, current_user_his_item_length, target_item_his_user_length, noclk_mid_his, noclk_cat_his = prepare_data(src, tgt, maxlen, user_maxlen)
                loss, acc, log_loss = model.train(sess, [uids, mids, cats, mid_his, cat_his, mid_mask,
                                                         item_user_his, item_user_his_mask, item_user_his_time,
                                                         item_user_his_mid, item_user_his_cat, item_user_his_mid_mask,
                                                         target, current_user_his_item_length, target_item_his_user_length, lr, noclk_mid_his, noclk_cat_his])
                loss_sum += loss
                accuracy_sum += acc
                aux_loss_sum += log_loss
                iter += 1
                sys.stdout.flush()
                if (iter % test_iter) == 0:
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                    print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- aux_loss: %.4f' % (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter), file=txt_log_train)
                    txt_log_train.flush()
                    print('                                                                                   test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- aux_loss: %.4f' % eval(sess, test_data, model, best_model_path, maxlen, user_maxlen))
                    eval_results = eval(sess, test_data, model, best_model_path, maxlen, user_maxlen)
                    print('iter %d ---- test_auc: %.4f ---- test_F1: %.4f ---- test_accuracy: %.4f ---- Logloss: %.4f' % ((iter,) + eval_results), file=txt_log_test)
                    txt_log_test.flush()
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    aux_loss_sum = 0.0
            lr *= 0.5  # 学习率衰减，其实可以改一下方式


def test(train_file, test_file, uid_voc, mid_voc, cat_voc, batch_size, user_maxlen, maxlen, model_type, seed):
    model_path = tf.train.latest_checkpoint("dnn_best_model/" + model_type + str(seed))
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_data = DataIterator(train_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        test_data = DataIterator(test_file, uid_voc, mid_voc, cat_voc, batch_size, maxlen)
        n_uid, n_mid, n_cat = train_data.get_n()
        if model_type == 'DNN':
            model = Model_DNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'SVDPP':
            model = Model_SVDPP(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'GRU4REC':
            model = Model_GRU4REC(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'PNN':
            model = Model_PNN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'Wide':
            model = Model_WideDeep(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIN':
            model = Model_DIN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN':
            model = DIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DIEN_neg':
            model = DIEN_with_neg(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN':
            model = Model_DNN_Multi_Head(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'TIEN':
            model = Model_TIEN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN_test':
            model = Model_DUMN_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DMIN_DUMN':
            model = Model_DMIN_DUMN(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DUMN_origin':
            model = Model_DUMN_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK':
            model = DRINK_origin(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'DRINK_test':
            model = DRINK_test(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        elif model_type == 'myModel':
            model = myModel(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        else:
            print("Invalid model_type : %s", model_type)
            return
        model.restore(sess, model_path)
        print('test_auc: %.4f ----test_F1: %.4f ---- test_accuracy: %.4f ---- aux_loss: %.4f' % eval(sess, test_data, model, model_path, maxlen, user_maxlen))


dirname = "Beauty"
filename = "Beauty"
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default="DMIN")
    parser.add_argument('--train_file', default="../dataset/{}/local_train_sample_sorted_by_time".format(dirname))
    parser.add_argument('--test_file', default="../dataset/{}/local_test_sample_sorted_by_time".format(dirname))
    parser.add_argument('--uid_voc', default="../dataset/{}/uid_voc.pkl".format(dirname))
    parser.add_argument('--mid_voc', default="../dataset/{}/mid_voc.pkl".format(dirname))
    parser.add_argument('--cat_voc', default="../dataset/{}/cat_voc.pkl".format(dirname))
    parser.add_argument('--max_len', type=int, default=20)  # 用户历史行为序列的最大长度，Electronic设置为15，books设置为20
    parser.add_argument('--user_maxlen', type=int, default=50)  # 用户历史行为序列的最大长度，Electronic设置为15，books设置为20
    parser.add_argument('--train_iter', type=int, default=100)
    # ------------------------------这些训练输出是需要根据数据量调节的------------------------------
    parser.add_argument('--test_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=200)
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--l2_reg', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.1不行，太大了。学习率过大会导致rnn梯度爆炸：Infinity in summary histogram for: rnn_2/GRU_outputs2
    parser.add_argument('--lr_decay_steps', type=int, default=10000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=128)
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

    type_name = 'train'
    if type_name == 'train':
        train(model_type=args.model_type, train_file=args.train_file, test_file=args.test_file, uid_voc=args.uid_voc, mid_voc=args.mid_voc, cat_voc=args.cat_voc, batch_size=args.batch_size, maxlen=args.max_len, user_maxlen=args.user_maxlen, test_iter=args.test_iter, seed=args.seed)
    elif type_name == 'test':
        test(model_type=args.model_type, train_file=args.train_file, test_file=args.test_file, uid_voc=args.uid_voc, mid_voc=args.mid_voc, cat_voc=args.cat_voc, batch_size=args.batch_size, maxlen=args.max_len, user_maxlen=args.user_maxlen, seed=args.seed)
    else:
        print('do nothing...')
