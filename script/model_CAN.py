# coding=utf-8
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from transformer import transformer_model, gelu
import numpy as np
import tensorflow as tf

TIME_INTERVAL = 21  # gap = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.7, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])  # 共有16个取值范围
ITEM_BH_CLS_CNT = 3  # cls采用的是3个

#### CAN config #####
weight_emb_w = [[16, 8], [8, 4]]
weight_emb_b = [0, 0]
print(weight_emb_w, weight_emb_b)
orders = 3
order_indep = False  # True
WEIGHT_EMB_DIM = (sum([w[0] * w[1] for w in weight_emb_w]) + sum(weight_emb_b))  # * orders 16*8+8*4=160
INDEP_NUM = 1
if order_indep:
    INDEP_NUM *= orders
CALC_MODE = "can"


def gen_coaction(weight_mlp, his_items, mode="can", mask=None):  # [128,160],[128,20,16]
    idx = 0
    weight_orders = []
    bias_orders = []
    for i in range(orders):  # 1,2
        weight, bias = [], []
        for w, b in zip(weight_emb_w, weight_emb_b):  # weight_emb_w = [[16, 8], [8, 4]]，weight_emb_b = [0, 0]-->([16, 8], 0),([8, 4], 0)
            weight.append(tf.reshape(weight_mlp[:, idx:idx + w[0] * w[1]], [-1, w[0], w[1]]))  # [128,160]:[128,128]->[128,16,8],[128,32]->[128,8，4]
            idx += w[0] * w[1]  # 16*8=128
            if b == 0:
                bias.append(None)
            else:
                bias.append(tf.reshape(weight_mlp[:, idx:idx + b], [-1, 1, b]))
                idx += b
        weight_orders.append(weight)  # [[[128,16,8],[128,8,4]],[[128,16,8],[128,8,4]]]
        bias_orders.append(bias)  # [[None,None]]
        if not order_indep:  # 如果order_indep为false的话，那么运行完一次就跳出
            break
    print("weight_orders", weight_orders)
    if mode == "can":
        out_seq = []
        hh = []
        for i in range(orders):
            hh.append(his_items ** (i + 1))  # 对his_items进行乘方，以获得高阶特征[[128,20,16],[128,20,16]]
        # hh = [sum(hh)]
        for i, h in enumerate(hh):  # [128,20,16],[128,20,16]
            if order_indep:
                weight, bias = weight_orders[i], bias_orders[i]  # [[128,16,8],[128,8,4]].[None,None]
            else:
                weight, bias = weight_orders[0], bias_orders[0]  # [128,16,8],[128,8,4]],[None,None]
            for j, (w, b) in enumerate(zip(weight, bias)):
                h = tf.matmul(h, w)  # [128,20,16] * [128,16,8]->[128,20,8]
                if b is not None:
                    h = h + b
                if j != len(weight) - 1:
                    h = tf.nn.tanh(h)  # 第一层过一个激活函数
                out_seq.append(h)  # [[128,20,8],[128,20,4]]
        out_seq = tf.concat(out_seq, 2)  # [[128,20,8],[128,20,4]]->[128,20,12]
        if mask is not None:
            mask = tf.expand_dims(mask, axis=-1)  # [128,20,1]
            out_seq = out_seq * mask  # [128,20,12]*[128,20,1]->[128,20,12]
    out = tf.reduce_sum(out_seq, 1)  # [128,20,12]->[128,12]
    return out, None  # [128,12]


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_coaction=False):
        with tf.name_scope('Inputs'):
            self.use_negsampling = use_negsampling
            self.use_coaction = use_coaction
            self.EMBEDDING_DIM = EMBEDDING_DIM
            self.is_training = tf.placeholder(tf.bool)  # 为使用Dropout,batch_normalization等

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # 1
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')  # 2
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')  # 3
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, 20], name='mid_his_batch_ph')  # 4
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, 20], name='cat_his_batch_ph')  # 5
            self.tiv_his_items = tf.placeholder(tf.int32, [None, 20], name='tiv_his_items')  # 6

            self.item_user_his_batch_ph = tf.placeholder(tf.int32, [None, 50], name='item_user_his_batch_ph')  # 1 [128,50]
            self.item_user_his_time_ph = tf.placeholder(tf.int32, [None, 50], name='item_user_his_time_ph')  # 2 [128,50]
            self.item_user_his_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_mid_batch_ph')  # 3 [128,50,20]
            self.item_user_his_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_cat_batch_ph')  # 4 [128,50,20]
            self.item_user_his_tiv_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_tiv_batch_ph')  # 5 [128,50,20]

            self.mask = tf.placeholder(tf.float32, [None, 20], name='mask')  # [128,20]当前用户的历史行为的mask，从后面填充的，被padding的部分mask值为0
            self.item_user_his_mask = tf.placeholder(tf.float32, [None, 50], name='item_user_his_mask')  # [128,50] 目标物品的历史购买用户的mask
            self.item_user_his_items_mask = tf.placeholder(tf.float32, [None, 50, 20], name='item_user_his_mid_mask')  # [128,50,20]目标物品的历史购买用户的历史行为的mask

            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')  # 历史行为的长度
            self.seq_len_u_ph = tf.placeholder(tf.int32, [None], name='seq_len_u_ph')  # 目标物品历史用户的长度

            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

            self.his_item_his_users_list = tf.placeholder(tf.int32, [None, None, None], name='his_item_his_users_list')  # [128,20,10] 购买历史物品的历史用户
            self.his_item_user_mask = tf.placeholder(tf.float32, [None, None, None], name='his_item_user_mask')  # [128,20,10]
            self.his_item_his_users_tiv_list = tf.placeholder(tf.int32, [None, None, None], name='his_item_his_users_tiv_list')  # [128,20,10] 购买历史物品的历史用户的购买时间距当前用户购买历史物品的时间间隔
        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            # ---------------------------------------uid_embedding---------------------------------------
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            self.uid_embeddings_var_2 = tf.get_variable("uid_embedding_var_2", [n_uid, EMBEDDING_DIM])
            self.uid_emb = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
            self.item_user_his_uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.item_user_his_batch_ph)  # [128,50,18]
            self.his_item_his_users_emb = tf.nn.embedding_lookup(self.uid_embeddings_var, self.his_item_his_users_list)  # [128,20,10,18]购买历史物品的历史用户
            # ---------------------------------------mid_embedding---------------------------------------
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            self.mid_embeddings_var_2 = tf.get_variable("mid_embedding_var_2", [n_mid, EMBEDDING_DIM])
            self.can_target_mid_embeddings_var = tf.get_variable("can_target_mid_embeddings_var", [n_mid, 176])
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)  # 目标物品id的embedding [128,18]
            self.mid_batch_embedded_2 = tf.nn.embedding_lookup(self.mid_embeddings_var_2, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)  # 当前用户历史物品id的embedding [128,20,18]
            self.mid_his_batch_embedded_2 = tf.nn.embedding_lookup(self.mid_embeddings_var_2, self.mid_his_batch_ph)
            self.can_target_mid_embeddings = tf.nn.embedding_lookup(self.can_target_mid_embeddings_var, self.mid_batch_ph)
            self.item_user_his_mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.item_user_his_mid_batch_ph)  # [128,50,20,18]
            self.item_user_his_mid_batch_embedded_2 = tf.nn.embedding_lookup(self.mid_embeddings_var_2, self.item_user_his_mid_batch_ph)  # [128,50,20,18]
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)
            # ---------------------------------------cat_embedding---------------------------------------
            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            self.cat_embeddings_var_2 = tf.get_variable("cat_embedding_var_2", [n_cat, EMBEDDING_DIM])
            self.can_target_cat_embeddings_var = tf.get_variable("can_target_cat_embeddings_var", [n_mid, 176])
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)  # 目标物品类别的embedding [128,18]
            self.cat_batch_embedded_2 = tf.nn.embedding_lookup(self.cat_embeddings_var_2, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)  # 当前用户历史物品id的embedding [128,20,18]
            self.cat_his_batch_embedded_2 = tf.nn.embedding_lookup(self.cat_embeddings_var_2, self.cat_his_batch_ph)
            self.can_target_cat_embeddings = tf.nn.embedding_lookup(self.can_target_cat_embeddings_var, self.cat_batch_ph)
            self.item_user_his_cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.item_user_his_cat_batch_ph)  # [128,50,20,18]
            self.item_user_his_cat_batch_embedded_2 = tf.nn.embedding_lookup(self.cat_embeddings_var_2, self.item_user_his_cat_batch_ph)  # [128,50,20,18]
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)
            # ---------------------------------------tiv_embedding---------------------------------------
            self.time_embeddings_var_for_item = tf.get_variable("time_embeddings_var_for_item", [TIME_INTERVAL, EMBEDDING_DIM])  # [21,18]
            self.time_embeddings_var_for_user = tf.get_variable("time_embeddings_var_for_user", [TIME_INTERVAL, EMBEDDING_DIM])  # [21,18]
            self.his_items_tiv_emb = tf.nn.embedding_lookup(self.time_embeddings_var_for_item, self.tiv_his_items)  # [128,20,18]
            # 相比mymodel1，这里改动了
            self.his_item_his_users_tiv_emb = tf.nn.embedding_lookup(self.time_embeddings_var_for_user, self.his_item_his_users_tiv_list)  # [128,20,10,18]
            self.item_bh_time_embeeded = tf.nn.embedding_lookup(self.time_embeddings_var_for_user, self.item_user_his_time_ph)  # [128,50,18]
            self.item_user_his_tivs_emb = tf.nn.embedding_lookup(self.time_embeddings_var_for_item, self.item_user_his_tiv_batch_ph)  # [128,50,20,18]
            # ---------------------------------------DRINK_cls_embedding---------------------------------------
            self.item_bh_cls_embedding = tf.get_variable("item_cls_embedding", [ITEM_BH_CLS_CNT, EMBEDDING_DIM * 2])  # [3,36]
        # ---------------------------------------对上面的Embedding做一些简单的处理：concat---------------------------------------
        self.target_item_emb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.target_item_emb_2 = tf.concat([self.mid_batch_embedded_2, self.cat_batch_embedded_2], 1)
        self.his_item_emb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.his_item_emb_2 = tf.concat([self.mid_his_batch_embedded_2, self.cat_his_batch_embedded_2], 2)

        self.item_user_his_items_emb = tf.concat([self.item_user_his_mid_batch_embedded, self.item_user_his_cat_batch_embedded], -1)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
        self.item_user_his_eb_2 = tf.concat([self.item_user_his_mid_batch_embedded_2, self.item_user_his_cat_batch_embedded_2], -1)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
        self.item_his_eb_sum = tf.reduce_sum(self.his_item_emb * tf.expand_dims(self.mask, -1), 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat([self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)  # [128,20,18],[128,20,18]->[128,20,36]
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)
        ###  co-action ###
        if self.use_coaction:
            ph_dict = {
                "item": [self.mid_batch_ph, self.mid_his_batch_ph],
                "cate": [self.cat_batch_ph, self.cat_his_batch_ph]}
            self.mlp_batch_embedded = []
            self.item_mlp_embeddings_var = tf.get_variable("item_mlp_embedding_var", [n_mid, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)
            self.cate_mlp_embeddings_var = tf.get_variable("cate_mlp_embedding_var", [n_cat, INDEP_NUM * WEIGHT_EMB_DIM], trainable=True)
            self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.item_mlp_embeddings_var, ph_dict['item'][0]))  # [128,160],[128,320]
            self.mlp_batch_embedded.append(tf.nn.embedding_lookup(self.cate_mlp_embeddings_var, ph_dict['cate'][0]))  # [128,160],[128,320]

            self.input_batch_embedded = []
            self.item_input_embeddings_var = tf.get_variable("item_input_embedding_var", [n_mid, weight_emb_w[0][0] * INDEP_NUM], trainable=True)
            self.cate_input_embeddings_var = tf.get_variable("cate_input_embedding_var", [n_cat, weight_emb_w[0][0] * INDEP_NUM], trainable=True)
            self.input_batch_embedded.append(tf.nn.embedding_lookup(self.item_input_embeddings_var, ph_dict['item'][1]))  # [128,20,16],[128,20,32]
            self.input_batch_embedded.append(tf.nn.embedding_lookup(self.cate_input_embeddings_var, ph_dict['cate'][1]))  # [128,20,16],[128,20,32]
        self.cross = []
        if self.use_coaction:
            tmp_sum, tmp_seq = [], []
            if INDEP_NUM == 2:
                for i, mlp_batch in enumerate(self.mlp_batch_embedded):  # [0]是item，[1]是类别，形状都为[128,320]
                    for j, input_batch in enumerate(self.input_batch_embedded):  # [0]是item，[1]是类别，形状都为[128,20,32]
                        coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, WEIGHT_EMB_DIM * j:  WEIGHT_EMB_DIM * (j + 1)],  # [128,:160]，[128,160:]
                                                                  input_batch[:, :, weight_emb_w[0][0] * i: weight_emb_w[0][0] * (i + 1)],  # [128,20,:16]
                                                                  mode=CALC_MODE, mask=self.mask)
                        tmp_sum.append(coaction_sum)
                        tmp_seq.append(coaction_seq)
            else:
                for i, (mlp_batch, input_batch) in enumerate(zip(self.mlp_batch_embedded, self.input_batch_embedded)):  # [0]是item，[1]是类别，形状都为[128,480]
                    coaction_sum, coaction_seq = gen_coaction(mlp_batch[:, : INDEP_NUM * WEIGHT_EMB_DIM],  # [128,480]
                                                              input_batch[:, :, : weight_emb_w[0][0]],  # [128,20,16]
                                                              mode=CALC_MODE, mask=self.mask)
                    tmp_sum.append(coaction_sum)
                    tmp_seq.append(coaction_seq)

            self.coaction_sum = tf.concat(tmp_sum, axis=1)  # [[128,12],[128,12]]->[128,24]
            self.cross.append(self.coaction_sum)  # [[128,24]]

    # self.all_interests = tf.get_variable("all_trends", [args.all_candidate_trends, args.item_emb_dim + args.cat_emb_dim + args.tiv_emb_dim + args.position_emb_dim])  # (100, 44)

    def build_fcn_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')  # 这里带上BN，但是并没有传入参数training
        # dnn_1 = tf.layers.dense(bn1, 800, activation=None, name='f_1')
        # dnn_1 = prelu(dnn_1, 'prelu_1')
        # dnn0 = tf.layers.dense(dnn_1, 400, activation=None, name='f0')
        # dnn0 = prelu(dnn0, 'prelu0')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)  # [128,2],[128,2]
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def focal_loss(self, prop, gamma=2.0, alpha=1.0,sigma=1.0, boundary = False):
        if boundary:
            return -alpha * tf.pow(1 - prop, gamma) * tf.log(prop)
        else:
            return -alpha * tf.pow(tf.minimum(sigma,1 - prop), gamma) * tf.log(prop)

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None, focal_loss_bool=False):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)  # [128,19,36],[128,19,36]->[128,19,72]
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)  # [128,19,36],[128,19,36]->[128,19,72]
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]  # [128,19,2]->[128,19]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 1]  # [128,19,2]->[128,19]

        if focal_loss_bool:
            click_loss_ = self.focal_loss(click_prop_) * mask  # [128,19]
            noclick_loss_ = self.focal_loss(noclick_prop_) * mask  # [128,19]
        else:
            click_loss_ = - tf.log(click_prop_) * mask  # [128,19]
            noclick_loss_ = - tf.log(noclick_prop_) * mask  # [128,19]
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)  # [128,19]->scalar
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):  # 使用了tf.AUTO_REUSE，代表每个位置都是复用的
        dnn1 = tf.layers.dense(in_, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)  # [128,19,72]->[128,19,100]
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)  # [128,19,100]->[128,19,50]
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)  # [128,19,50]->[128,19,2]
        # dnn3_left = dnn3[:, :-1, :]
        # dnn3_right = dnn3[:, 1:, :]
        # dnn3 = tf.concat([dnn3[:, 0:1, :], dnn3_right - dnn3_left], axis=1)  # [128,18,2]-[128,18,2]
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat  # [128,19,2]

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.tiv_his_items: inps[5],
                self.mask: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_tiv_batch_ph: inps[12],
                self.item_user_his_items_mask: inps[13],
                self.target_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.seq_len_u_ph: inps[16],
                self.lr: inps[17],
                self.his_item_his_users_list: inps[18],
                self.his_item_user_mask: inps[19],
                self.his_item_his_users_tiv_list: inps[20],
                self.noclk_mid_batch_ph: inps[21],
                self.noclk_cat_batch_ph: inps[22],
                self.is_training: np.bool(True)
            })
            return loss, accuracy, aux_loss
        else:
            loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.tiv_his_items: inps[5],
                self.mask: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_tiv_batch_ph: inps[12],
                self.item_user_his_items_mask: inps[13],
                self.target_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.seq_len_u_ph: inps[16],
                self.lr: inps[17],
                self.his_item_his_users_list: inps[18],
                self.his_item_user_mask: inps[19],
                self.his_item_his_users_tiv_list: inps[20],
                self.is_training: np.bool(True)
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.tiv_his_items: inps[5],
                self.mask: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_tiv_batch_ph: inps[12],
                self.item_user_his_items_mask: inps[13],
                self.target_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.seq_len_u_ph: inps[16],
                self.his_item_his_users_list: inps[17],
                self.his_item_user_mask: inps[18],
                self.his_item_his_users_tiv_list: inps[19],
                self.noclk_mid_batch_ph: inps[20],
                self.noclk_cat_batch_ph: inps[21],
                self.is_training: np.bool(False)
            })
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.tiv_his_items: inps[5],
                self.mask: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_tiv_batch_ph: inps[12],
                self.item_user_his_items_mask: inps[13],
                self.target_ph: inps[14],
                self.seq_len_ph: inps[15],
                self.seq_len_u_ph: inps[16],
                self.his_item_his_users_list: inps[17],
                self.his_item_user_mask: inps[18],
                self.his_item_his_users_tiv_list: inps[19],
                self.is_training: np.bool(False)
            })
            return probs, loss, accuracy, 0

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False, use_coaction=False):
        super(DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_coaction=use_coaction)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_DIEN_CAN(Model):
    def __init__(self, n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True, use_coaction=True):
        super(Model_DIEN_CAN, self).__init__(n_uid, n_mid, n_cate, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling, use_coaction=use_coaction)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
        if self.use_coaction:
            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2] + self.cross, 1)
        else:
            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp)
