# coding=utf-8
import numpy as np

from Dice import dice
from rnn import dynamic_rnn
from transformer import transformer_model, gelu
from utils import *

TIME_INTERVAL = 21  # gap = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.7, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])  # 共有16个取值范围
ITEM_BH_CLS_CNT = 3  # cls采用的是3个


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        with tf.name_scope('Inputs'):
            self.context_weights_matrices1 = []
            self.context_weights_matrices2 = []
            self.use_negsampling = use_negsampling  # 是否使用辅助损失
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
            self.loss_tiv_weight_ph = tf.placeholder(tf.float32, [None, ], name='loss_tiv_weight')
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

            self.his_item_his_users_list = tf.placeholder(tf.int32, [None, None, None], name='his_item_his_users_list')  # [128,20,10] 购买历史物品的历史用户
            self.his_item_user_mask = tf.placeholder(tf.float32, [None, None, None], name='his_item_user_mask')  # [128,20,10]
            self.his_item_his_users_tiv_list = tf.placeholder(tf.int32, [None, None, None], name='his_item_his_users_tiv_list')  # [128,20,10] 购买历史物品的历史用户的购买时间距当前用户购买历史物品的时间间隔
            # self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)  # 设定 L2 正则化比例
        # Embedding layer
        # with tf.variable_scope('Embedding_layer', regularizer=self.l2_regularizer):
        with tf.variable_scope('Embedding_layer'):
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
            # 相比原来的mymodel1，这里改动了
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
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], EMBEDDING_DIM * 2])
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)  # [128,20,18],[128,20,18]->[128,20,36]
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

    def build_fcn_net(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')  # 这里带上BN，但是并没有传入参数training
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'prelu1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)  # [128,2],[128,2]
            # ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph * tf.expand_dims(self.loss_tiv_weight_ph,1))  # [128,2],[128,2]
            # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # self.loss = ctr_loss + tf.reduce_sum(reg_losses)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def focal_loss(self, prop, gamma=2.0, alpha=1.0, sigma=1.0, boundary=False):
        if boundary:
            return -alpha * tf.pow(1 - prop, gamma) * tf.log(prop)
        else:
            return -alpha * tf.pow(tf.minimum(sigma, 1 - prop), gamma) * tf.log(prop)

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None, focal_loss_bool=False):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)  # [128,19,36],[128,19,36]->[128,19,72]
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)  # [128,19,36],[128,19,36]->[128,19,72]
        click_prop_ = self.auxiliary_net(click_input_, stag = stag)[:, :, 0]  # [128,19,2]->[128,19]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag = stag)[:, :, 1]  # [128,19,2]->[128,19]

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

    def calculate_atten_loss(self, attention, hidden_size):
        # 计算平均值
        C_mean = tf.reduce_mean(attention, axis=2, keep_dims=True)
        C_reg = attention - C_mean
        C_reg = tf.matmul(C_reg, C_reg, transpose_b=True) / hidden_size
        dr = tf.linalg.diag_part(C_reg)  # 提取对角线并计算二范数
        n2 = tf.norm(dr, axis=1) ** 2
        return tf.reduce_sum(n2)  # 返回范数的平方和

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
                self.loss_tiv_weight_ph: inps[23],
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
                self.loss_tiv_weight_ph: inps[23],
                self.is_training: np.bool(True)
            })
            return loss, accuracy, 0

    def calculate(self, sess, inps):
        if self.use_negsampling:
            probs, loss, accuracy, aux_loss,context_weights_matrices1,context_weights_matrices2 = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss,self.context_weights_matrices1,self.context_weights_matrices2], feed_dict={
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
                self.loss_tiv_weight_ph: inps[22],
                self.is_training: np.bool(False)
            })
            return probs, loss, accuracy, aux_loss,context_weights_matrices1,context_weights_matrices2
        else:
            probs, loss, accuracy,context_weights_matrices1,context_weights_matrices2  = sess.run([self.y_hat, self.loss, self.accuracy,self.context_weights_matrices1,self.context_weights_matrices2], feed_dict={
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
                self.loss_tiv_weight_ph: inps[22],
                self.is_training: np.bool(False)
            })
            return probs, loss, accuracy, 0, context_weights_matrices1,context_weights_matrices2

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_LR(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_LR, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # ----------------------------------------一次项------------------------------------------
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)  # 其实对于类别向量来说就是1乘以权重（也就是选取权重）
        w_cate_var = tf.get_variable("w_cate_var", [n_cat, 1], trainable=True)
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)  # 选取偏置
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))  # [128,1]目标物品的权重
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cat_batch_ph))  # [128,1]目标类别的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))  # [128,5,1]->[128,1]历史物品的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cat_his_batch_ph), axis=1))  # [128,5,1]->[128,1]历史类别的权重
        logit = tf.reduce_sum(tf.concat(wx, axis=1), axis=1, keepdims=True) + b  # [[128,1],[128,1],[128,1],[128,1]]->[128,4]->[128,1]
        positive_prob = tf.nn.sigmoid(logit)  # [128,1] 正类概率
        negative_prob = 1 - positive_prob  # [128,1] 负类概率
        self.y_hat = tf.concat([positive_prob, negative_prob], axis=-1) + 0.00000001  # [128,2]

        with tf.name_scope('Metrics'):
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)  # [128,2],[128,2]
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()

def FMLayer(fea_list):
    fea_list = tf.stack(fea_list, axis=1)  # [[128,36],[128,36],[128,36],[128,36]]->[128,4,36]
    square_of_sum = tf.reduce_sum(fea_list, axis=1, keep_dims=True) ** 2  # [128,1,36]
    sum_of_square = tf.reduce_sum(fea_list ** 2, axis=1, keep_dims=True)  # [128,1,36]
    fm_term = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=-1)  # [128,1,36]->[128,1]
    return fm_term

class Model_FM(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_FM, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # ----------------------------------------一次项------------------------------------------
        w_item_var = tf.get_variable("w_item_var", [n_mid, 1], trainable=True)  # 其实对于类别向量来说就是1乘以权重（也就是选取权重）
        w_cate_var = tf.get_variable("w_cate_var", [n_cat, 1], trainable=True)
        b = tf.get_variable("b_var", [1], initializer=tf.zeros_initializer(), trainable=True)  # 选取偏置
        wx = []
        wx.append(tf.nn.embedding_lookup(w_item_var, self.mid_batch_ph))  # [128,1]目标物品的权重
        wx.append(tf.nn.embedding_lookup(w_cate_var, self.cat_batch_ph))  # [128,1]目标类别的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_item_var, self.mid_his_batch_ph), axis=1))  # [128,20,1]->[128,1]历史物品的权重
        wx.append(tf.reduce_sum(tf.nn.embedding_lookup(w_cate_var, self.cat_his_batch_ph), axis=1))  # [128,20,1]->[128,1]历史类别的权重
        linear = tf.reduce_sum(tf.concat(wx, axis=1), axis=1, keepdims=True) + b  # [[128,1],[128,1],[128,1],[128,1]]->[128,4]->[128,1]
        # ----------------------------------------二次项------------------------------------------
        fea_list = [self.mid_batch_embedded, self.cat_batch_embedded, tf.reduce_sum(self.mid_his_batch_embedded, axis=1), tf.reduce_sum(self.cat_his_batch_embedded, axis=1)]  # [[128,36],[128,36],[128,36],[128,36]]
        logit = linear + FMLayer(fea_list)  # [128,1]+[128,1]->[128,1]

        positive_prob = tf.nn.sigmoid(logit)  # [128,1] 正类概率
        negative_prob = 1 - positive_prob  # [128,1] 负类概率
        self.y_hat = tf.concat([positive_prob, negative_prob], axis=-1) + 0.00000001  # [128,2]

        with tf.name_scope('Metrics'):
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)  # [128,2],[128,2]
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_WideDeep(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_WideDeep, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum], 1)
        # Fully connected layer
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, 'p1')
        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, 'p2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        d_layer_wide = tf.concat([tf.concat([self.target_item_emb, self.item_his_eb_sum], axis=-1), self.target_item_emb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)


class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,ATTENTION_SIZE)
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp)


class Model_DIN_FRNet(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN_FRNet, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        self.his_items_tiv_emb = tf.layers.dense(self.his_items_tiv_emb, EMBEDDING_DIM * 2, name='tiv_his_items_emb')
        # with tf.name_scope("frnet"):
        #     self.his_item_emb = frnet(inputs=self.his_item_emb + self.his_items_tiv_emb, seq_length=20, embed_dim=EMBEDDING_DIM * 2, padding_mask=self.mask, weight_type="vector")  # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.target_item_emb, self.his_item_emb + self.his_items_tiv_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp)


class Model_DIN_Dynamic_MLP(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN_Dynamic_MLP, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, att_fea], -1)

        rnn_outputs, rnn_output = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_items_tiv_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
        rnn_outputs = rnn_outputs * tf.expand_dims(self.mask, axis=-1)  # [128,20,36],[128,20,1]->[128,20,36]

        rnn_outputs = tf.reshape(rnn_outputs, [tf.shape(rnn_outputs)[0], 720])  # [128,20*36]
        mlp_weights = tf.layers.dense(rnn_outputs, units=512)  # [128,20*36]->[128,]
        mlp_weights = tf.keras.layers.PReLU()(mlp_weights)  # PReLU activation
        mlp_weights = tf.layers.dense(mlp_weights, units=200)  # [128,20*36]->[128,200]
        mlp_weights = tf.layers.dropout(mlp_weights, rate=0.2, training=self.is_training)

        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 256, activation=None, name='f1')
        dnn1 = dnn1 * mlp_weights
        use_dice = True
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 128, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsampling:
                self.loss += self.aux_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))

        self.merged = tf.summary.merge_all()


class DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores = tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp)


class CAN_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(CAN_DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ------------------------------------------------------------CAN-------------------------------------------------------------------------
        mid_weight1, mid_weight2 = tf.split(self.can_target_mid_embeddings, [144, 32], 1)  # [128,128],[128,32]
        cat_weight1, cat_weight2 = tf.split(self.can_target_cat_embeddings, [144, 32], 1)  # [128,128],[128,32]

        mid_weight1 = tf.reshape(mid_weight1, [-1, EMBEDDING_DIM, 8])  # [128,16,8]
        mid_weight2 = tf.reshape(mid_weight2, [-1, 8, 4])  # [128,8,4]

        cat_weight1 = tf.reshape(cat_weight1, [-1, EMBEDDING_DIM, 8])  # [128,16,8]
        cat_weight2 = tf.reshape(cat_weight2, [-1, 8, 4])  # [128,8,4]

        can_mid_layer1 = tf.matmul(self.mid_his_batch_embedded_2, mid_weight1)  # [128,20,16],[128,16,8]->[128,20,8]
        can_mid_layer1 = tf.tanh(can_mid_layer1)  # [128,20,8]
        can_mid_layer1 = tf.matmul(can_mid_layer1, mid_weight2)  # [128,20,8],[128,8,4]->[128,20,4]

        can_cat_layer1 = tf.matmul(self.cat_his_batch_embedded_2, cat_weight1)  # [128,20,16],[128,16,8]->[128,20,8]
        can_cat_layer1 = tf.tanh(can_cat_layer1)  # [128,20,8]
        can_cat_layer1 = tf.matmul(can_cat_layer1, cat_weight2)  # [128,20,8],[128,8,4]->[128,20,4]

        can_action = tf.concat([can_mid_layer1, can_cat_layer1], -1)  # [128,20,8]
        can_action_out = tf.reduce_sum(can_action * tf.expand_dims(self.mask, -1), axis=1)  # [128,20,8]*[128,20,1]->[128,20,8]->[128,8]

        # ------------------------------------------------------------DIEN------------------------------------------------------------------------
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores = tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2, can_action_out], 1)
        self.build_fcn_net(inp)


class DIEN_with_neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIEN_with_neg, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores = tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp)

class DIEN_TIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIEN_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            _, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 将user_embedding抓换成2*Embedding，near k behavior sum-pooling, the best T=5
        self.item_user_his_uid_batch_embedded = tf.layers.dense(self.item_user_his_uid_batch_embedded, INC_DIM, name='item_user_his_uid_batch_embedded_2dim')
        self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=5)  # [128,50,18],[128],[128,50,1]
        self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_k * self.item_bh_mask_k, 1)  # [128,50,18],[128,50,1]-># [128,18]

        # near item_his_users_maxlen behavior
        self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18]

        # 2.sequential modeling for user/item behaviors
        with tf.name_scope('rnn'):
            gru_outputs_i, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_ib")  # 建模物品行为序列
        # 3.将时间的embedding变成一样的维度
        with tf.name_scope('time-signal'):
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
        # 4. attention layer
        with tf.name_scope('att'):
            # 5. prevent noisy
            self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_t * self.item_bh_mask_t, 1)  # 这里是先算求和，下面再算平均，要不然分母会偏大

            item_bh_seq_len = (tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32), [-1, 1]) + 1)  # [128,36]+[128,36]
            dec = tf.layers.dense(self.uid_emb, INC_DIM, name='uid_batch_embedded_2dim') + self.item_bh_embedded_sum_t / (tf.where(tf.equal(item_bh_seq_len, tf.zeros_like(item_bh_seq_len)), tf.ones_like(item_bh_seq_len), item_bh_seq_len))

            gru_outputs_ib_with_t = gru_outputs_i + self.item_bh_time_emb_padding  # [128,50,36]
            i_att, _ = attention_net_v1(enc=gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

        # 5.time-aware representation layer
        with tf.name_scope('time_gru'):
            _, it_state = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_emb_padding, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_it")
            i_att_ta = i_att + it_state

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2,
                         self.item_bh_embedded_sum, i_att, i_att_ta], 1)  # 这里是TIEN的贡献
        # Fully connected layer
        self.build_fcn_net(inp)


class DIEN_with_neg_high(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIEN_with_neg_high, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")

        aux_loss_1 = self.auxiliary_loss(rnn_outputs2[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp)


class Model_GRU4REC(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_GRU4REC, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        with tf.name_scope('rnn1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb,sequence_length=self.seq_len_ph, dtype=tf.float32,scope="gru1")
            rnn_outputs1, final_state_1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            u_att, _ = attention_net_v1(enc=rnn_outputs1, sl=self.seq_len_ph, dec=self.target_item_emb, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ub', value=rnn_outputs1)


            item_user_len = tf.reduce_sum(self.item_user_his_mask, axis=-1)
            item_user_rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded,sequence_length=item_user_len, dtype=tf.float32,scope="gru3")
            rnn_outputs2, final_state_2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=item_user_rnn_outputs, sequence_length=item_user_len, dtype=tf.float32, scope="gru4")
            u_att, _ = attention_net_v1(enc=rnn_outputs2, sl=item_user_len, dec=self.uid_emb, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ib', value=rnn_outputs2)

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state_1, final_state_2], 1)
        self.build_fcn_net(inp)


class Model_SVDPP(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_SVDPP, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        self.uid_b = tf.get_variable("uid_b", [n_uid, 1])
        self.user_b = tf.nn.embedding_lookup(self.uid_b, self.uid_batch_ph)
        self.mid_b = tf.get_variable("mid_b", [n_mid, 1])
        self.item_b = tf.nn.embedding_lookup(self.mid_b, self.mid_batch_ph)
        # print(self.item_b)
        self.mu = tf.get_variable('mu', [], initializer=tf.truncated_normal_initializer)
        self.user_w = tf.get_variable('user_w', [EMBEDDING_DIM * 3, EMBEDDING_DIM * 2], initializer=tf.truncated_normal_initializer)
        neighbors_rep_seq = tf.concat([self.item_user_his_uid_batch_embedded, tf.reduce_sum(self.item_user_his_items_emb, axis=2)], axis=-1)
        user_rep = tf.concat([self.uid_emb, self.item_his_eb_sum], axis=-1)
        user_rep = tf.matmul(user_rep, self.user_w)
        print(user_rep)
        neighbors_norm = tf.sqrt(tf.expand_dims(tf.norm(neighbors_rep_seq, 1, (1, 2)), 1))
        neighbors_norm = tf.where(neighbors_norm > 0, neighbors_norm, tf.ones_like(neighbors_norm))
        neighbor_emb = tf.reduce_sum(neighbors_rep_seq, 1) / neighbors_norm
        neighbor_emb = tf.matmul(neighbor_emb, self.user_w)
        print(neighbor_emb)
        score = tf.reduce_sum(self.target_item_emb * (user_rep + neighbor_emb), 1) + tf.reshape(self.user_b, [-1]) + tf.reshape(self.item_b, [-1]) + self.mu
        pred = tf.reshape(tf.nn.sigmoid(score), [-1, 1])
        self.y_hat = tf.concat([pred, 1 - pred], -1) + 0.00000001
        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()


class Model_BST(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_BST, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        his_item_with_tiv_emb = tf.concat([self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
        his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2)
        input_embed = tf.concat([ his_item_with_tiv_emb,tf.expand_dims(self.target_item_emb, 1)], axis=1)  # [128,36],[128,20,36]->#[128,1,36],[128,20,36]->[128,21,36]
        ones = tf.ones([tf.shape(self.mask)[0], 1], dtype=self.mask.dtype)
        # 将 ones 拼接到 self.mask 的最后一列
        self.mask_expand = tf.concat([self.mask, ones], axis=1)
        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(input_embed, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask_expand, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        multihead_attention_outputs = multihead_attention_outputs * tf.expand_dims(self.mask_expand, -1)  # [128,21]->[128,21,1]
        input = tf.reshape(multihead_attention_outputs, [tf.shape(multihead_attention_outputs)[0], 21 * 36])
        inp = tf.concat([input, self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)


class Model_DNN_Multi_Head(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])  # [20,2]
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]

        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_for_items"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)

class Model_DNN_Multi_Head_contrasive(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head_contrasive, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,36]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,36]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # [128,20,36]

        with tf.name_scope("multi_head_attention_0"):
            his_item_emb_aux1 = self_multi_head_attn
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
            # 下面两行是point-wise feed_forward
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            multihead_attention_outputs = tf.reduce_mean(multihead_attention_outputs, axis=1)  # [128,20,36]->[128,36]
        multihead_attention_outputs_aux1 = multihead_attention_outputs[0::2, :]  # [128,36]->[64,36]

        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
            # 下面两行是point-wise feed_forward
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs

        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_2"):
            weight_matrix_size = 324
            context_input = tf.concat([tf.reshape(self.his_item_emb, [tf.shape(self.his_item_emb)[0], 20 * 36])], axis=-1)  # [128,56].[128,20],[128,20*56]->[128,1120]
            context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
            # multihead_attention_outputss = context_aware_multi_head_self_attention_v2(self.his_item_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0.2, is_training=self.is_training, name="multihead_attention_outputss")
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0.2, is_training=self.is_training)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DNN_Multi_Head_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])  # [20,2]
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]

        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_for_items"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class Model_DNN_Multi_Head_SENET(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head_SENET, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,36]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,36]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # [128,20,36]

        # his_item_emb_pooling = tf.reduce_mean(self.his_item_emb * tf.expand_dims(self.mask, -1), axis=-1)  # [128,20,36]->[128,20]
        # senet_middle = tf.layers.dense(his_item_emb_pooling, 10, activation=tf.nn.sigmoid)  # [128,20]->[128,10]
        # senet_output = tf.layers.dense(senet_middle, 20)  # [128,20]
        #
        # self.his_item_emb = tf.expand_dims(senet_output, -1) * self.his_item_emb
        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
            # 下面两行是point-wise feed_forward
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_2"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                # # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_TIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 将user_embedding转换成2*Embedding
        self.item_user_his_uid_batch_embedded = tf.layers.dense(self.item_user_his_uid_batch_embedded, INC_DIM, name='item_user_his_uid_batch_embedded_2dim')
        self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=5)  # [128,50,18],[128],[128,50,1]
        self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_k * self.item_bh_mask_k, 1)  # [128,50,18],[128,50,1]-># [128,18]

        # near item_his_users_maxlen behavior
        self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18]

        # 2.sequential modeling for user/item behaviors
        with tf.name_scope('rnn'):
            gru_outputs_u, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru_ub")  # 类似DIEN建模用户行为序列
            gru_outputs_i, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_ib")  # 建模物品行为序列
        # 3.将时间的embedding变成一样的维度
        with tf.name_scope('time-signal'):
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
        # 4. attention layer
        with (tf.name_scope('att')):
            # 对当前用户行为建模的attention，类似DIN
            u_att, _ = attention_net_v1(enc=gru_outputs_u, sl=self.seq_len_ph, dec=self.target_item_emb, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
            # 5. prevent noisy
            self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_t * self.item_bh_mask_t, 1)  # 这里是先算求和，下面再算平均，要不然分母会偏大

            item_bh_seq_len = (tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32), [-1, 1]) + 1)  # [128,36]+[128,36]
            dec = tf.layers.dense(self.uid_emb, INC_DIM, name='uid_batch_embedded_2dim') + self.item_bh_embedded_sum_t / (tf.where(tf.equal(item_bh_seq_len, tf.zeros_like(item_bh_seq_len)), tf.ones_like(item_bh_seq_len), item_bh_seq_len))

            gru_outputs_ib_with_t = gru_outputs_i + self.item_bh_time_emb_padding  # [128,50,36]
            i_att, _ = attention_net_v1(enc=gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

        # 5.time-aware representation layer
        with tf.name_scope('time_gru'):
            _, it_state = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_emb_padding, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_it")
            i_att_ta = i_att + it_state

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, u_att,
                         self.item_bh_embedded_sum, i_att, i_att_ta], 1)  # 这里是TIEN的贡献
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DUMN_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DUMN_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        with tf.name_scope('DUMN'):
            # --------------------------------------获得当前用户的表示-----------------------------------------
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)  # [128,36],[128,20,36]->[128,1,36]
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来[128,18],[128,36]->[128,54]
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            item_user_his_attention_output = din_attention(query=tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_items_emb)[1] * tf.shape(self.item_user_his_items_emb)[2]]),  # [128,36]->[128,50*20*36]
                                                           facts=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           mask=tf.reshape(self.item_user_his_items_mask, [-1, tf.shape(self.item_user_his_items_mask)[2]]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状：[128*50,1,36]
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_items_emb)[1], 36])  # [128*50,20,36]->[128*50,36]->[128,50,36]
            item_user_bhvs_feat = tf.concat([self.item_user_his_uid_batch_embedded, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来 [128,50,18],[128,50,36]->[128,50,54]
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask  # [128,54],[128,50,54]->[128,50]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,50]->[128,1]
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54],[128,50]->[128,50,54],[128,50,1]->[128,50,54]->[128,54]
        inp = tf.concat([self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, user_feat, sim_att, sim_score_sum], -1)
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DMIN_DUMN_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DMIN_DUMN_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])  # [20,2]
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        with tf.name_scope('DUMN'):
            INC_DIM = 36
            # --------------------------------------获得当前用户的表示-----------------------------------------
            # attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)  # [128,36],[128,20,36]->[128,1,36]
            # att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            # user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来[128,18],[128,36]->[128,54]

            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=self.his_item_emb,  # [128,5,36]
                                                                 padding_mask=self.mask, causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * tf.expand_dims(self.mask,axis=-1), axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            current_user_interest_emb = tf.concat([self.uid_emb, current_user_interest_emb], axis=-1)
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            # item_user_his_attention_output = din_attention(query=tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_items_emb)[1] * tf.shape(self.item_user_his_items_emb)[2]]),  # [128,36]->[128,50*20*36]
            #                                                facts=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
            #                                                mask=tf.reshape(self.item_user_his_items_mask, [-1, tf.shape(self.item_user_his_items_mask)[2]]),  # [128,50,20]->[128*50,20]
            #                                                need_tile=False)  # 返回的形状：[128*50,1,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
                                                                 padding_mask=tf.reshape(self.item_user_his_items_mask, [-1, tf.shape(self.item_user_his_items_mask)[2]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            item_user_his_attention_output = mul_att_current_user_output + item_user_his_attention_output  # [128,5,36]

            item_user_his_att = tf.reduce_mean(item_user_his_attention_output * tf.reshape(self.item_user_his_items_mask, [-1, tf.shape(self.item_user_his_items_mask)[2],1]), axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_att = tf.reshape(item_user_his_att, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            item_user_bhvs_feat = tf.concat([self.item_user_his_uid_batch_embedded, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来 [128,50,18],[128,50,36]->[128,50,54]
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            # sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask  # [128,54],[128,50,54]->[128,50]
            din_att = din_attention(query=current_user_interest_emb,facts= item_user_bhvs_feat, mask=self.item_user_his_mask,stag = "din_att")  # [128,54],[128,50,54]->[128,50]
            din_att = tf.reduce_sum(din_att, 1)  # [128,1,36]->[128,36]
            # sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,50]->[128,1]
            # sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54],[128,50]->[128,50,54],[128,50,1]->[128,50,54]->[128,54]
        inp = tf.concat([inp, din_att], -1)
        self.build_fcn_net(inp)

class Model_DMIN_DUMN_origin_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DMIN_DUMN_origin_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])  # [20,2]
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        with tf.name_scope('DUMN'):
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(self.uid_emb, self.item_user_his_uid_batch_embedded, need_tile=True) * self.item_user_his_mask  # [128,54],[128,50,54]->[128,50]
            # din_att = din_attention(query=user_feat,facts= item_user_bhvs_feat, need_tile=True,mask=self.item_user_his_mask,stag = "din_att")  # [128,54],[128,50,54]->[128,50]
            # din_att = tf.reduce_sum(din_att, 1)  # [128,1,36]->[128,36]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,50]->[128,1]
            sim_att = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54],[128,50]->[128,50,54],[128,50,1]->[128,50,54]->[128,54]
        inp = tf.concat([inp,sim_score_sum, sim_att], -1)
        self.build_fcn_net(inp)

class Model_DMIN_DUMN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DMIN_DUMN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])  # [20,2]
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        with tf.name_scope('DUMN'):
            item_his_users_maxlen =30
            INC_DIM = 36
            item_his_users_item_truncated_len = 5
            # --------------------------------------获得当前用户的表示-----------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(self.his_item_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)
            attention_output = din_attention(self.target_item_emb, current_user_k_bh_emb, tf.squeeze(current_user_bh_mask,axis=-1))  # [128,36],[128,20,36]->[128,1,36]
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来[128,18],[128,36]->[128,54]
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
            self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

             # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            self.item_user_his_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]

            item_user_his_attention_output = din_attention(query=tf.tile(self.target_item_emb, [1, item_his_users_maxlen * item_his_users_item_truncated_len]),  # [128,36]->[128,50*20*36]
                                                           facts=tf.reshape(self.item_user_his_items_emb, [-1, item_his_users_item_truncated_len, 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, item_his_users_item_truncated_len]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状：[128*50,1,36]
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, item_his_users_maxlen, 36])  # [128*50,20,36]->[128*50,36]->[128,50,36]
            self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            item_user_bhvs_feat = tf.concat([self.item_bh_k_user_emb, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来 [128,50,18],[128,50,36]->[128,50,54]
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * tf.squeeze(self.item_bh_mask_t,axis=-1)  # [128,54],[128,50,54]->[128,50]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,50]->[128,1]
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54], [128,50,1]->[128,54]
        inp = tf.concat([inp, user_feat, sim_att, sim_score_sum], -1)
        self.build_fcn_net(inp)


class Model_DUMN_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DUMN_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # 128,20,2

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有norm（加norm效果不好）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有norm（加norm效果不好）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)
        # -----------------------------------------------------item module -----------------------------------------------------
        with tf.name_scope('DUMN'):
            item_his_users_maxlen = 50
            INC_DIM = EMBEDDING_DIM

            self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm1', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # ----------------------------------------获得当前用户的表示----------------------------------------
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)  # [128,1,36]
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # [[128,18],[128,36]]->[128,18] 这里将当前user_id和user兴趣concat起来
            # ----------------------------------------获得目标物品历史用户的表示----------------------------------------
            item_user_his_attention_output = din_attention(tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_items_emb)[1] * tf.shape(self.item_user_his_items_emb)[2]]),  # [128,36]->[128,50*20*36]
                                                           tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           tf.reshape(self.item_user_his_items_mask, [-1, tf.shape(self.item_user_his_items_mask)[2]]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状是[128*50,1,36]

            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_items_emb)[1], 36])  # [128*50,1,36]->[128*50,36]->[128,50,36]
            item_user_his_att = transformer_model(item_user_his_att, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm2', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            item_user_bhvs_feat = tf.concat([self.item_bh_drink_trm_output, item_user_his_att], axis=-1)  # [[128,50,18],[128,50,36]]->[128,50,54] 这里将目标物品的历史user_id和user兴趣concat起来
            # ----------------------------------------计算当前用户和目标物品历史用户的相似性----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask  # [128,50]*[128,50]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,1]
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54]*[128,50,1]->[128,54]

        inp = tf.concat([inp, user_feat, sim_score_sum, sim_att], -1)
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_Context_DUMN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_Context_DUMN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        with tf.name_scope('DUMN'):
            item_his_users_maxlen =30
            INC_DIM = 36
            item_his_users_item_truncated_len = 5
            # --------------------------------------获得当前用户的表示-----------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(self.his_item_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)
            attention_output = din_attention(self.target_item_emb, current_user_k_bh_emb, tf.squeeze(current_user_bh_mask,axis=-1))  # [128,36],[128,20,36]->[128,1,36]
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来[128,18],[128,36]->[128,54]
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
            self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

             # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            self.item_user_his_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]

            item_user_his_attention_output = din_attention(query=tf.tile(self.target_item_emb, [1, item_his_users_maxlen * item_his_users_item_truncated_len]),  # [128,36]->[128,50*20*36]
                                                           facts=tf.reshape(self.item_user_his_items_emb, [-1, item_his_users_item_truncated_len, 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, item_his_users_item_truncated_len]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状：[128*50,20,36]
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, item_his_users_maxlen, 36])  # [128*50,20,36]->[128*50,36]->[128,50,36]
            self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            item_user_bhvs_feat = tf.concat([self.item_bh_k_user_emb, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来 [128,50,18],[128,50,36]->[128,50,54]
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * tf.squeeze(self.item_bh_mask_t,axis=-1)  # [128,54],[128,50,54]->[128,50]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,50]->[128,1]
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54], [128,50,1]->[128,54]
        inp = tf.concat([inp, user_feat, sim_att, sim_score_sum], -1)
        self.build_fcn_net(inp)


class DRINK_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------

        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # 拼接上多个cls的embedding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
            # 这里对拼接的embedding过Transformer的encoder部分
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), tf.float32), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :], sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=False, reuse=False, scope='item_user_his_eb_att')
            # -----------------------------------------------------处理cls的表示 -----------------------------------------------------
            item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]  # [128,3,36]这里只包含cls的embedding
            user_embs_for_cls = tf.tile(tf.expand_dims(self.user_embedded, 1), [1, ITEM_BH_CLS_CNT, 1])  # [128,36]->[128,1,36]->[128,3,36]
            # dot
            item_bh_cls_dot = item_bh_cls_embs * user_embs_for_cls  # [128,3,36],[128,3,36]->[128,3,36]
            item_bh_cls_dot = tf.reshape(item_bh_cls_dot, [-1, INC_DIM * ITEM_BH_CLS_CNT])  # [128,3,36]->[128,108]
            # matmul
            item_bh_cls_mat = tf.matmul(item_bh_cls_embs, user_embs_for_cls, transpose_b=True)  # [128,3,36],[128,3,36]->[128,3,36],[128,36,3]->[128,3,3]
            item_bh_cls_mat = tf.reshape(item_bh_cls_mat[:, 0, :], [-1, ITEM_BH_CLS_CNT])  # [128,1,3]->[128,3]这里取0的原因是因为冗余了，[0,1,2]维度的值一样

        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, item_bh_cls_dot, item_bh_cls_mat, decoupling_part], 1)
        self.build_fcn_net(inp)


class DRINK_FRnet(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_FRnet, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = frnet(inputs=self.his_item_emb, seq_length=20, embed_dim=EMBEDDING_DIM * 2, padding_mask=self.mask, weight_type="bit")  # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------

        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.concat([self.item_bh_k_user_emb, self.item_bh_time_emb_padding], axis=-1)
            # item_bh_drink_trm_input = tf.layers.dense(item_bh_drink_trm_input, INC_DIM, name='item_bh_drink_trm_input')  # 如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            # 拼接上多个cls的embedding
            # 这里对拼接的embedding过Transformer的encoder部分
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]
            self.item_bh_drink_trm_output = frnet(inputs=item_bh_drink_trm_input, seq_length=50, embed_dim=EMBEDDING_DIM * 2, padding_mask=att_mask_input, weight_type="bit")  # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')
        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, decoupling_part], 1)
        self.build_fcn_net(inp)

class DRINK_no_cls(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_no_cls, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------

        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')

        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, decoupling_part], 1)
        self.build_fcn_net(inp)


class DRINK_time(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_time, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------

        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            self.item_bh_drink_time_trm_output = transformer_model(self.item_bh_time_emb_padding, hidden_size=INC_DIM, attention_mask=self.item_bh_mask_t, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_time_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            self.item_bh_drink_time_trm_output = tf.pad(self.item_bh_drink_time_trm_output, [[0, 0], [3, 0], [0, 0]], "CONSTANT")

            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')
            # item_bh_drink_trm_input = tf.concat([self.item_bh_k_user_emb, self.item_bh_time_emb_padding], axis=-1)
            # item_bh_drink_trm_input = tf.layers.dense(item_bh_drink_trm_input, INC_DIM, name='item_bh_drink_trm_input')  # 如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            # 拼接上多个cls的embedding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
            # 这里对拼接的embedding过Transformer的encoder部分
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), tf.float32), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            self.item_bh_drink_trm_output = self.item_bh_drink_trm_output + self.item_bh_drink_time_trm_output
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :], sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')
            # -----------------------------------------------------处理cls的表示 -----------------------------------------------------
            item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]  # [128,3,36]这里只包含cls的embedding
            user_embs_for_cls = tf.tile(tf.expand_dims(self.user_embedded, 1), [1, ITEM_BH_CLS_CNT, 1])  # [128,36]->[128,1,36]->[128,3,36]
            # dot
            item_bh_cls_dot = item_bh_cls_embs * user_embs_for_cls  # [128,3,36],[128,3,36]->[128,3,36]
            item_bh_cls_dot = tf.reshape(item_bh_cls_dot, [-1, INC_DIM * ITEM_BH_CLS_CNT])  # [128,3,36]->[128,108]
            # matmul
            item_bh_cls_mat = tf.matmul(item_bh_cls_embs, user_embs_for_cls, transpose_b=True)  # [128,3,36],[128,3,36]->[128,3,36],[128,36,3]->[128,3,3]
            item_bh_cls_mat = tf.reshape(item_bh_cls_mat[:, 0, :], [-1, ITEM_BH_CLS_CNT])  # [128,1,3]->[128,3]这里取0的原因是因为冗余了，[0,1,2]维度的值一样

        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, item_bh_cls_dot, item_bh_cls_mat, decoupling_part], 1)
        self.build_fcn_net(inp)


class DRINK_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,36]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,36]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # [128,20,36]
            self.context_embedding = tf.concat([self.position_his_eb, self.his_items_tiv_emb], -1)  # [128,20,18],[128,20,18]->[128,20,20]

            self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
            self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
                                                                              padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
                                                                              num_units=INC_DIM, num_heads=4, dropout_rate=0.5, is_training=self.is_training, name="his_item_user_attention_output")

            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
            # ADD但是没有layer_normal
            # self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]
            self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]

            self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]

            self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM)

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                self.target_item_user_emb = tf.concat([self.uid_emb, self.target_item_emb], -1)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_user_emb, multihead_attention_outputs_v2, self.context_embedding, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.concat([self.item_bh_k_user_emb, self.item_bh_time_emb_padding], axis=-1)
            # item_bh_drink_trm_input = tf.layers.dense(item_bh_drink_trm_input, INC_DIM, name='item_bh_drink_trm_input')  # 如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            # 拼接上多个cls的embedding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
            # 这里对拼接的embedding过Transformer的encoder部分
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), tf.float32), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :], sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')
            # -----------------------------------------------------处理cls的表示 -----------------------------------------------------
            item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]  # [128,3,36]这里只包含cls的embedding
            user_embs_for_cls = tf.tile(tf.expand_dims(self.user_embedded, 1), [1, ITEM_BH_CLS_CNT, 1])  # [128,36]->[128,1,36]->[128,3,36]
            # dot
            item_bh_cls_dot = item_bh_cls_embs * user_embs_for_cls  # [128,3,36],[128,3,36]->[128,3,36]
            item_bh_cls_dot = tf.reshape(item_bh_cls_dot, [-1, INC_DIM * ITEM_BH_CLS_CNT])  # [128,3,36]->[128,108]
            # matmul
            item_bh_cls_mat = tf.matmul(item_bh_cls_embs, user_embs_for_cls, transpose_b=True)  # [128,3,36],[128,3,36]->[128,3,36],[128,36,3]->[128,3,3]
            item_bh_cls_mat = tf.reshape(item_bh_cls_mat[:, 0, :], [-1, ITEM_BH_CLS_CNT])  # [128,1,3]->[128,3]这里取0的原因是因为冗余了，[0,1,2]维度的值一样

        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, item_bh_cls_dot, item_bh_cls_mat, decoupling_part], 1)
        self.build_fcn_net(inp)


class myModel0(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs")
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- item module -----------------------------------------------------

        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.concat([self.item_bh_k_user_emb, self.item_bh_time_emb_padding], axis=-1)
            # item_bh_drink_trm_input = tf.layers.dense(item_bh_drink_trm_input, INC_DIM, name='item_bh_drink_trm_input')  # 如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            # self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            multihead_attention_outputs_origin = self_multi_head_attn(item_bh_drink_trm_input, num_units=INC_DIM, num_heads=4, padding_mask=att_mask_input, causality_mask_bool=False, dropout_rate=0.2, is_training=self.is_training, name="multihead_attention_outputs_origin")
            multihead_attention_outputs_act = tf.layers.dense(multihead_attention_outputs_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs_act = tf.layers.dense(multihead_attention_outputs_act, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
            self.item_bh_drink_trm_output = multihead_attention_outputs_origin + multihead_attention_outputs_act
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=multihead_attention_outputs_origin, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum], 1)
        self.build_fcn_net(inp)


class myModel1(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # ----------------------------------------------------- DMIN -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = EMBEDDING_DIM * 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50]-> [128,50,18],[128,50,1]-> [128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]-> [128,36]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18]

        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]->[128,50,1]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # ----------------------------------------------------- 处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')  # 将当前用户的维度变成INC_DIM
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')

        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            #  得到目标物品的每个历史购买用户的表征
            # self.item_user_his_items_emb[128,50,20,36]
            item_his_users_item_truncated_len = 10  # 目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            self.item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            self.item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(self.item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,50]->[128*50]得到[128*50,k,36],[128*50],[128*50,k,1]
            print("self.item_user_his_k_items_emb", self.item_user_his_k_items_emb.get_shape())
            item_user_his_attention_output = self_multi_head_attn(inputs=self.item_user_his_k_items_emb,  # [128*50,k,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*50,k,1]->[128*50,k]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            print("self.item_user_his_attention_output", item_user_his_attention_output.get_shape())
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)
            # ADD但是没有layer_normal
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*50,k,36]
            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*50,k,36],[128*50,k,1]->[128*50,k,36]->[128*50,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM]) + self.item_bh_time_emb_padding  # [128*50,36]->[128,50,36]
            # --------------------------------------------------------- 得到当前用户的表征 -----------------------------------------------------------
            current_user_interest_emb = self_multi_head_attn(inputs=self.his_item_emb, name="current_user_interest_emb", padding_mask=self.mask, num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training)  # [128,20,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,20,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * tf.expand_dims(self.mask, -1), axis=1)  # [128,20,36]*[128,20,1]->[128,20,36]->[128,36]

            interest_emb, _ = attention_net_v1(enc=item_user_his_items_avg, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,4,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                               dec=current_user_interest_emb,  # dec=decoder=query[128,36]
                                               num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                               is_training=self.is_training, reuse=False, scope='interest_emb')

        inp = tf.concat([inp, interest_emb, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, item_user_his_eb_att_sum], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # ----------------------------------------------------- DMIN -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,36]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,36]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # [128,20,36]
            self.context_embedding = tf.concat([self.position_his_eb, self.his_items_tiv_emb], -1)  # [128,20,18],[128,20,18]->[128,20,20]

            self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
            self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
                                                                              padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
                                                                              num_units=INC_DIM, num_heads=4, dropout_rate=0.5, is_training=self.is_training, name="his_item_user_attention_output")

            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
            # ADD但是没有layer_normal
            # self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]
            self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]

            self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]

            self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM)

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, name="multihead_attention_outputs", num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                self.target_item_user_emb = tf.concat([self.uid_emb, self.target_item_emb], -1)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_user_emb, multihead_attention_outputs_v2, self.context_embedding, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 建模历史物品的历史购买用户-----------------------------------------------------
        # with tf.name_scope('his_items_his_users'):
        # self.his_item_user_attention_output = din_attention(query=tf.tile(self.uid_emb, [1, tf.shape(self.his_item_his_users_emb)[1] * tf.shape(self.his_item_his_users_emb)[2]]),  # [128,18]-> [128,20*10*18]
        #                                                     facts=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]),  # [128*20,10,18]
        #                                                     mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
        #                                                     need_tile=False, stag="DIN_U2U_attention")  # 返回的是[128*20,1,18]
        # self.his_item_user_attention_output = tf.reshape(tf.reduce_sum(self.his_item_user_attention_output, 1), [-1, tf.shape(self.mask)[-1], 18])  # [128*20,1,18]->[128*20,18]->[128,20,18]
        # item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(self.mask, axis=2), tf.expand_dims(self.mask, axis=1)), dtype=tf.int32)  # [128,20,1],[128,1,20]->[128,20,20]
        # self.his_item_user_transformer_output = transformer_model(self.his_item_user_attention_output, hidden_size=EMBEDDING_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='his_item_user_transformer_output', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
        # self.his_item_user_transformer_output_sum = tf.layers.dense(tf.reshape(self.his_item_user_transformer_output, [tf.shape(self.his_item_user_transformer_output)[0], 360]), EMBEDDING_DIM * 2, name='his_item_user_transformer_output_sum')  # [128,20,18]

        # self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18]->[128,50,20,36]
        # self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
        #                                                                   padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
        #                                                                   num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="his_item_user_attention_output")
        #
        # self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
        # self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
        # # ADD但是没有layer_normal
        # self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]
        # self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]
        #
        # self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]
        # # self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
        # self.his_item_user_att_sum, _ = attention_net_v1(enc=self.his_item_user_avg, sl=self.seq_len_ph,  # enc=encoder=keys [128,20,36]
        #                                                  dec=self.user_embedded,  # dec=decoder=query[128,36]
        #                                                  num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
        #                                                  is_training=self.is_training, reuse=False, scope='his_item_user_att_sum')  # 返回值是[128,36]
        # -------------------------------------------------------------------------------- item module -------------------------------------------------------------------------------

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50]-> [128,50,18],[128,50,1]-> [128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]-> [128,36]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18]

        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]->[128,50,1]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # ----------------------------------------------------- 处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')  # 将当前用户的维度变成INC_DIM
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')

        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            #  得到目标物品的每个历史购买用户的表征
            # self.item_user_his_items_emb[128,50,20,36]
            item_his_users_item_truncated_len = 10  # 目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            self.item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            self.item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(self.item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,50]->[128*50]得到[128*50,k,36],[128*50],[128*50,k,1]
            # self.item_user_his_k_items_emb = tf.reshape(self.item_user_his_k_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1, INC_DIM])  # [128*50,k,36] -> [128,50,k,36]
            # self.item_user_seq_truncated_lens = tf.reshape(self.item_user_seq_truncated_lens, tf.shape(self.item_his_users_items_len))  # [128*50] -> [128,50]
            # self.item_user_seq_truncated_mask = tf.reshape(self.item_user_seq_truncated_mask, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1, 1])  # [128*50,k,1]->[128,50,k,1]
            print("self.item_user_his_k_items_emb", self.item_user_his_k_items_emb.get_shape())
            item_user_his_attention_output = self_multi_head_attn(inputs=self.item_user_his_k_items_emb,  # [128*50,k,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*50,k,1]->[128*50,k]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            print("self.item_user_his_attention_output", item_user_his_attention_output.get_shape())
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)
            # ADD但是没有layer_normal
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*50,k,36]
            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*50,k,36],[128*50,k,1]->[128*50,k,36]->[128*50,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM]) + self.item_bh_time_emb_padding  # [128*50,36]->[128,50,36]
            # --------------------------------------------------------- 得到目标物品的每个历史购买用户的表征所形成的群体兴趣 -----------------------------------------------------------
            # hidden_size = 64
            # num_interests = 10
            # num_heads = num_interests
            #
            # target_item_his_users_item_hidden = tf.layers.dense(item_user_his_items_avg, hidden_size * 4, activation=tf.nn.tanh, name='users_interest_1')  # [128,50,36]->[128,50,256]
            # item_att_w = tf.layers.dense(target_item_his_users_item_hidden, num_heads, activation=None, name='users_interest_2', reuse=tf.AUTO_REUSE)  # [128,50,256]->[128,50,10]
            # item_att_w = tf.transpose(item_att_w, [0, 2, 1])  # [128,50,10]->[128,10,50]
            # atten_mask = tf.tile(tf.expand_dims(self.item_user_his_mask, axis=1), [1, num_heads, 1])  # [128,50]->[128,1,50]->[128,10,50]
            # paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)  # [128,10,50]
            # item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            # item_att_w = tf.nn.softmax(item_att_w)  # [128,10,50]->[128,10,50]
            # target_item_his_users_interest_emb = tf.matmul(item_att_w, item_user_his_items_avg)  # [128,10,50]*[128,50,36]->[128,10,36]

            # --------------------------------------------------------- 得到当前用户的表征 -----------------------------------------------------------
            current_user_interest_emb = self_multi_head_attn(inputs=self.his_item_emb, name="current_user_interest_emb", padding_mask=self.mask, num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training)  # [128,20,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,20,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * tf.expand_dims(self.mask, -1), axis=1)  # [128,20,36]*[128,20,1]->[128,20,36]->[128,36]

            interest_emb, _ = attention_net_v1(enc=item_user_his_items_avg, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,4,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                               dec=current_user_interest_emb,  # dec=decoder=query[128,36]
                                               num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                               is_training=self.is_training, reuse=False, scope='interest_emb')

        # inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, interest_emb], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        # interest_emb,self.his_item_user_att_sum, self.his_item_user_att_sum * item_user_his_eb_att_sum, self.his_item_user_att_sum * self.item_user_his_eb_sum,
        inp = tf.concat([inp, interest_emb, current_user_interest_emb * self.target_item_emb, item_user_his_eb_att_sum, self.item_user_his_eb_sum], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel_0_last_Self(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Self, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Self_notiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Self_notiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, None, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Self_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Self_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                _, context_emb = dynamic_rnn(GRUCell(36), inputs=his_item_with_tiv_emb * tf.expand_dims(self.mask, axis=-1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru_for_pos_with_tiv_context")
                context_emb = tf.expand_dims(context_emb, axis=1)  # [128,20]->[128,1,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    multihead_attention_outputs_v2 *= context_emb
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_test(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_multi_head(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 324  # 18*18=324=4*9*9
                # context_input一定要mask掉不必要的噪声
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_input = self_multi_head_attn(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,324]
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_multi_head_new(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head_new, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class DDSM1_TIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DDSM1_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        with tf.name_scope('TIEN_item_module'):
            item_his_users_maxlen = 30
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            # 将user_embedding转换成2*Embedding
            self.item_user_his_uid_batch_embedded = tf.layers.dense(self.item_user_his_uid_batch_embedded, INC_DIM, name='item_user_his_uid_batch_embedded_2dim')
            self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=5)  # [128,50,18],[128],[128,50,1]
            self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_k * self.item_bh_mask_k, 1)  # [128,50,18],[128,50,1]-># [128,18]

            # near item_his_users_maxlen behavior
            self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
            self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18]

            # 2.sequential modeling for user/item behaviors
            with tf.name_scope('rnn'):
                gru_outputs_i, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_ib")  # 建模物品行为序列
            # 3.将时间的embedding变成一样的维度
            with tf.name_scope('time-signal'):
                self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
            # 4. attention layer
            with (tf.name_scope('att')):
                # 对当前用户行为建模的attention，类似DIN
                # 5. prevent noisy
                self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_t * self.item_bh_mask_t, 1)  # 这里是先算求和，下面再算平均，要不然分母会偏大

                item_bh_seq_len = (tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32), [-1, 1]) + 1)  # [128,36]+[128,36]
                dec = tf.layers.dense(self.uid_emb, INC_DIM, name='uid_batch_embedded_2dim') + self.item_bh_embedded_sum_t / (tf.where(tf.equal(item_bh_seq_len, tf.zeros_like(item_bh_seq_len)), tf.ones_like(item_bh_seq_len), item_bh_seq_len))
                gru_outputs_ib_with_t = gru_outputs_i + self.item_bh_time_emb_padding  # [128,50,36]
                i_att, _ = attention_net_v1(enc=gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ib', value=gru_outputs_i)
            # 5.time-aware representation layer
            with tf.name_scope('time_gru'):
                _, it_state = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_emb_padding, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_it")
                i_att_ta = i_att + it_state
            inp = tf.concat([inp,self.item_bh_embedded_sum, i_att, i_att_ta], 1)
        self.build_fcn_net(inp)

class DDSM1_DRINK(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DDSM1_DRINK, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        with tf.name_scope('DMIN_item_module'):
            item_his_users_maxlen = 30
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

            # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
            self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
            self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

            # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
            # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
            self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],[128,50,1]
            self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)

            # sequential modeling for item behaviors
            with tf.name_scope('target_item_his_users_representation'):
                # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
                # 对历史物品的用户和时间做element-wise_sum_pooling
                self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
                item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
                # 拼接上多个cls的embedding
                item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
                item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
                # 这里对拼接的embedding过Transformer的encoder部分
                att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), tf.float32), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
                item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
                self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
                # -----------------------------------------------------处理users的表示 -----------------------------------------------------
                self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
                item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :], sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                               dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                               num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                               is_training=False, reuse=False, scope='item_user_his_eb_att')
                # -----------------------------------------------------处理cls的表示 -----------------------------------------------------
                item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]  # [128,3,36]这里只包含cls的embedding
                user_embs_for_cls = tf.tile(tf.expand_dims(self.user_embedded, 1), [1, ITEM_BH_CLS_CNT, 1])  # [128,36]->[128,1,36]->[128,3,36]
                # dot
                item_bh_cls_dot = item_bh_cls_embs * user_embs_for_cls  # [128,3,36],[128,3,36]->[128,3,36]
                item_bh_cls_dot = tf.reshape(item_bh_cls_dot, [-1, INC_DIM * ITEM_BH_CLS_CNT])  # [128,3,36]->[128,108]
                # matmul
                item_bh_cls_mat = tf.matmul(item_bh_cls_embs, user_embs_for_cls, transpose_b=True)  # [128,3,36],[128,3,36]->[128,3,36],[128,36,3]->[128,3,3]
                item_bh_cls_mat = tf.reshape(item_bh_cls_mat[:, 0, :], [-1, ITEM_BH_CLS_CNT])  # [128,1,3]->[128,3]这里取0的原因是因为冗余了，[0,1,2]维度的值一样

            # Decoupling
            with tf.name_scope('decoupling'):
                _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
                _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
                decoupling_part = rnn_outputs1 + rnn_outputs2
            inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, item_bh_cls_dot, item_bh_cls_mat, decoupling_part], 1)
        self.build_fcn_net(inp)

class myModel_0_last_Context_multi_head_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接

                self.denoise_embedding = tf.get_variable("denoise_embedding", [1, EMBEDDING_DIM * 8])  # [1,36]
                self.denoise_embedding = tf.layers.dense(self.denoise_embedding, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                self.denoise_embedding = tf.layers.dense(self.denoise_embedding, EMBEDDING_DIM * 2)
                # 拼接上多个cls的embedding
                denoise_embedding_tile = tf.tile(tf.expand_dims(self.denoise_embedding, 0), [tf.shape(his_item_with_tiv_emb)[0], 1, 1])  # [1,36]->[1,1,36]->[128,1,36]
                item_bh_drink_trm_input = tf.concat([denoise_embedding_tile, his_item_with_tiv_emb], axis=1)  # [128,1,36],[128,20,36]->[128,21,36]
                # 这里对拼接的embedding过Transformer的encoder部分
                tile_mask = tf.concat([tf.cast(tf.ones([tf.shape(self.mask)[0], 1]), tf.float32), tf.reshape(self.mask, [tf.shape(self.mask)[0], -1])], 1)  # [128,1],[128,20]->[128,21]

                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(item_bh_drink_trm_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=tile_mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                # context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, None, tile_mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, multihead_attention_outputs_v2[:, 0, :], att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_multi_head_new_DIB(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head_new_DIB, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 10
            target_item_his_users_maxlen = 10
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得历史物品的历史购买用户的位置向量
            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)  # 5
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [-1, his_items_his_users_maxlen, position_embedding_size])  # [128*20,5,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 30
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]
            self.his_item_bh_k_tiv_emb, _, _ = mapping_to_k(tf.reshape(self.his_item_his_users_tiv_emb, [-1, tf.shape(self.his_item_his_users_tiv_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.his_position_k_his_users_eb = self.position_his_items_his_users_eb[:, :his_items_his_users_maxlen, :]  # [128*20,5,2]->[128*20,5,2]注意position_embedding这里只要从前面保留就行

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb, self.his_position_k_his_users_eb, self.his_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            self.target_item_bh_k_tiv_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.target_position_k_his_users_eb = self.position_his_users_eb[:, :target_item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb, self.target_position_k_his_users_eb, self.target_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]

                self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, [-1, his_items_maxlen, his_items_his_users_maxlen, INC_DIM])  # [128*20,5,36]->[128,20,5,36]

                user_query_for_his = tf.layers.dense(self.user_embedded, INC_DIM, activation=None)  # [128,18]->[128,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.his_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.his_item_user_attention_output, tf.reshape(tf.squeeze(self.his_item_bh_mask_k, axis=-1), [-1, his_items_maxlen, his_items_his_users_maxlen]))  # [128,36],[128,20,5,36]->[128,20,36]

                self.his_item_user_avg = tf.squeeze(self.his_item_user_avg, axis=2)  # [128,20,1,36]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]
                self.target_item_user_attention_output = tf.expand_dims(self.target_item_user_attention_output, axis=1)  # [128,5,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.target_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.target_item_user_attention_output, tf.expand_dims(tf.squeeze(self.target_item_bh_mask_k, axis=-1), axis=1))  # [128,36],[128,20,5,36]->[128,1,36]
                self.target_item_user_avg = tf.squeeze(tf.squeeze(self.target_item_user_avg, axis=1), axis=1)  # [128,1,1,36]->[128,1,36]->[128,36]

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_emb, self.target_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        self.target_item_emb_dynamic = tf.layers.dense(tf.concat([self.target_item_emb_dynamic, self.user_embedded], axis=-1), EMBEDDING_DIM * 2)
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_multi_head_nodecouple(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head_nodecouple, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), multihead_attention_outputs], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(multihead_attention_outputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_multi_head_notiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_multi_head_notiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                # context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, None, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_test_bit_wsie(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_test_bit_wsie, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                emb_matrix_size = 324  # 18*18=324=4*9*9
                weight_matrix_size = 324
                # context_input一定要mask掉不必要的噪声
                context_input_1 = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input_1)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_input_1 = self_multi_head_attn(context_input_1, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_1")
                context_input_1 = tf.reshape(context_input_1 * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                context_embs_matrix = get_context_matrix(context_input_1, embed_dims=[emb_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,324]

                context_input_2 = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input_2)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_input_2 = self_multi_head_attn(context_input_2, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
                context_input_2 = tf.reshape(context_input_2 * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                context_weights_matrix = get_context_matrix(context_input_2, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,18]

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_bit_wise(his_item_with_tiv_emb, context_embs_matrix, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_0_last_Context_test_vector_wsie(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_0_last_Context_test_vector_wsie, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                emb_matrix_size = 324  # 18*18=324=4*9*9
                weight_matrix_size = 36
                # context_input一定要mask掉不必要的噪声
                context_input_1 = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input_1)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_input_1 = self_multi_head_attn(context_input_1, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_1")
                context_input_1 = tf.reshape(context_input_1 * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                context_embs_matrix = get_context_matrix(context_input_1, embed_dims=[emb_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,324]

                context_input_2 = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input_2)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_input_2 = self_multi_head_attn(context_input_2, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
                context_input_2 = tf.reshape(context_input_2 * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                context_weights_matrix = get_context_matrix(context_input_2, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,18]

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_vector_wise(his_item_with_tiv_emb, context_embs_matrix, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel_test_comirec(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_test_comirec, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1, _ = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 324  # 18*18=324
                context_input = tf.concat([self.target_item_emb, tf.reshape(his_item_with_tiv_emb, [tf.shape(his_item_with_tiv_emb)[0], 20 * 36])], axis=-1)  # [128,56].[128,20],[128,20*56]->[128,1120]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                # multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_position_embeddings_var = tf.get_variable("item_his_position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_position_embeddings_var, self.position_his_users)  # [20,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_users_eb.get_shape().as_list()[-1]])  # [128,20,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],item_his_users_maxlen-> [128,50,18],[128],[128,50,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],item_his_users_maxlen-> [128,50,18],[128],[128,50,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 324  # 18*18
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50,1]->[128,50]
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1)  # [128,50,2],[128,50,18],[128,50,18]->[128,50,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,50,38]->[128,50,36]
            # context_input = tf.concat([tf.reshape(item_bh_k_input, [-1, item_his_users_maxlen * INC_DIM])], axis=-1)  # [128,18],[128,36],[128,38],[128,50],[128,50*38]->[128,2042]
            # context_weights_matrix = get_context_matrix(context_input, [weight_matrix_size], is_training=self.is_training)

            target_item_his_att_users_output = self_multi_head_attn(item_bh_k_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            # context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,50,2],[128,50,18]->[128,50,20]
            # --------------------------------------------------------- 得到目标物品的每个历史购买用户的表征所形成的群体兴趣 -----------------------------------------------------------
            hidden_size = 36
            num_interests = 10
            num_heads = num_interests

            target_item_his_users_item_hidden = tf.layers.dense(target_item_his_att_users_output, hidden_size * 4, activation=tf.nn.tanh, name='users_interest_1')  # [128,20,36]->[128,50,144]
            item_att_A = tf.layers.dense(target_item_his_users_item_hidden, num_heads, activation=None, name='users_interest_2', reuse=tf.AUTO_REUSE)  # [128,20,144]->[128,20,10]
            item_att_A = tf.transpose(item_att_A, [0, 2, 1])  # [128,50,10]->[128,10,50]

            atten_mask = tf.tile(tf.expand_dims(att_mask_k_input, axis=1), [1, num_heads, 1])  # [128,20]->[128,1,20]->[128,10,20]
            paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)  # [128,10,20]
            item_att_A = tf.where(tf.equal(atten_mask, 0), paddings, item_att_A)
            item_att_A = tf.nn.softmax(item_att_A)  # [128,10,20]->[128,10,20]
            target_item_his_users_interest_emb = tf.matmul(item_att_A, target_item_his_att_users_output)  # [128,10,20]*[128,20,36]->[128,10,36]

            interest_emb, _ = attention_net_v1(enc=target_item_his_users_interest_emb, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,4,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                               dec=self.uid_emb,  # dec=decoder=query[128,36]
                                               num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                               is_training=self.is_training, reuse=False, scope='interest_emb')

        inp = tf.concat([inp, interest_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.aux_loss += self.calculate_atten_loss(item_att_A, hidden_size)
        self.build_fcn_net(inp)


class myModel0_test_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gruaux_loss_1")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                self.context_weights_matrices1 = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (his_items_maxlen+1) * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    self.context_weights_matrices1.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(self.context_weights_matrices1, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接

                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            self.context_weights_matrices2 = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                self.context_weights_matrices2.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(self.context_weights_matrices2, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel_base(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_base, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                weight_matrix_size = 324  # 18*18=324
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                context_input = tf.concat([tf.reshape(his_item_with_tiv_emb, [tf.shape(his_item_with_tiv_emb)[0], 20 * 56])], axis=-1)  # [128,56].[128,20],[128,20*56]->[128,1120]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
                context_weights_matrix1 = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)

                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2)
                # multihead_attention_outputs = context_aware_self_attention(his_item_with_tiv_emb, context_weights_matrix, embed_dim=EMBEDDING_DIM * 2, num_units=weight_matrix_size ** 0.5, padding_mask=self.mask)
                # multihead_attention_outputs = context_aware_multi_head_self_attention_ieu(his_item_with_tiv_emb, context_weights_matrix,context_weights_matrix1, num_heads=4, num_units=9, padding_mask=self.mask,dropout_rate=0.2,is_training=self.is_training, causality_mask_bool=False)
                # 这里如果用self-attention的话，要加上point-wise feed_forward
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                # 下面三行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                self.his_item_emb_new = tf.concat([self.his_item_emb, tf.tile(tf.expand_dims(self.target_item_emb, 1), [1, 20, 1])], axis=2)  # [128,20,36],[128,36]
                self.his_item_emb_new = tf.layers.dense(self.his_item_emb_new, EMBEDDING_DIM * 2)

            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb_new[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                # context_input = tf.concat([self.uid_emb, tf.reshape(multihead_attention_outputs, [tf.shape(multihead_attention_outputs)[0], 20 * 36])], axis=-1)#
                # context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
                self.context_embedding = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2(multihead_attention_outputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0.2, is_training=self.is_training, name="multihead_attention_outputss")
                # multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, self.context_embedding, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        self.build_fcn_net(inp)


class myModel0_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gruaux_loss_1")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (his_items_maxlen+1) * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接

                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_1user(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_1user, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 1  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        # print("self.item_bh_k_user_emb.get_shape()",self.item_bh_k_user_emb.get_shape()) #[128,1,18]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            users_interest = self.user_embedded * tf.squeeze(item_bh_k_input, axis=1)  # [128,36],[128,1,36]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, users_interest], 1)
        self.build_fcn_net(inp)


class myModel0_test_only_short_old(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_only_short_old, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [20, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_user_position_embeddings_var = tf.get_variable("item_his_user_position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_user_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]

            self.position_item_users_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_item_users_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]

            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,1,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, tf.shape(item_user_his_k_tivs_emb))  # [128,30,5,18]->[128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]
            item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                  padding_mask=tf.squeeze(self.item_user_seq_truncated_mask),  # [128*30,5,1]->[128*30,5]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,
                                                             padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]),
                                                             num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="current_user_interest_emb")  # [128,5,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            att_mask_k_input = tf.squeeze(self.item_bh_mask_t)  # [128,30,1]->[128,30]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, short_group_interest], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel0_test_only_short(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_only_short, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,30,36],[128,30,1]->[128,36]
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            att_mask_k_input = tf.squeeze(self.item_bh_mask_t, axis=-1)  # [128,30,1]->[128,30]
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel0_test_only_short_1(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_only_short_1, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,36],[128,20*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_user_position_embeddings_var = tf.get_variable("item_his_user_position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_user_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 1  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]

            self.position_item_users_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_item_users_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]

            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,1,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, tf.shape(item_user_his_k_tivs_emb))  # [128,30,5,18]->[128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_k_items_emb_with_tiv, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_k_items_emb_with_tiv  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            mul_att_current_user_output = tf.layers.dense(current_user_k_bh_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_k_bh_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            att_mask_k_input = tf.squeeze(self.item_bh_mask_t, axis=-1)  # [128,30,1]->[128,30]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, short_group_interest], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel0_test_only_short_DIB(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_only_short_DIB, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 5
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0], tf.shape(self.his_item_emb)[1], -1, position_embedding_size])  # [128,20,2]

            self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.position_his_items_his_users_eb, self.his_item_his_users_tiv_emb], axis=-1), EMBEDDING_DIM * 2)  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]
            self.his_item_emb_tile = tf.tile(tf.expand_dims(self.his_item_emb, 2), [1, 1, his_items_his_users_maxlen, 1])  # [128,20,36]->[128,20,1,36]->[128,20,5,36]
            self.uid_emb_tile = tf.reshape(tf.tile(tf.expand_dims(self.uid_emb, 1), [1, his_items_maxlen * his_items_his_users_maxlen, 1]), [-1, his_items_maxlen, his_items_his_users_maxlen, EMBEDDING_DIM])  # [128,18]->[128,1,18]->[128,20*5,18]->[128,20,5,18]
            self.uid_with_his_items_emb = tf.concat([self.his_item_emb_tile, self.uid_emb_tile], axis=-1)  # [128,20,5,36]
            self.his_item_user_avg = din_attention_DIB_V2(self.uid_with_his_items_emb, self.his_item_his_users_emb, self.his_item_user_mask)  # [128,18],[128,20,5,36],[128,20,1,36]
            # self.uid_with_target_emb = tf.concat([self.uid_emb, self.target_item_emb], axis=-1)
            # self.uid_with_his_items_emb = tf.concat([self.uid_emb, self.target_item_emb], axis=-1)  # [128,20,5,36]
            # self.his_item_user_avg = din_attention_DIB(self.uid_with_target_emb, self.his_item_his_users_emb, self.his_item_user_mask)  # [128,18],[128,20,5,36],[128,20,1,36]
            self.his_item_user_avg = tf.reduce_sum(self.his_item_user_avg, 2)  # [128,20,1,36]->[128,20,36]
            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_user_avg, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_user_position_embeddings_var = tf.get_variable("item_his_user_position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_user_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]

            self.position_item_users = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_item_users)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]
            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]

            item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*30,5,1]->[128*30,5]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            item_user_his_items_avg = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, item_user_his_items_avg], axis=-1) * self.item_bh_mask_t  # [128,30,2], [128,30,18], [128,30,18]-> [128,30,38]
            item_user_his_items_avg = tf.layers.dense(item_user_his_items_avg, INC_DIM)  # [128,30,38]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,
                                                             padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]),
                                                             num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="current_user_interest_emb")  # [128,5,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users" + str(i))
            short_group_interest = tf.reduce_sum(attention_output, 1)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel0_test_notiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_notiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                # context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, None, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]

            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_usernotiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_usernotiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            # context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, None, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_item_usernotiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_item_usernotiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                # context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, None, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            # context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, None, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_user_self(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_user_self, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = self_multi_head_attn_v2(item_bh_k_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_user_item_self(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_user_item_self, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = self_multi_head_attn_v2(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = self_multi_head_attn_v2(item_bh_k_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_DMIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_DMIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_items_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [20,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_users_eb.get_shape().as_list()[-1]])  # [128,20,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]

            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel0_test_comirec(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel0_test_comirec, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 324  # 18*18=324
                context_input = tf.concat([self.target_item_emb, tf.reshape(his_item_with_tiv_emb, [tf.shape(his_item_with_tiv_emb)[0], 20 * 36])], axis=-1)  # [128,56].[128,20],[128,20*56]->[128,1120]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_position_embeddings_var = tf.get_variable("item_his_position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_position_embeddings_var, self.position_his_users)  # [20,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_users_eb.get_shape().as_list()[-1]])  # [128,20,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 324  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([self.user_embedded, tf.reshape(item_bh_k_input, [-1, item_his_users_maxlen * INC_DIM])], axis=-1)  # [128,18],[128,36],[128,38],[128,30],[128,30*38]->[128,2042]
            context_weights_matrix = get_context_matrix(context_input, [weight_matrix_size], is_training=self.is_training)
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, multihead_attention_outputs_v2, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        inp = tf.concat([inp, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)
        self.build_fcn_net(inp)


class myModel1_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]
            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_pos = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_pos,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users" + str(i))
            short_group_interest = tf.reduce_sum(attention_output, 1)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel1_test_with_target_item(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_target_item, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / tf.reduce_sum(current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb,self.target_item_emb],axis=-1),INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users" + str(i))
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel1_test_SENET(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_SENET, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                feature_fields = []
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        # inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
                        feature_fields.append(att_fea)
        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    # inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
                    feature_fields.append(att_fea)
        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb,  # [128*30,5,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*30,5,1]->[128*30,5]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            item_user_his_items_avg = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, item_user_his_items_avg], axis=-1) * self.item_bh_mask_t  # [128,30,2], [128,30,18], [128,30,18]-> [128,30,38]
            item_user_his_items_avg = tf.layers.dense(item_user_his_items_avg, INC_DIM)  # [128,30,38]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(self.his_item_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,
                                                             padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]),
                                                             num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="current_user_interest_emb")  # [128,5,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users" + str(i))
            short_group_interest = tf.reduce_sum(attention_output, 1)
            feature_fields.append(short_group_interest)  # 9个[128,36]
            feature_fields_stack = tf.stack(feature_fields, axis=1)  # [[128,36],[128,36],[128,36],[128,36],[128,36]]->[128,5,36]
            feature_fields_avg = tf.reduce_mean(feature_fields_stack, axis=-1)
            senet_middle = tf.layers.dense(feature_fields_avg, 3, activation=tf.nn.sigmoid)  # [128,20]->[128,10]
            senet_output = tf.layers.dense(senet_middle, 9)  # [128,20]
            reweight_feature_fields = tf.expand_dims(senet_output, -1) * feature_fields_stack  # [128,9,36]->[128,324]
            feature_fields_concat = tf.reshape(reweight_feature_fields, [tf.shape(reweight_feature_fields)[0], -1])
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, feature_fields_concat], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel1_test_with_user_tiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_user_tiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gruaux_loss_1")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                self.context_weights_matrices1 = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (his_items_maxlen+1) * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    self.context_weights_matrices1.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(self.context_weights_matrices1, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接

                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            self.context_weights_matrices2 = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                self.context_weights_matrices2.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(self.context_weights_matrices2, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]

            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel1_test_with_user_tiv_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_user_tiv_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_aux"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gruaux_loss_1")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                self.context_weights_matrices1 = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (his_items_maxlen+1) * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    self.context_weights_matrices1.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(self.context_weights_matrices1, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接

                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            self.context_weights_matrices2 = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                self.context_weights_matrices2.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(self.context_weights_matrices2, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]

        self.build_fcn_net(inp)

class myModel1_test_with_user_tiv_GRU(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(myModel1_test_with_user_tiv_GRU, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        self.context_loss_weight_mean = 0.
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]
        his_item_with_tiv_emb = tf.concat([self.position_his_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
        his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('rnn_1'):
            gru_outputs_u, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru_ub")  # 类似DIEN建模用户行为序列
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            u_att, _ = attention_net_v1(enc=gru_outputs_u, sl=self.seq_len_ph, dec=self.target_item_emb, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, u_att], 1)
        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)

class myModel1_test_with_user_tiv_DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_user_tiv_DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        self.context_loss_weight_mean = 0.
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]
        his_item_with_tiv_emb = tf.concat([self.position_his_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
        his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru1")
        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
        self.aux_loss = aux_loss_1
        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            _, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)

class myModel1_test_with_user_tiv_DMIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_user_tiv_DMIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        self.context_loss_weight_mean = 0.
        maxlen = 20
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,2]
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[-1]])  # [128,20,2]
        his_item_with_tiv_emb = tf.concat([self.position_his_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
        his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_for_items"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context_origin(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag="Attention_layer_for_items" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)

class myModel1_test_with_user_tiv_SENET(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel1_test_with_user_tiv_SENET, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [50, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]

            with tf.name_scope("multi_head_attention_1"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), his_item_with_tiv_emb], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21*36]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(his_item_with_tiv_emb, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")
                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                feature_fields = []
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        # inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
                        feature_fields.append(att_fea)
        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_user_position_embeddings_var = tf.get_variable("item_his_user_position_embeddings_var", [50, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_user_position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # 注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 51 * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    # inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
                    feature_fields.append(att_fea)
        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]

            self.position_item_users = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_item_users)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]
            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM)  # [128*30,5,56]->[128*30,5,36]

            item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*30,5,1]->[128*30,5]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*30,5,36]
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]

            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            item_user_his_items_avg = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, item_user_his_items_avg], axis=-1) * self.item_bh_mask_t  # [128,30,2], [128,30,18], [128,30,18]-> [128,30,38]
            item_user_his_items_avg = tf.layers.dense(item_user_his_items_avg, INC_DIM)  # [128,30,38]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,
                                                             padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]),
                                                             num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="current_user_interest_emb")  # [128,5,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users" + str(i))
            short_group_interest = tf.reduce_sum(attention_output, 1)
            feature_fields.append(short_group_interest)  # 9个[128,36]
            feature_fields_stack = tf.stack(feature_fields, axis=1)  # [[128,36],[128,36],[128,36],[128,36],[128,36]]->[128,5,36]
            feature_fields_avg = tf.reduce_mean(feature_fields_stack, axis=-1)
            senet_middle = tf.layers.dense(feature_fields_avg, 3, activation=tf.nn.sigmoid)  # [128,20]->[128,10]
            senet_output = tf.layers.dense(senet_middle, 9)  # [128,20]
            reweight_feature_fields = tf.expand_dims(senet_output, -1) * feature_fields_stack  # [128,9,36]->[128,364]
            feature_fields_concat = tf.reshape(reweight_feature_fields, [tf.shape(reweight_feature_fields)[0], 324])  # [128,364]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, feature_fields_concat], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 10
            target_item_his_users_maxlen = 10
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得历史物品的历史购买用户的位置向量
            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)  # 5
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [-1, his_items_his_users_maxlen, position_embedding_size])  # [128*20,5,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 30
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]
            self.his_item_bh_k_tiv_emb, _, _ = mapping_to_k(tf.reshape(self.his_item_his_users_tiv_emb, [-1, tf.shape(self.his_item_his_users_tiv_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.his_position_k_his_users_eb = self.position_his_items_his_users_eb[:, :his_items_his_users_maxlen, :]  # [128*20,5,2]->[128*20,5,2]注意position_embedding这里只要从前面保留就行

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb, self.his_position_k_his_users_eb, self.his_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            self.target_item_bh_k_tiv_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.target_position_k_his_users_eb = self.position_his_users_eb[:, :target_item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb, self.target_position_k_his_users_eb, self.target_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]

                count_his_item_bh_k = tf.reduce_sum(self.his_item_bh_mask_k, axis=1)  # [128*20,5,1]->[128*20,1]
                self.his_item_user_avg = tf.reduce_sum(self.his_item_user_attention_output * self.his_item_bh_mask_k, axis=1) / (tf.where(tf.equal(count_his_item_bh_k, tf.zeros_like(count_his_item_bh_k)), tf.ones_like(count_his_item_bh_k), count_his_item_bh_k))  # [128*20,5,36],[128*20,5,1]->[128*20,5,36]->[128*20,36]
                self.his_item_user_avg = tf.reshape(self.his_item_user_avg, [-1, his_items_maxlen, INC_DIM])  # [128*20,36]->[128,20,36]
                # self.his_item_user_avg = tf.nn.dropout(self.his_item_user_avg, rate=0.2)

                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg, self.his_item_emb - self.his_item_user_avg, self.his_item_emb * self.his_item_user_avg], axis=-1), INC_DIM * 4, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]

                count_target_item_bh_k = tf.reduce_sum(self.target_item_bh_mask_k, axis=1)  # [128,5,1]->[128,1]
                self.target_item_user_avg = tf.reduce_sum(self.target_item_user_attention_output * self.target_item_bh_mask_k, axis=1) / (tf.where(tf.equal(count_target_item_bh_k, tf.zeros_like(count_target_item_bh_k)), tf.ones_like(count_target_item_bh_k), count_target_item_bh_k))  # [128*20,5,36],[128*20,5,1]->[128*20,5,36]->[128*20,36]
                # self.target_item_user_avg = tf.nn.dropout(self.target_item_user_avg, rate=0.2)

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_emb, self.target_item_user_avg, self.target_item_emb - self.target_item_user_avg, self.target_item_emb * self.target_item_user_avg], axis=-1), INC_DIM * 4, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (his_items_maxlen+1) * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        self.target_item_emb_dynamic = tf.layers.dense(tf.concat([self.target_item_emb_dynamic, self.user_embedded], axis=-1), EMBEDDING_DIM * 2)
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM)  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            # -------------------------------------------------------- 得到目标物品历史用户的短期兴趣表征 --------------------------------------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]
            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # ------------------------------------------------------------------------------------------------------------------------------------------------------
            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test_no_postiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test_no_postiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 5
            target_item_his_users_maxlen = 5
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 5
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]

                count_his_item_bh_k = tf.reduce_sum(self.his_item_bh_mask_k, axis=1)  # [128*20,5,1]->[128*20,1]
                self.his_item_user_avg = tf.reduce_sum(self.his_item_user_attention_output * self.his_item_bh_mask_k, axis=1) / (tf.where(tf.equal(count_his_item_bh_k, tf.zeros_like(count_his_item_bh_k)), tf.ones_like(count_his_item_bh_k), count_his_item_bh_k))  # [128*20,5,36],[128*20,5,1]->[128*20,5,36]->[128*20,36]
                self.his_item_user_avg = tf.reshape(self.his_item_user_avg, [-1, his_items_maxlen, INC_DIM])  # [128*20,36]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, 1, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]

                count_target_item_bh_k = tf.reduce_sum(self.target_item_bh_mask_k, axis=1)  # [128,5,1]->[128,1]
                self.target_item_user_avg = tf.reduce_sum(self.target_item_user_attention_output * self.target_item_bh_mask_k, axis=1) / (tf.where(tf.equal(count_target_item_bh_k, tf.zeros_like(count_target_item_bh_k)), tf.ones_like(count_target_item_bh_k), count_target_item_bh_k))  # [128*20,5,36],[128*20,5,1]->[128*20,5,36]->[128*20,36]

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_emb, self.target_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, 1, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(his_item_with_tiv_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1)  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM) * self.item_bh_mask_t  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test_1(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test_1, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 20
            target_item_his_users_maxlen = 20
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
            num_units = 9
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得历史物品的历史购买用户的位置向量
            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)  # 5
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [-1, his_items_his_users_maxlen, position_embedding_size])  # [128*20,5,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 30
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]
            self.his_item_bh_k_tiv_emb, _, _ = mapping_to_k(tf.reshape(self.his_item_his_users_tiv_emb, [-1, tf.shape(self.his_item_his_users_tiv_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.his_position_k_his_users_eb = self.position_his_items_his_users_eb[:, :his_items_his_users_maxlen, :]  # [128*20,5,2]->[128*20,5,2]注意position_embedding这里只要从前面保留就行

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb, self.his_position_k_his_users_eb, self.his_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            self.target_item_bh_k_tiv_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.target_position_k_his_users_eb = self.position_his_users_eb[:, :target_item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb, self.target_position_k_his_users_eb, self.target_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                # self.his_item_bh_k_user_emb_with_pos_tiv = tf.stop_gradient(self.his_item_bh_k_user_emb_with_pos_tiv)
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]
                self.his_item_user_attention_output = self.his_item_user_attention_output + self.his_item_user_attention_output_origin
                self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, [-1, his_items_maxlen, his_items_his_users_maxlen, INC_DIM])  # [128*20,5,36]->[128,20,5,36]

                user_query_for_his = tf.layers.dense(self.user_embedded, INC_DIM, activation=None)  # [128,18]->[128,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.his_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.his_item_user_attention_output, tf.reshape(tf.squeeze(self.his_item_bh_mask_k, axis=-1), [-1, his_items_maxlen, his_items_his_users_maxlen]))  # [128,36],[128,20,5,36]->[128,20,36]

                self.his_item_user_avg = tf.squeeze(self.his_item_user_avg, axis=2)  # [128,20,1,36]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                # self.target_item_bh_k_user_emb_with_pos_tiv = tf.stop_gradient(self.target_item_bh_k_user_emb_with_pos_tiv)
                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]
                self.target_item_user_attention_output = self.target_item_user_attention_output + self.target_item_user_attention_output_origin

                self.target_item_user_attention_output = tf.expand_dims(self.target_item_user_attention_output, axis=1)  # [128,5,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.target_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.target_item_user_attention_output, tf.expand_dims(tf.squeeze(self.target_item_bh_mask_k, axis=-1), axis=1))  # [128,36],[128,20,5,36]->[128,1,36]
                self.target_item_user_avg = tf.squeeze(tf.squeeze(self.target_item_user_avg, axis=1), axis=1)  # [128,1,1,36]->[128,1,36]->[128,36]

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_emb, self.target_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb_2], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * EMBEDDING_DIM * 2])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix_item = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix_item, num_heads=4, num_units=num_units, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        self.target_item_emb_dynamic = tf.layers.dense(tf.concat([self.target_item_emb_dynamic, self.user_embedded], axis=-1), EMBEDDING_DIM * 2)
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM) * self.item_bh_mask_t  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * EMBEDDING_DIM * 2])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix_user = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix_user, num_heads=4, num_units=num_units, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub

                # target_item_his_att_users_output_sub = layer_norm(target_item_his_att_users_output_sub, name="target_item_his_att_users_output_sub_layernorm" + str(i))

                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            # item_user_his_attention_output = layer_norm(item_user_his_attention_output,name="item_user_his_attention_output_layernorm")  # [128,20,36]->[128,20,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        # addition = tf.layers.dense(tf.concat([self.user_embedded,self.target_item_emb,self.user_embedded*self.target_item_emb,self.user_embedded-self.target_item_emb],axis=-1),2*INC_DIM,activation=tf.nn.relu)
        # addition = tf.layers.dense(addition,INC_DIM,activation=tf.nn.relu)
        # addition = tf.layers.dense(addition,INC_DIM,activation=None)

        # addition1 = tf.layers.dense(context_weights_matrix, 4 * INC_DIM, activation=tf.nn.relu)
        # addition1 = tf.layers.dense(addition1, 2 * INC_DIM, activation=tf.nn.relu)
        # addition1 = tf.layers.dense(addition1, INC_DIM, activation=None)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test_2(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test_2, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 5
            target_item_his_users_maxlen = 5
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得历史物品的历史购买用户的位置向量
            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)  # 5
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [-1, his_items_his_users_maxlen, position_embedding_size])  # [128*20,5,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 5
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]
            self.his_item_bh_k_tiv_emb, _, _ = mapping_to_k(tf.reshape(self.his_item_his_users_tiv_emb, [-1, tf.shape(self.his_item_his_users_tiv_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.his_position_k_his_users_eb = self.position_his_items_his_users_eb[:, :his_items_his_users_maxlen, :]  # [128*20,5,2]->[128*20,5,2]注意position_embedding这里只要从前面保留就行

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb, self.his_position_k_his_users_eb, self.his_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            self.target_item_bh_k_tiv_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.target_position_k_his_users_eb = self.position_his_users_eb[:, :target_item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb, self.target_position_k_his_users_eb, self.target_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]
                self.his_item_user_attention_output = self.his_item_user_attention_output + self.his_item_user_attention_output_origin
                self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, [-1, his_items_maxlen, his_items_his_users_maxlen, INC_DIM])  # [128*20,5,36]->[128,20,5,36]

                with tf.variable_scope('din_attention_DIB_with_item', reuse=tf.AUTO_REUSE):
                    self.his_item_user_avg = din_attention_DIB_with_item(self.his_item_emb, self.his_item_user_attention_output, tf.reshape(tf.squeeze(self.his_item_bh_mask_k, axis=-1), [-1, his_items_maxlen, his_items_his_users_maxlen]))  # [128,36],[128,20,5,36]->[128,20,36]

                self.his_item_user_avg = tf.squeeze(self.his_item_user_avg, axis=2)  # [128,20,1,36]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]
                self.target_item_user_attention_output = self.target_item_user_attention_output + self.target_item_user_attention_output_origin
                self.target_item_user_attention_output = tf.expand_dims(self.target_item_user_attention_output, axis=1)  # [128,5,36]->[128,1,5,36]

                item_query_for_target = tf.expand_dims(self.target_item_emb, axis=1)  # [128,1,36]
                with tf.variable_scope('din_attention_DIB_with_item', reuse=tf.AUTO_REUSE):
                    self.target_item_user_avg = din_attention_DIB_with_item(item_query_for_target, self.target_item_user_attention_output, tf.expand_dims(tf.squeeze(self.target_item_bh_mask_k, axis=-1), axis=1))  # [128,36],[128,20,5,36]->[128,1,36]
                self.target_item_user_avg = tf.squeeze(tf.squeeze(self.target_item_user_avg, axis=1), axis=1)  # [128,1,1,36]->[128,1,36]->[128,36]

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_emb, self.target_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * 36])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        self.target_item_emb_dynamic = tf.layers.dense(tf.concat([self.target_item_emb_dynamic, self.user_embedded], axis=-1), EMBEDDING_DIM * 2)
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 50  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1)  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM) * self.item_bh_mask_t  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * 36])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]
            # current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * current_user_bh_mask, axis=1)  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test_3(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test_3, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        self.user_embedded = tf.layers.dense(self.uid_emb, EMBEDDING_DIM * 2, name='user_map_2dim')
        # ----------------------------------------------------- user module -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            his_items_his_users_maxlen = 10
            target_item_his_users_maxlen = 10
            position_embedding_size = 2
            INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
            item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
            num_units = 9
            # 这里是可学习的物品position-embedding
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
            # 获得历史物品的位置向量
            self.position_his_item = tf.range(his_items_maxlen)  # 20
            self.position_his_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,20,2]
            # 获得历史物品的历史购买用户的位置向量
            self.position_his_item_his_users = tf.range(his_items_his_users_maxlen)  # 5
            self.position_his_items_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_item_his_users)  # [5,2]
            self.position_his_items_his_users_eb = tf.tile(self.position_his_items_his_users_eb, [tf.shape(self.his_item_emb)[0] * tf.shape(self.his_item_emb)[1], 1])  # [128*20*5,2]
            self.position_his_items_his_users_eb = tf.reshape(self.position_his_items_his_users_eb, [-1, his_items_his_users_maxlen, position_embedding_size])  # [128*20,5,2]
            # 获得目标物品的历史购买用户的位置向量
            self.position_his_users = tf.range(item_his_users_maxlen)  # 30
            self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [30,2]
            self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*30,2]
            self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,30,2]

            # 截断历史行为物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.seq_len_his_u_ph = tf.count_nonzero(self.his_item_his_users_list, axis=-1)  # [128,20,10]-> [128,20]目标物品每个历史用户行为的长度
            self.his_item_bh_k_user_emb, self.his_item_bh_seq_len_k, self.his_item_bh_mask_k = mapping_to_k(tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,20,10,18],[128,20],5 -> [128*20,5,18],[128*20],[128*20,5,1]
            self.his_item_bh_k_tiv_emb, _, _ = mapping_to_k(tf.reshape(self.his_item_his_users_tiv_emb, [-1, tf.shape(self.his_item_his_users_tiv_emb)[2], EMBEDDING_DIM]), tf.reshape(self.seq_len_his_u_ph, [-1]), k=his_items_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.his_position_k_his_users_eb = self.position_his_items_his_users_eb[:, :his_items_his_users_maxlen, :]  # [128*20,5,2]->[128*20,5,2]注意position_embedding这里只要从前面保留就行

            # 将用户的历史行为物品的历史购买用户加上位置向量和时间间隔向量
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.his_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.his_item_bh_k_user_emb, self.his_position_k_his_users_eb, self.his_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,20,5,18],[128,20,5,2],[128,20,5,18]->[128,20,5,38]->[128,20,5,36]

            # 截断目标物品的历史购买用户，然后加上位置向量和时间间隔向量
            self.target_item_bh_k_user_emb, self.target_item_bh_seq_len_k, self.target_item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,5,18],[128],[128,5,1]
            self.target_item_bh_k_tiv_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=target_item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
            self.target_position_k_his_users_eb = self.position_his_users_eb[:, :target_item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
            with tf.variable_scope('dense_layer_for_users', reuse=tf.AUTO_REUSE):
                self.target_item_bh_k_user_emb_with_pos_tiv = tf.layers.dense(tf.concat([self.target_item_bh_k_user_emb, self.target_position_k_his_users_eb, self.target_item_bh_k_tiv_emb], axis=-1), INC_DIM, name='my_dense')  # [128,5,18],[128,5,2],[128,5,18]->[128,5,38]->[128,5,36]

            # ------------------------------------------------------ 开始生成历史物品和目标物品的动态embedding ------------------------------------------------------
            with tf.variable_scope('self_multi_head_attn_layer_for_DIB', reuse=tf.AUTO_REUSE):
                # self.his_item_bh_k_user_emb_with_pos_tiv = tf.stop_gradient(self.his_item_bh_k_user_emb_with_pos_tiv)
                self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=self.his_item_bh_k_user_emb_with_pos_tiv,  # [128*20,5,36]
                                                                                  padding_mask=tf.squeeze(self.his_item_bh_mask_k, axis=-1),  # [128*20,5]
                                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128*20,5,36]
                self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128*20,5,36]
                self.his_item_user_attention_output = self.his_item_user_attention_output + self.his_item_user_attention_output_origin
                self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, [-1, his_items_maxlen, his_items_his_users_maxlen, INC_DIM])  # [128*20,5,36]->[128,20,5,36]

                user_query_for_his = tf.layers.dense(self.user_embedded, INC_DIM, activation=None)  # [128,18]->[128,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.his_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.his_item_user_attention_output, tf.reshape(tf.squeeze(self.his_item_bh_mask_k, axis=-1), [-1, his_items_maxlen, his_items_his_users_maxlen]))  # [128,36],[128,20,5,36]->[128,20,36]

                self.his_item_user_avg = tf.squeeze(self.his_item_user_avg, axis=2)  # [128,20,1,36]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.his_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic_weight = tf.layers.dense(self.his_item_emb_dynamic_weight, 1, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.his_item_emb_dynamic = self.his_item_emb_dynamic_weight * self.his_item_emb + (1 - self.his_item_emb_dynamic_weight) * self.his_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb_dynamic, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

                # self.target_item_bh_k_user_emb_with_pos_tiv = tf.stop_gradient(self.target_item_bh_k_user_emb_with_pos_tiv)
                self.target_item_user_attention_output_origin = self_multi_head_attn(inputs=self.target_item_bh_k_user_emb_with_pos_tiv,  # [128,5,36]
                                                                                     padding_mask=tf.squeeze(self.target_item_bh_mask_k, axis=-1),  # [128,5]
                                                                                     num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_and_target_item_user_attention_output")
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_layer_for_DIB_dense1")  # [128,5,36]
                self.target_item_user_attention_output = tf.layers.dense(self.target_item_user_attention_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_layer_for_DIB_dense2")  # [128,5,36]
                self.target_item_user_attention_output = self.target_item_user_attention_output + self.target_item_user_attention_output_origin

                self.target_item_user_attention_output = tf.expand_dims(self.target_item_user_attention_output, axis=1)  # [128,5,36]
                with tf.variable_scope('din_attention_DIB_with_user', reuse=tf.AUTO_REUSE):
                    self.target_item_user_avg = din_attention_DIB_with_user(user_query_for_his, self.target_item_user_attention_output, tf.expand_dims(tf.squeeze(self.target_item_bh_mask_k, axis=-1), axis=1))  # [128,36],[128,20,5,36]->[128,1,36]
                self.target_item_user_avg = tf.squeeze(tf.squeeze(self.target_item_user_avg, axis=1), axis=1)  # [128,1,1,36]->[128,1,36]->[128,36]

                self.target_item_emb_dynamic_weight = tf.layers.dense(tf.concat([self.target_item_user_avg], axis=-1), INC_DIM, activation=tf.nn.sigmoid, name="dynamic_weight_dense1")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic_weight = tf.layers.dense(self.target_item_emb_dynamic_weight, 1, activation=tf.nn.sigmoid, name="dynamic_weight_dense2")  # [128,20,36],[128,20,36]->[128,20,72]->[128,20,36]
                self.target_item_emb_dynamic = self.target_item_emb_dynamic_weight * self.target_item_emb + (1 - self.target_item_emb_dynamic_weight) * self.target_item_user_avg  # [128,20,36],[128,20,36],[128,20,2],[128,20,18]->[128,20,92]->[128,36]

            with tf.name_scope("multi_head_attention_aux_loss"):
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2) * tf.expand_dims(self.mask, axis=-1)  # [128,20,56]->[128,20,36]
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.user_embedded, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 81  # 18*18=324=4*9*9
                context_input = tf.concat([tf.expand_dims(self.target_item_emb, 1), self.his_item_emb_dynamic], axis=1)  # [128,36],[128,20,36]->[128,21,36]
                mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), self.mask], 1)  # [128,1],[128,20]->[128,21]
                context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input")
                context_weights_matrices = []  # 初始化空列表以存储每次循环的结果
                for i, context_input in enumerate(context_inputs):
                    # context_input一定要mask掉不必要的噪声
                    context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], 21 * EMBEDDING_DIM * 2])  # [128,21,36]*[128,21,1]->[128,756]
                    context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training, scope=str(i))  # [128,756]->[128,81]
                    context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
                context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324] 循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2_no_weight(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=num_units, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputss")

                context_embedding_for_item = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        self.target_item_emb_dynamic = tf.layers.dense(tf.concat([self.target_item_emb_dynamic, self.user_embedded], axis=-1), EMBEDDING_DIM * 2)
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_emb_dynamic, multihead_attention_outputs_v2, context_embedding_for_item, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        item_his_users_maxlen = 30  # 购买目标物品的历史用户的长度
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # 这里是可学习的用户position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.position_his_users_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his_users)  # [50,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*50,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, position_embedding_size])  # [128,50,2]

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50,1]->[128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]->[128,36]
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.item_bh_k_time_emb, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],30-> [128,30,18],[128],[128,30,1]
        self.position_k_his_users_eb = self.position_his_users_eb[:, :item_his_users_maxlen, :]  # [128,30,2]注意position_embedding这里只要从前面保留就行
        # 因为截断了用户，所以这里连同用户的历史行为也一同截断
        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]
        self.item_user_his_tivs_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [tf.shape(self.item_user_his_tivs_emb)[0], tf.shape(self.item_user_his_tivs_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,18]->[128,50,20*18],[128,50]->[128*50]得到[128,30,20*18],[128],[128,30,1]
        self.item_user_his_tivs_emb = tf.reshape(self.item_user_his_tivs_emb_truncated, [tf.shape(self.item_user_his_tivs_emb)[0], item_his_users_maxlen, -1, EMBEDDING_DIM])  # [128,30,20*18]->[128,30,20,18]
        # ----------------------------------------------------- 处理目标物品历史购买用户users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 81  # 18*18
            item_bh_k_input = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb, self.item_bh_k_user_emb], axis=-1) * self.item_bh_mask_t  # [128,30,2],[128,30,18],[128,30,18]->[128,30,38]
            item_bh_k_input = tf.layers.dense(item_bh_k_input, INC_DIM) * self.item_bh_mask_t  # [128,30,38]->[128,30,36]
            context_input = tf.concat([tf.expand_dims(self.user_embedded, 1), item_bh_k_input], axis=1)  # [128,36],[128,30,36]->[128,31,36]
            mask_context_input = tf.concat([tf.cast(tf.ones([tf.shape(context_input)[0], 1]), tf.float32), tf.squeeze(self.item_bh_mask_t, axis=-1)], 1)  # [128,1],[128,30]->[128,31]
            context_inputs = self_multi_head_attn_v2(context_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=mask_context_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="multihead_attention_outputs_for_context_input_2")
            context_weights_matrices = []
            for i, context_input in enumerate(context_inputs):
                context_input = tf.reshape(context_input * tf.expand_dims(mask_context_input, axis=-1), [tf.shape(self.his_item_emb)[0], (item_his_users_maxlen + 1) * EMBEDDING_DIM * 2])  # [128,36],[128,50*36]->[128,1836]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)  # [128,1836]->[128,81]
                context_weights_matrices.append(context_weights_matrix)  # 将结果添加到列表中
            context_weights_matrix = tf.concat(context_weights_matrices, axis=-1)  # [[128,81],[128,81],[128,81],[128,81]]->[128,324]  循环结束后，使用 tf.concat 将所有矩阵按最后一维拼接
            att_mask_k_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,30,1]->[128,30]
            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2_no_weight(item_bh_k_input, context_weights_matrix, num_heads=4, num_units=num_units, padding_mask=att_mask_k_input, causality_mask_bool=False, dropout_rate=0., is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_k_his_users_eb, self.item_bh_k_time_emb], -1)  # [128,30,2],[128,30,18]->[128,30,20]

            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,30,36],[128,30,36],[128,30,36],[128,30,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                target_item_his_att_users_output_sub = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub

                # target_item_his_att_users_output_sub = layer_norm(target_item_his_att_users_output_sub, name="target_item_his_att_users_output_sub_layernorm" + str(i))

                with tf.name_scope('Attention_layer_for_users' + str(i)):
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, target_item_his_att_users_output_sub, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_his_items_representation'):
            item_his_users_item_truncated_len = 5  # 最大数目为20，目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,30,20]-> [128,30]目标物品每个历史用户行为的长度
            item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,30]->[128*30]得到[128*30,5,36],[128*30],[128*30,5,1]
            item_user_his_k_tivs_emb, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_tivs_emb, [-1, tf.shape(self.item_user_his_tivs_emb)[2], EMBEDDING_DIM]), seq_len=tf.reshape(item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,18]->[128*30,20,18],[128,30]->[128*30]得到[128*30,5,18],[128*30],[128*30,5,1]
            # 这里是目标物品的历史用户购买历史物品的位置向量
            self.position_targetitem_user_items = tf.range(item_his_users_item_truncated_len)
            self.position_user_items_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_targetitem_user_items)  # [5,2]
            self.position_user_items_eb = tf.tile(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, 1])  # [128*30*5,2]
            self.position_user_items_eb = tf.reshape(self.position_user_items_eb, [tf.shape(self.his_item_emb)[0] * item_his_users_maxlen, item_his_users_item_truncated_len, position_embedding_size])  # [128*30,5,2]

            # 这里是目标物品的历史用户的位置向量
            self.position_k_his_users_eb_expand = tf.tile(tf.expand_dims(self.position_k_his_users_eb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,2]->[128,30,5,2]
            self.position_k_his_users_eb_expand = tf.reshape(self.position_k_his_users_eb_expand, tf.shape(self.position_user_items_eb))  # [128*30,5,2]
            # 这里是目标物品的历史用户的时间间隔向量
            self.item_bh_k_time_emb_expand = tf.tile(tf.expand_dims(self.item_bh_k_time_emb, axis=-2), [1, 1, item_his_users_item_truncated_len, 1])  # [128,30,18]->[128,30,5,18]
            self.item_bh_k_time_emb_expand = tf.reshape(self.item_bh_k_time_emb_expand, [tf.shape(self.position_user_items_eb)[0], tf.shape(self.position_user_items_eb)[1], EMBEDDING_DIM])  # [128*30,5,18]

            item_user_his_k_items_emb_with_tiv = tf.layers.dense(tf.concat([self.position_k_his_users_eb_expand, self.item_bh_k_time_emb_expand, self.position_user_items_eb, item_user_his_k_tivs_emb, item_user_his_k_items_emb], axis=-1), INC_DIM) * self.item_user_seq_truncated_mask  # [128*30,5,56]->[128*30,5,36]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                item_user_his_attention_output = self_multi_head_attn(inputs=item_user_his_k_items_emb_with_tiv,  # [128*30,5,36]
                                                                      padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]), causality_mask_bool=False,  # [128*30,5,1]->[128*30,5]
                                                                      num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")
                mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")  # [128*30,5,36]
                mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM, name="self_multi_head_attn_dense2")  # [128*30,5,36]
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*30,5,36]
            # item_user_his_attention_output = layer_norm(item_user_his_attention_output,name="item_user_his_attention_output_layernorm")  # [128,20,36]->[128,20,36]
            count_user_items_truncated = tf.reduce_sum(self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,1]->[128*30,1]
            item_user_his_items_avg = tf.reduce_sum(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated))  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            # item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*30,5,36],[128*30,5,1]->[128*30,5,36]->[128*30,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM])  # [128*30,36]->[128,30,36]

            # --------------------------------------------------------- 得到当前用户的短期兴趣表征 -----------------------------------------------------------
            current_user_k_bh_emb, current_user_bh_seq_len, current_user_bh_mask = mapping_to_k(his_item_with_tiv_emb, self.seq_len_ph, k=item_his_users_item_truncated_len)  # [128,20,36],[128],5-> [128,5,36],[128],[128,5,1]
            with tf.variable_scope('self_multi_head_attn_dense_layer', reuse=tf.AUTO_REUSE):
                current_user_interest_emb = self_multi_head_attn(inputs=current_user_k_bh_emb,  # [128,5,36]
                                                                 padding_mask=tf.reshape(current_user_bh_mask, [-1, tf.shape(current_user_bh_mask)[1]]), causality_mask_bool=False,  # [128,5,1]->[128,5]
                                                                 num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_attention_output")  # [128,5,36]
                mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu, name="self_multi_head_attn_dense1")
                mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2, name="self_multi_head_attn_dense2")
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,5,36]
            current_user_interest_emb = tf.reduce_sum(current_user_interest_emb * current_user_bh_mask, axis=1) / (tf.reduce_sum(current_user_bh_mask, axis=1))  # [128,5,36]*[128,5,1]->[128,5,36]->[128,36]

            item_user_his_items_avg_ex = tf.reshape(tf.reduce_sum(item_user_his_k_items_emb * self.item_user_seq_truncated_mask, axis=1) / (tf.where(tf.equal(count_user_items_truncated, tf.zeros_like(count_user_items_truncated)), tf.ones_like(count_user_items_truncated), count_user_items_truncated)), [tf.shape(current_user_interest_emb)[0], item_his_users_maxlen, INC_DIM])  # [128*30,5,36],[128*30,5,1]->[128*30,36]->[128,30,36]
            count_user_truncated = tf.reduce_sum(self.item_bh_mask_t, axis=1)  # [128,30,1]->[128,1]
            current_user_interest_emb_prevent_noisy = current_user_interest_emb + tf.reduce_sum(item_user_his_items_avg_ex * self.item_bh_mask_t, axis=1) / (tf.where(tf.equal(count_user_truncated, tf.zeros_like(count_user_truncated)), tf.ones_like(count_user_truncated), count_user_truncated))  # [128,36]+([128,30,36],[128,30,1]->[128,36])
            current_user_interest_emb_ex = tf.layers.dense(tf.concat([current_user_interest_emb_prevent_noisy, self.target_item_emb], axis=-1), INC_DIM)
            attention_output, _, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb_ex, item_user_his_items_avg, context_embedding_for_user, att_mask_k_input, stag="Attention_layer_for_target_item_users")
            short_group_interest = tf.reduce_sum(attention_output, 1)
            attention_scores = tf.reduce_sum(tf.reduce_sum(attention_scores_no_softmax, 1), 1, keepdims=True)  # [128,1,20]->[128,1]
        # addition = tf.layers.dense(tf.concat([self.user_embedded,self.target_item_emb,self.user_embedded*self.target_item_emb,self.user_embedded-self.target_item_emb],axis=-1),2*INC_DIM,activation=tf.nn.relu)
        # addition = tf.layers.dense(addition,INC_DIM,activation=tf.nn.relu)
        # addition = tf.layers.dense(addition,INC_DIM,activation=None)

        # addition1 = tf.layers.dense(context_weights_matrix, 4 * INC_DIM, activation=tf.nn.relu)
        # addition1 = tf.layers.dense(addition1, 2 * INC_DIM, activation=tf.nn.relu)
        # addition1 = tf.layers.dense(addition1, INC_DIM, activation=None)

        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded, short_group_interest, attention_scores], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel2_test_old(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel2_test_old, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        # ----------------------------------------------------- user module -----------------------------------------------------
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        with tf.name_scope('DMIN_user_module'):
            his_items_maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his_item = tf.range(his_items_maxlen)
            self.his_item_position_embeddings_var = tf.get_variable("position_embeddings_var", [his_items_maxlen, position_embedding_size])  # [20,2]
            self.position_his_items_eb = tf.nn.embedding_lookup(self.his_item_position_embeddings_var, self.position_his_item)  # [20,2]
            self.position_his_items_eb = tf.tile(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
            self.position_his_items_eb = tf.reshape(self.position_his_items_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_items_eb.get_shape().as_list()[-1]])  # [128,20,2]

            self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
            self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
                                                                              padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
                                                                              num_units=INC_DIM, num_heads=4, dropout_rate=0., is_training=self.is_training, name="his_item_user_attention_output")

            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
            self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]

            self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]
            self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]

            with tf.name_scope("multi_head_attention_1"):
                self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg, self.position_his_items_eb, self.his_items_tiv_emb], axis=-1), INC_DIM)
                his_item_with_tiv_emb = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb, self.his_item_emb], axis=-1) * tf.expand_dims(self.mask, axis=-1)  # [128,20,2],[128,20,18],[128,20,36]->[128,20,56]
                his_item_with_tiv_emb = tf.layers.dense(his_item_with_tiv_emb, EMBEDDING_DIM * 2)
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0.2, is_training=self.is_training, name="multihead_attention_outputs_for_items")
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru", focal_loss_bool=False)
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_for_items"):
                weight_matrix_size = 324  # 18*18=324
                context_input = tf.concat([tf.reshape(his_item_with_tiv_emb, [tf.shape(his_item_with_tiv_emb)[0], 20 * 36])], axis=-1)  # [128,56].[128,20],[128,20*56]->[128,1120]
                context_weights_matrix = get_context_matrix(context_input, embed_dims=[weight_matrix_size], dropout_rate=0., is_training=self.is_training)
                context_embedding = tf.concat([self.position_his_items_eb, self.his_items_tiv_emb], -1)  # [128,20,2],[128,20,18]->[128,20,20]
                self.target_item_user_emb = tf.concat([self.uid_emb, self.target_item_emb], -1)
                multihead_attention_outputss = context_aware_multi_head_self_attention_v2(self.his_item_emb_dynamic, context_weights_matrix, num_heads=4, num_units=9, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0.2, is_training=self.is_training, name="multihead_attention_outputss")
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope("Attention_layer_for_items" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_user_emb, multihead_attention_outputs_v2, context_embedding, self.mask, stag="Attention_layer_for_items" + str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        # ----------------------------------------------------- item module -----------------------------------------------------
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        item_his_users_maxlen = 50
        position_embedding_size = 2
        # 这里是可学习的position-embedding
        self.position_his_users = tf.range(item_his_users_maxlen)
        self.item_his_position_embeddings_var = tf.get_variable("item_his_position_embeddings_var", [item_his_users_maxlen, position_embedding_size])  # [20,2]
        self.position_his_users_eb = tf.nn.embedding_lookup(self.item_his_position_embeddings_var, self.position_his_users)  # [20,2]
        self.position_his_users_eb = tf.tile(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,2]
        self.position_his_users_eb = tf.reshape(self.position_his_users_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_users_eb.get_shape().as_list()[-1]])  # [128,20,2]

        self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留更长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],item_his_users_maxlen-> [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,18],[128],item_his_users_maxlen-> [128,50,18],[128],[128,50,1]
        # ----------------------------------------------------- 处理目标物品users的表示 -----------------------------------------------------
        with tf.name_scope('target_item_his_users_representation'):
            weight_matrix_size = 324
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50,1]->[128,50]
            item_bh_drink_trm_input = tf.concat([self.position_his_users_eb, self.item_bh_time_embeeded, self.item_bh_k_user_emb], axis=-1) * tf.expand_dims(self.item_user_his_mask, axis=-1)  # [128,50,2],[128,50,18],[128,50,18]->[128,50,38]
            context_input = tf.concat([tf.reshape(item_bh_drink_trm_input, [-1, item_his_users_maxlen * 38])], axis=-1)  # [128,18],[128,36],[128,38],[128,50],[128,50*38]->[128,2042]
            context_weights_matrix = get_context_matrix(context_input, [weight_matrix_size], is_training=self.is_training)
            item_bh_drink_trm_input = tf.layers.dense(item_bh_drink_trm_input, EMBEDDING_DIM * 2)  # [128,50,38]->[128,50,36]

            item_bh_drink_trm_input = self_multi_head_attn(item_bh_drink_trm_input, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=att_mask_input, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="item_bh_drink_trm_input")
            item_bh_drink_trm_input1 = tf.layers.dense(item_bh_drink_trm_input, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            item_bh_drink_trm_input1 = tf.layers.dense(item_bh_drink_trm_input1, EMBEDDING_DIM * 2)
            item_bh_drink_trm_input = item_bh_drink_trm_input1 + item_bh_drink_trm_input  # [128,50,36]

            target_item_his_att_users_output = context_aware_multi_head_self_attention_v2(item_bh_drink_trm_input, context_weights_matrix, num_heads=4, num_units=9, padding_mask=att_mask_input, causality_mask_bool=False, dropout_rate=0.2, is_training=self.is_training, name="target_item_his_att_users_output")
            context_embedding_for_user = tf.concat([self.position_his_users_eb, self.item_bh_time_embeeded], -1)  # [128,50,2],[128,50,18]->[128,50,20]
            for i, target_item_his_att_users_output_sub in enumerate(target_item_his_att_users_output):  # [[128,50,36],[128,50,36],[128,50,36],[128,50,36],]
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                target_item_his_att_users_output_sub1 = tf.layers.dense(target_item_his_att_users_output_sub1, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = target_item_his_att_users_output_sub1 + target_item_his_att_users_output_sub
                with tf.name_scope('Attention_layer_for_users' + str(i)):  # 每次for循环都会创建新的din_attention_with_context，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.user_embedded, multihead_attention_outputs_v2, context_embedding_for_user, self.item_user_his_mask, stag="Attention_layer_for_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea, att_fea * self.user_embedded], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]

        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            item_his_users_item_truncated_len = 5  # 目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            self.item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            self.item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(self.item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,50]->[128*50]得到[128*50,k,36],[128*50],[128*50,k,1]
            item_user_his_attention_output = self_multi_head_attn(inputs=self.item_user_his_k_items_emb,  # [128*50,k,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*50,k,1]->[128*50,k]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*50,k,36]
            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*50,k,36],[128*50,k,1]->[128*50,k,36]->[128*50,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM]) + tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM)  # [128*50,36]->[128,50,36]
            # --------------------------------------------------------- 得到当前用户的表征 -----------------------------------------------------------
            current_user_interest_emb = self_multi_head_attn(inputs=self.his_item_emb, name="current_user_interest_emb", padding_mask=self.mask, num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training)  # [128,20,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,20,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * tf.expand_dims(self.mask, -1), axis=1)  # [128,20,36]*[128,20,1]->[128,20,36]->[128,36]

            target_item_his_user_his_items_avg = self_multi_head_attn_v2(item_user_his_items_avg, num_units=INC_DIM, num_heads=4, padding_mask=att_mask_input,
                                                                         causality_mask_bool=False, dropout_rate=0, is_training=self.is_training, name="target_item_his_user_his_items_avg")
            for i, multihead_attention_outputs_v2 in enumerate(target_item_his_user_his_items_avg):
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope("Attention_layer_for_target_item_users" + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(current_user_interest_emb, multihead_attention_outputs_v2, None, att_mask_input, stag="Attention_layer_for_target_item_users" + str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
                    inp = tf.concat([inp, att_fea, att_fea * current_user_interest_emb], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        inp = tf.concat([inp, current_user_interest_emb * self.target_item_emb, self.item_user_his_eb_sum, self.item_user_his_eb_sum * self.user_embedded], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class myModel_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(myModel_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        item_his_users_maxlen = 30
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # ----------------------------------------------------- DMIN -----------------------------------------------------
        with tf.name_scope('DMIN_user_module'):
            maxlen = 20
            position_embedding_size = 2
            # 这里是可学习的position-embedding
            self.position_his = tf.range(maxlen)
            self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
            self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # [20,36]
            self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # [128*20,36]
            self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # [128,20,36]
            self.context_embedding = tf.concat([self.position_his_eb, self.his_items_tiv_emb], -1)  # [128,20,18],[128,20,18]->[128,20,20]

            self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
            self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
                                                                              padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
                                                                              num_units=INC_DIM, num_heads=4, dropout_rate=0.5, is_training=self.is_training, name="his_item_user_attention_output")

            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
            self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
            # ADD但是没有layer_normal
            # self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]
            self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]

            self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]

            self.his_item_emb_dynamic = tf.layers.dense(tf.concat([self.his_item_emb, self.his_item_user_avg], axis=-1), INC_DIM)

            with tf.name_scope("multi_head_attention_1"):
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb_dynamic, name="multihead_attention_outputs", num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                self.target_item_user_emb = tf.concat([self.uid_emb, self.target_item_emb], -1)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_context(self.target_item_user_emb, multihead_attention_outputs_v2, self.context_embedding, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)  # [128,18],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36],[128,36]->[128,260]
        # ----------------------------------------------------- 建模历史物品的历史购买用户-----------------------------------------------------
        # with tf.name_scope('his_items_his_users'):
        # self.his_item_user_attention_output = din_attention(query=tf.tile(self.uid_emb, [1, tf.shape(self.his_item_his_users_emb)[1] * tf.shape(self.his_item_his_users_emb)[2]]),  # [128,18]-> [128,20*10*18]
        #                                                     facts=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], EMBEDDING_DIM]),  # [128*20,10,18]
        #                                                     mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
        #                                                     need_tile=False, stag="DIN_U2U_attention")  # 返回的是[128*20,1,18]
        # self.his_item_user_attention_output = tf.reshape(tf.reduce_sum(self.his_item_user_attention_output, 1), [-1, tf.shape(self.mask)[-1], 18])  # [128*20,1,18]->[128*20,18]->[128,20,18]
        # item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(self.mask, axis=2), tf.expand_dims(self.mask, axis=1)), dtype=tf.int32)  # [128,20,1],[128,1,20]->[128,20,20]
        # self.his_item_user_transformer_output = transformer_model(self.his_item_user_attention_output, hidden_size=EMBEDDING_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='his_item_user_transformer_output', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
        # self.his_item_user_transformer_output_sum = tf.layers.dense(tf.reshape(self.his_item_user_transformer_output, [tf.shape(self.his_item_user_transformer_output)[0], 360]), EMBEDDING_DIM * 2, name='his_item_user_transformer_output_sum')  # [128,20,18]

        # self.his_item_his_users_emb = tf.layers.dense(tf.concat([self.his_item_his_users_emb, self.his_item_his_users_tiv_emb], axis=-1), INC_DIM)  # [128,50,20,18]->[128,50,20,36]
        # self.his_item_user_attention_output_origin = self_multi_head_attn(inputs=tf.reshape(self.his_item_his_users_emb, [-1, tf.shape(self.his_item_his_users_emb)[2], INC_DIM]),  # [128,20,10,36]->[128*20,10,36]
        #                                                                   padding_mask=tf.reshape(self.his_item_user_mask, [-1, tf.shape(self.his_item_user_mask)[2]]),  # [128*20,10]
        #                                                                   num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="his_item_user_attention_output")
        #
        # self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output_origin, EMBEDDING_DIM * 4, activation=tf.nn.relu)  # [128*20,10,36]
        # self.his_item_user_attention_output = tf.layers.dense(self.his_item_user_attention_output, EMBEDDING_DIM * 2)  # [128*20,10,36]
        # # ADD但是没有layer_normal
        # self.his_item_user_attention_output = self.his_item_user_attention_output_origin + self.his_item_user_attention_output  # [128*20,10,36]
        # self.his_item_user_attention_output = tf.reshape(self.his_item_user_attention_output, tf.shape(self.his_item_his_users_emb))  # [128*20,10,36]->[128,20,10,36]
        #
        # self.his_item_user_avg = tf.reduce_mean(self.his_item_user_attention_output * tf.expand_dims(self.his_item_user_mask, -1), axis=2)  # [128,20,10,36],[128,20,10,1]->[128,20,36]
        # # self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
        # self.his_item_user_att_sum, _ = attention_net_v1(enc=self.his_item_user_avg, sl=self.seq_len_ph,  # enc=encoder=keys [128,20,36]
        #                                                  dec=self.user_embedded,  # dec=decoder=query[128,36]
        #                                                  num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
        #                                                  is_training=self.is_training, reuse=False, scope='his_item_user_att_sum')  # 返回值是[128,36]
        # -------------------------------------------------------------------------------- item module -------------------------------------------------------------------------------

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)  # [128,50,18],[128,50]-> [128,50,18],[128,50,1]-> [128,18]
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')  # [128,18]-> [128,36]

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=item_his_users_maxlen)  # 得到[128,50,18]

        self.item_user_his_items_emb_truncated, _, _ = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1]), seq_len=self.seq_len_u_ph, k=item_his_users_maxlen)  # [128,50,20,36]->[128,50,20*36],[128,50]->[128*50]得到[128,30,20*36],[128],[128,30,1]
        self.item_user_his_items_emb = tf.reshape(self.item_user_his_items_emb_truncated, [tf.shape(self.item_user_his_items_emb)[0], item_his_users_maxlen, -1, INC_DIM])  # [128,30,20*36]->[128,30,20,36]

        with tf.name_scope('target_item_his_users_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]->[128,50,1]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # ----------------------------------------------------- 处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')  # 将当前用户的维度变成INC_DIM
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_user_his_eb_att')

        with tf.name_scope('target_item_his_users_his_items_representation'):
            # ----------------------------------------------------- 处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            #  得到目标物品的每个历史购买用户的表征
            # self.item_user_his_items_emb[128,50,20,36]
            item_his_users_item_truncated_len = 10  # 目标物品的历史用户的历史行为截断到多少，代表近期的购买兴趣
            self.item_his_users_items_len = tf.count_nonzero(self.item_user_his_mid_batch_ph, axis=-1)  # [128,50,20]-> [128,50]目标物品每个历史用户行为的长度
            self.item_user_his_k_items_emb, self.item_user_seq_truncated_lens, self.item_user_seq_truncated_mask = mapping_to_k(sequence=tf.reshape(self.item_user_his_items_emb, [-1, tf.shape(self.item_user_his_items_emb)[2], INC_DIM]), seq_len=tf.reshape(self.item_his_users_items_len[:, 50 - item_his_users_maxlen:], [-1]), k=item_his_users_item_truncated_len)  # [128,30,20,36]->[128*30,20,36],[128,50]->[128*50]得到[128*50,k,36],[128*50],[128*50,k,1]
            # self.item_user_his_k_items_emb = tf.reshape(self.item_user_his_k_items_emb, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1, INC_DIM])  # [128*50,k,36] -> [128,50,k,36]
            # self.item_user_seq_truncated_lens = tf.reshape(self.item_user_seq_truncated_lens, tf.shape(self.item_his_users_items_len))  # [128*50] -> [128,50]
            # self.item_user_seq_truncated_mask = tf.reshape(self.item_user_seq_truncated_mask, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], -1, 1])  # [128*50,k,1]->[128,50,k,1]
            print("self.item_user_his_k_items_emb", self.item_user_his_k_items_emb.get_shape())
            item_user_his_attention_output = self_multi_head_attn(inputs=self.item_user_his_k_items_emb,  # [128*50,k,36]
                                                                  padding_mask=tf.reshape(self.item_user_seq_truncated_mask, [-1, tf.shape(self.item_user_seq_truncated_mask)[1]]),  # [128*50,k,1]->[128*50,k]
                                                                  num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training, name="item_user_his_attention_output")
            print("self.item_user_his_attention_output", item_user_his_attention_output.get_shape())
            mul_att_item_user_his_attention_output = tf.layers.dense(item_user_his_attention_output, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_item_user_his_attention_output = tf.layers.dense(mul_att_item_user_his_attention_output, INC_DIM)
            # ADD但是没有layer_normal
            item_user_his_attention_output = mul_att_item_user_his_attention_output + item_user_his_attention_output  # [128*50,k,36]
            item_user_his_items_avg = tf.reduce_mean(item_user_his_attention_output * self.item_user_seq_truncated_mask, axis=1)  # [128*50,k,36],[128*50,k,1]->[128*50,k,36]->[128*50,36]
            item_user_his_items_avg = tf.reshape(item_user_his_items_avg, [tf.shape(self.item_user_his_items_emb)[0], tf.shape(self.item_user_his_items_emb)[1], INC_DIM]) + self.item_bh_time_emb_padding  # [128*50,36]->[128,50,36]
            # --------------------------------------------------------- 得到目标物品的每个历史购买用户的表征所形成的群体兴趣 -----------------------------------------------------------
            # hidden_size = 64
            # num_interests = 10
            # num_heads = num_interests
            #
            # target_item_his_users_item_hidden = tf.layers.dense(item_user_his_items_avg, hidden_size * 4, activation=tf.nn.tanh, name='users_interest_1')  # [128,50,36]->[128,50,256]
            # item_att_w = tf.layers.dense(target_item_his_users_item_hidden, num_heads, activation=None, name='users_interest_2', reuse=tf.AUTO_REUSE)  # [128,50,256]->[128,50,10]
            # item_att_w = tf.transpose(item_att_w, [0, 2, 1])  # [128,50,10]->[128,10,50]
            # atten_mask = tf.tile(tf.expand_dims(self.item_user_his_mask, axis=1), [1, num_heads, 1])  # [128,50]->[128,1,50]->[128,10,50]
            # paddings = tf.ones_like(atten_mask) * (-2 ** 32 + 1)  # [128,10,50]
            # item_att_w = tf.where(tf.equal(atten_mask, 0), paddings, item_att_w)
            # item_att_w = tf.nn.softmax(item_att_w)  # [128,10,50]->[128,10,50]
            # target_item_his_users_interest_emb = tf.matmul(item_att_w, item_user_his_items_avg)  # [128,10,50]*[128,50,36]->[128,10,36]

            # --------------------------------------------------------- 得到当前用户的表征 -----------------------------------------------------------
            current_user_interest_emb = self_multi_head_attn(inputs=self.his_item_emb, name="current_user_interest_emb", padding_mask=self.mask, num_units=INC_DIM, num_heads=4, dropout_rate=0, is_training=self.is_training)  # [128,20,36]
            mul_att_current_user_output = tf.layers.dense(current_user_interest_emb, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            mul_att_current_user_output = tf.layers.dense(mul_att_current_user_output, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal
            current_user_interest_emb = mul_att_current_user_output + current_user_interest_emb  # [128,20,36]
            current_user_interest_emb = tf.reduce_mean(current_user_interest_emb * tf.expand_dims(self.mask, -1), axis=1)  # [128,20,36]*[128,20,1]->[128,20,36]->[128,36]

            interest_emb, _ = attention_net_v1(enc=item_user_his_items_avg, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,4,36]这里不包含cls的embedding,sl是keys的真实长度（截断后），
                                               dec=current_user_interest_emb,  # dec=decoder=query[128,36]
                                               num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                               is_training=self.is_training, reuse=False, scope='interest_emb')

        # inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, interest_emb], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        # interest_emb,self.his_item_user_att_sum, self.his_item_user_att_sum * item_user_his_eb_att_sum, self.his_item_user_att_sum * self.item_user_his_eb_sum,
        inp = tf.concat([inp, interest_emb, current_user_interest_emb * self.target_item_emb, item_user_his_eb_att_sum, self.item_user_his_eb_sum], 1)  # [128,260],[128,36],[128,36],[128,36]->[128,368]
        self.build_fcn_net(inp)


class Model_IUI(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_IUI, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        local_num = 1
        INC_DIM = EMBEDDING_DIM * 2
        with tf.name_scope("multi_head_attention_IUI"):
            global_multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask_bool=False, dropout_rate=0, is_training=self.is_training)
            global_multihead_attention_outputs1 = tf.layers.dense(global_multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            global_multihead_attention_outputs1 = tf.layers.dense(global_multihead_attention_outputs1, EMBEDDING_DIM * 2)
            global_multihead_attention_outputs = global_multihead_attention_outputs1 + global_multihead_attention_outputs
        G_global = global_multihead_attention_outputs * tf.expand_dims(self.mask, -1)  # [128,20,36]
        hat_e_global = IUI_attention(tf.expand_dims(self.target_item_emb, axis=1), G_global, self.mask, name="hat_e_global_IUI_attention")  # mask:[128,20]
        hat_e_global = tf.reduce_sum(hat_e_global, axis=1)  # [128,36]
        G_global = tf.transpose(G_global, perm=[0, 2, 1])  # [128,20,36] -> [128,36,20]

        self.local_emb, self.local_seq_len, self.local_mask = mapping_to_k(self.his_item_emb, self.seq_len_ph, k=local_num)
        e_n, _, _ = mapping_to_k(self.his_item_emb, self.seq_len_ph, k=1)
        e_n = tf.transpose(e_n, perm=[0, 2, 1])  # [128,1,36]->[128,36,1]
        hat_L_int = IUI_attention(self.local_emb, global_multihead_attention_outputs, self.mask, name="hat_L_int_IUI_attention") * self.local_mask  # [128,3,36],[128,20,36],[128,20]->[128,3,36]
        L_local = tf.layers.dense(hat_L_int, EMBEDDING_DIM * 4, activation=tf.nn.relu)
        L_local = tf.layers.dense(L_local, EMBEDDING_DIM * 2, activation=tf.nn.relu)  # [128,3,36]

        hat_e_local = IUI_attention(tf.expand_dims(self.target_item_emb, axis=1), L_local, tf.squeeze(self.local_mask, axis=-1), name="hat_e_local_IUI_attention")  # mask:[128,3,1]
        hat_e_local = tf.reduce_sum(hat_e_local, axis=1)  # [128,36]

        L_local = tf.transpose(L_local, perm=[0, 2, 1])  # [128,3,36] -> [128,36,3]

        W3 = tf.get_variable("W3", [INC_DIM, INC_DIM])
        M_A = tf.nn.tanh(tf.matmul(tf.matmul(G_global, W3, transpose_a=True), L_local))  # [128,36,20],[36,36],[128,36,3]->[128,20,36],[36,36],[128,36,3]->[20,3]

        K = EMBEDDING_DIM
        W4 = tf.get_variable("W4", [K, INC_DIM])
        W5 = tf.get_variable("W5", [K, INC_DIM])
        W6 = tf.get_variable("W6", [K, INC_DIM])
        W7 = tf.get_variable("W7", [2 * INC_DIM, INC_DIM])
        H_g = tf.nn.tanh(tf.matmul(W4, G_global) + tf.matmul(tf.matmul(W5, L_local) + tf.matmul(W6, e_n), M_A, transpose_b=True))  # [128,18,20]: [18,36],[128,36,20]->[128,18,20];([18,36],[128,36,3]->[128,18,3])+([18,36],[128,36,1]->[128,18,1])->[128,18,3]*[3,20]->[128,18,20]
        w_g = tf.get_variable("w_g", [1, K])  # [1,18]
        alpha_g = tf.nn.softmax(tf.matmul(w_g, H_g))  # [1,18],[128,18,20]->[128,1,20]
        bar_e_global = tf.reduce_sum(tf.matmul(alpha_g, G_global, transpose_b=True), axis=1)  # [128,1,20],[128,36,20]->[128,1,36]->[128,36]
        e_global = tf.matmul(tf.concat([self.target_item_emb, bar_e_global], axis=-1), W7)  # [128,36],[128,36]->[128,72],[72,36]->[128,36]

        W8 = tf.get_variable("W8", [2 * INC_DIM, INC_DIM])
        H_l = tf.nn.tanh(tf.matmul(W5, L_local) + tf.matmul(W6, e_n) + tf.matmul(tf.matmul(W4, G_global), M_A))  # [128,18,3]:([18,36],[128,36,3]->[128,18,3])+([18,36],[128,36,1]->[128,18,1])->[128,18,3];[18,36],[128,36,20]->[128,18,20]*[20,3]->[128,18,3]

        w_l = tf.get_variable("w_l", [1, K])  # [1,18]
        alpha_l = tf.nn.softmax(tf.matmul(w_l, H_l))  # [1,18],[128,18,3]->[128,1,3]
        bar_e_local = tf.reduce_sum(tf.matmul(alpha_l, L_local, transpose_b=True), axis=1)  # [128,1,3],[128,36,3]->[128,1,36]->[128,36]
        e_local = tf.matmul(tf.concat([self.target_item_emb, bar_e_local], axis=-1), W8)  # [128,36],[128,36]->[128,72],[72,36]->[128,36]

        W9 = tf.get_variable("W9", [INC_DIM, INC_DIM])
        W10 = tf.get_variable("W10", [INC_DIM, INC_DIM])
        alph = tf.nn.sigmoid(tf.matmul(hat_e_global, W9) + tf.matmul(hat_e_local, W10))  # [128,36]
        hat_e_ideal = alph * hat_e_global + (1 - alph) * hat_e_local
        inp = tf.concat([hat_e_ideal, e_global, e_local, self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)
