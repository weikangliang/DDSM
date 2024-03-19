# coding=utf-8
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from rnn import dynamic_rnn
from utils import *
from Dice import dice
from transformer import transformer_model, gelu
import numpy as np

TIME_INTERVAL = 16  # gap = np.array([1.1, 1.4, 1.7, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])  # 共有16个取值范围
ITEM_BH_CLS_CNT = 3  # cls采用的是3个


class Model(object):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        with tf.name_scope('Inputs'):
            self.use_negsampling = use_negsampling
            self.EMBEDDING_DIM = EMBEDDING_DIM
            self.is_training = tf.placeholder(tf.bool)  # 为使用Dropout,batch_normalization等

            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')  # 1
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')  # 2
            self.cat_batch_ph = tf.placeholder(tf.int32, [None, ], name='cat_batch_ph')  # 3
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')  # 4
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cat_his_batch_ph')  # 5

            self.item_user_his_batch_ph = tf.placeholder(tf.int32, [None, 50], name='item_user_his_batch_ph')  # 1
            self.item_user_his_time_ph = tf.placeholder(tf.int32, [None, None], name='item_user_his_time_ph')  # 2
            self.item_user_his_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_mid_batch_ph')  # 3 [128,50,20]
            self.item_user_his_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_cat_batch_ph')  # 4

            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')  # 当前用户的历史行为的mask，从后面填充的，被padding的部分mask值为0
            self.item_user_his_mask = tf.placeholder(tf.float32, [None, None], name='mask')  # 目标物品的历史购买用户的mask
            self.item_user_his_mid_mask = tf.placeholder(tf.float32, [None, None, None], name='mask')  # [128,50,20]目标物品的历史购买用户的历史行为的mask

            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')  # 历史行为的长度
            self.seq_len_u_ph = tf.placeholder(tf.int32, [None], name='seq_len_u_ph')  # 目标物品历史用户的长度
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram('uid_embeddings_var', self.uid_embeddings_var)
            self.uid_emb = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
            self.item_user_his_uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.item_user_his_batch_ph)
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram('mid_embeddings_var', self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            self.item_user_his_mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.item_user_his_mid_batch_ph)  # [128,50,20,18]
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram('cat_embeddings_var', self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            self.item_user_his_cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.item_user_his_cat_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

            self.time_embeddings_var = tf.get_variable("time_embedding_var", [TIME_INTERVAL, EMBEDDING_DIM])
            tf.summary.histogram('time_embedding_var', self.time_embeddings_var)
            self.item_bh_time_embeeded = tf.nn.embedding_lookup(self.time_embeddings_var, self.item_user_his_time_ph)

            self.item_bh_cls_embedding = tf.get_variable("item_cls_embedding", [ITEM_BH_CLS_CNT, EMBEDDING_DIM * 2])

        self.target_item_emb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.his_item_emb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_user_his_eb = tf.concat([self.item_user_his_mid_batch_embedded, self.item_user_his_cat_batch_embedded], -1)  # [128,50,20,18],[128,50,20,18]->[128,50,20,36]
        self.item_his_eb_sum = tf.reduce_sum(self.his_item_emb * tf.expand_dims(self.mask, -1), 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat([self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1)  # 0 means only using the first negative item ID. 3 item IDs are inputed in the line 24.
            self.noclk_item_his_eb = tf.reshape(self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], 36])  # cat embedding 18 concate item embedding 18.
            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

        # self.all_interests = tf.get_variable("all_trends", [args.all_candidate_trends, args.item_emb_dim + args.cat_emb_dim + args.tiv_emb_dim + args.position_emb_dim])  # (100, 44)

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
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
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        return y_hat

    def train(self, sess, inps):
        if self.use_negsampling:
            loss, accuracy, aux_loss, _ = sess.run([self.loss, self.accuracy, self.aux_loss, self.optimizer], feed_dict={
                self.uid_batch_ph: inps[0],
                self.mid_batch_ph: inps[1],
                self.cat_batch_ph: inps[2],
                self.mid_his_batch_ph: inps[3],
                self.cat_his_batch_ph: inps[4],
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.lr: inps[15],
                self.noclk_mid_batch_ph: inps[16],
                self.noclk_cat_batch_ph: inps[17],
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
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.lr: inps[15],
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
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
                self.noclk_mid_batch_ph: inps[15],
                self.noclk_cat_batch_ph: inps[16],
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
                self.mask: inps[5],
                self.item_user_his_batch_ph: inps[6],
                self.item_user_his_mask: inps[7],
                self.item_user_his_time_ph: inps[8],
                self.item_user_his_mid_batch_ph: inps[9],
                self.item_user_his_cat_batch_ph: inps[10],
                self.item_user_his_mid_mask: inps[11],
                self.target_ph: inps[12],
                self.seq_len_ph: inps[13],
                self.seq_len_u_ph: inps[14],
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
        d_layer_wide = tf.concat([tf.concat([self.target_item_emb, self.item_his_eb_sum], axis=-1),
                                  self.target_item_emb * self.item_his_eb_sum], axis=-1)
        d_layer_wide = tf.layers.dense(d_layer_wide, 2, activation=None, name='f_fm')
        self.y_hat = tf.nn.softmax(dnn3 + d_layer_wide)

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            self.loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        self.merged = tf.summary.merge_all()


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE)

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum], 1)
        self.build_fcn_net(inp)


class Model_PNN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_PNN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE,
                                        ATTENTION_SIZE)

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum,
                         self.target_item_emb * self.item_his_eb_sum], 1)

        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DIN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            tf.summary.histogram('att_fea', att_fea)
        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, att_fea], -1)
        self.build_fcn_net(inp)


class DIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(DIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs, att_scores=tf.expand_dims(alphas, -1), sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp)


class DIEN_with_neg(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DIEN_with_neg, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)

        # RNN layer(-s)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        aux_loss_1 = self.auxiliary_loss(rnn_outputs[:, :-1, :], self.his_item_emb[:, 1:, :],
                                         self.noclk_item_his_eb[:, 1:, :],
                                         self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_fcn_attention(self.target_item_emb, rnn_outputs, ATTENTION_SIZE, self.mask, softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores=tf.expand_dims(alphas, -1),
                                                     sequence_length=self.seq_len_ph, dtype=tf.float32,
                                                     scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state2], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_GRU4REC(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_GRU4REC, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        with tf.name_scope('rnn1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb,
                                         sequence_length=self.seq_len_ph, dtype=tf.float32,
                                         scope="gru1")
            _, final_state_1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                           sequence_length=self.seq_len_ph, dtype=tf.float32,
                                           scope="gru2")
        with tf.name_scope('rnn1'):
            item_user_len = tf.reduce_sum(self.item_user_his_mask, axis=-1)
            item_user_rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded,
                                                   sequence_length=item_user_len, dtype=tf.float32,
                                                   scope="gru3")
            _, final_state_2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=item_user_rnn_outputs,
                                           sequence_length=item_user_len, dtype=tf.float32,
                                           scope="gru4")

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, final_state_1, final_state_2], 1)
        # Fully connected layer
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
        neighbors_rep_seq = tf.concat([self.item_user_his_uid_batch_embedded, tf.reduce_sum(self.item_user_his_eb, axis=2)], axis=-1)
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


class Model_DNN_Multi_Head(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # 128,20,2

        with tf.name_scope("multi_head_attention_1"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
            # 下面两行是point-wise feed_forward
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
        aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
        self.aux_loss = aux_loss_1

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention_2"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                # ADD但是没有layer_normal（加上layer_normal会降低三个百分点）
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_TIEN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_TIEN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的
        # near k behavior sum-pooling, the best T=5
        # 将user_embedding抓换成2*Embedding
        self.item_user_his_uid_batch_embedded = tf.layers.dense(self.item_user_his_uid_batch_embedded, INC_DIM, name='item_user_his_uid_batch_embedded_2dim')
        self.item_bh_k, self.item_bh_seq_len_k, self.item_bh_mask_k = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=5)
        self.item_bh_masked_embedded = self.item_bh_k * self.item_bh_mask_k
        self.item_bh_embedded_sum = tf.reduce_sum(self.item_bh_masked_embedded, 1)
        # near SEQ_USER_T behavior
        self.item_bh_t, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)  # [128,50,18]

        # 2.sequential modeling for user/item behaviors
        with tf.name_scope('rnn'):
            gru_outputs_u, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.his_item_emb, sequence_length=self.seq_len_ph, dtype=tf.float32, scope="gru_ub")  # 类似DIEN建模用户行为序列
            gru_outputs_i, _ = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_t, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_ib")  # 建模物品行为序列
        # 3.将时间的embedding变成一样的维度
        with tf.name_scope('time-signal'):
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
        # 4. attention layer
        with tf.name_scope('att'):
            # 对当前用户行为建模的attention，类似DIN
            u_att, _ = attention_net_v1(enc=gru_outputs_u, sl=self.seq_len_ph, dec=self.target_item_emb, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ub', value=gru_outputs_u)
            # 5. prevent noisy
            self.item_bh_masked_embedded_t = self.item_bh_t * self.item_bh_mask_t

            self.item_bh_embedded_sum_t = tf.reduce_sum(self.item_bh_masked_embedded_t, 1)
            dec = tf.layers.dense(self.uid_emb, INC_DIM, name='uid_batch_embedded_2dim') + self.item_bh_embedded_sum_t / tf.reshape(tf.cast(self.item_bh_seq_len_t, tf.float32), [-1, 1]) + 1e5  # [128,36]+[128,36]

            gru_outputs_ib_with_t = gru_outputs_i + self.item_bh_time_emb_padding  # [128,50,36]
            i_att, _ = attention_net_v1(enc=gru_outputs_ib_with_t, sl=self.item_bh_seq_len_t, dec=dec, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=False, reuse=False, scope='ib', value=gru_outputs_i)

        # 5.time-aware representation layer
        with tf.name_scope('time_gru'):
            _, it_state = tf.nn.dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_emb_padding, sequence_length=self.item_bh_seq_len_t, dtype=tf.float32, scope="gru_it")
            i_att_ta = i_att + it_state

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, u_att,
                         self.item_bh_embedded_sum, i_att, i_att_ta], 1)  # 这里是TIEN的贡献
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class Model_DUMN_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DUMN_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        with tf.name_scope('DUMN'):
            # --------------------------------------获得当前用户的表示-----------------------------------------
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            item_user_his_attention_output = din_attention(tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_eb)[1] * tf.shape(self.item_user_his_eb)[2]]),
                                                           tf.reshape(self.item_user_his_eb, [-1, tf.shape(self.item_user_his_eb)[2], 36]),
                                                           tf.reshape(self.item_user_his_mid_mask, [-1, tf.shape(self.item_user_his_mid_mask)[2]]),
                                                           need_tile=False)
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_eb)[1], 36])
            item_user_bhvs_feat = tf.concat([self.item_user_his_uid_batch_embedded, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)

        inp = tf.concat([self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum, user_feat, sim_att, sim_score_sum], -1)
        # Fully connected layer
        self.build_fcn_net(inp)


class Model_DMIN_DUMN(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=False):
        super(Model_DMIN_DUMN, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
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
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有norm（加norm效果不好）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            # aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            # self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有norm（加norm效果不好）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)
        with tf.name_scope('DUMN'):
            # --------------------------------------获得当前用户的表示-----------------------------------------
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # 这里将当前user_id和user兴趣concat起来
            # --------------------------------------获得目标物品历史用户的表示-----------------------------------------
            item_user_his_attention_output = din_attention(tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_eb)[1] * tf.shape(self.item_user_his_eb)[2]]),
                                                           tf.reshape(self.item_user_his_eb, [-1, tf.shape(self.item_user_his_eb)[2], 36]),
                                                           tf.reshape(self.item_user_his_mid_mask, [-1, tf.shape(self.item_user_his_mid_mask)[2]]),
                                                           need_tile=False)
            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_eb)[1], 36])
            item_user_bhvs_feat = tf.concat([self.item_user_his_uid_batch_embedded, item_user_his_att], axis=-1)  # 这里将目标物品的历史user_id和user兴趣concat起来
            # --------------------------------------获得当前用户和目标物品历史用户的相似性-----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)
        inp = tf.concat([inp, user_feat, sim_score_sum, sim_att], -1)
        # Fully connected layer
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
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
                # 下面两行是point-wise feed_forward
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                # ADD但是没有norm（加norm效果不好）
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention_2"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, padding_mask=self.mask, causality_mask=True, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]
                    # 下面两行是point-wise feed_forward，每次for循环都会创建新的dense层，这里每个头不共享参数
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    # ADD但是没有norm（加norm效果不好）
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    with tf.name_scope('Attention_layer' + str(i)):  # 每次for循环都会创建新的din_attention_new，也就是每个头不共享参数
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)
        # -----------------------------------------------------item module -----------------------------------------------------
        with tf.name_scope('DUMN'):
            SEQ_USER_T = 50
            INC_DIM = EMBEDDING_DIM

            self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm1', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            # ----------------------------------------获得当前用户的表示----------------------------------------
            attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)  # [128,1,36]
            att_fea = tf.reduce_sum(attention_output, 1)  # [128,1,36]->[128,36]
            user_feat = tf.concat([self.uid_emb, att_fea], axis=-1)  # [[128,18],[128,36]]->[128,18] 这里将当前user_id和user兴趣concat起来
            # ----------------------------------------获得目标物品历史用户的表示----------------------------------------
            item_user_his_attention_output = din_attention(tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_eb)[1] * tf.shape(self.item_user_his_eb)[2]]),  # [128,36]->[128,50*20*36]
                                                           tf.reshape(self.item_user_his_eb, [-1, tf.shape(self.item_user_his_eb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           tf.reshape(self.item_user_his_mid_mask, [-1, tf.shape(self.item_user_his_mid_mask)[2]]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状是[128*50,1,36]

            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_eb)[1], 36])  # [128*50,1,36]->[128*50,36]->[128,50,36]
            item_user_his_att = transformer_model(item_user_his_att, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm2', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            item_user_bhvs_feat = tf.concat([self.item_bh_drink_trm_output, item_user_his_att], axis=-1)  # [[128,50,18],[128,50,36]]->[128,50,54] 这里将目标物品的历史user_id和user兴趣concat起来
            # ----------------------------------------计算当前用户和目标物品历史用户的相似性----------------------------------------
            sim_score = user_similarity(user_feat, item_user_bhvs_feat, need_tile=True) * self.item_user_his_mask  # [128,50]*[128,50]
            sim_score_sum = tf.reduce_sum(sim_score, axis=-1, keep_dims=True)  # [128,1]
            sim_att = tf.reduce_sum(item_user_bhvs_feat * tf.expand_dims(sim_score, -1), axis=1)  # [128,50,54]*[128,50,1]->[128,54]

        inp = tf.concat([inp, user_feat, sim_score_sum, sim_att], -1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)


class DRINK_origin(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_origin, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
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
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, dropout_rate=0, is_training=self.is_training)
                print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                    with tf.name_scope('Attention_layer' + str(i)):
                        print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
                        print('self.target_item_emb.get_shape()', self.target_item_emb.get_shape())
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        print('attention_output.get_shape()', attention_output.get_shape())
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)

        # ----------------------------------------------------- item module -----------------------------------------------------

        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_items_representation'):
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 对历史物品的用户和时间做element-wise_sum_pooling
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            self.item_bh_k_user_emb = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')  # [128,50,18]->[128,50,36]刚开始的时间Embedding是18，转换成36
            item_bh_drink_trm_input = self.item_bh_k_user_emb + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            # 拼接上多个cls的embedding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
            # 这里对拼接的embedding过Transformer的encoder部分
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), float), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            # -----------------------------------------------------处理users的表示 -----------------------------------------------------
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')
            item_user_his_eb_att_sum, _ = attention_net_v1(enc=self.item_bh_drink_trm_output[:, ITEM_BH_CLS_CNT:, :], sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                           dec=self.user_embedded,  # dec=decoder=query[128,36]
                                                           num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                           is_training=self.is_training, reuse=False, scope='item_bh_att')
            # -----------------------------------------------------处理cls的表示 -----------------------------------------------------
            item_bh_cls_embs = self.item_bh_drink_trm_output[:, :ITEM_BH_CLS_CNT, :]  # 这里只包含cls的embedding
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
        self.build_fcn_net(inp, use_dice=True)


class DRINK_test(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(DRINK_test, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
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
                multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, dropout_rate=0, is_training=self.is_training)
                print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
                multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
                # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')
            aux_loss_1 = self.auxiliary_loss(multihead_attention_outputs[:, :-1, :], self.his_item_emb[:, 1:, :], self.noclk_item_his_eb[:, 1:, :], self.mask[:, 1:], stag="gru")
            self.aux_loss = aux_loss_1

            inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
            with tf.name_scope("multi_head_attention"):
                multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
                for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                    multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                    multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                    # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                    print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                    with tf.name_scope('Attention_layer' + str(i)):
                        print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
                        print('self.target_item_emb.get_shape()', self.target_item_emb.get_shape())
                        attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                        print('attention_output.get_shape()', attention_output.get_shape())
                        att_fea = tf.reduce_sum(attention_output, 1)
                        inp = tf.concat([inp, att_fea], 1)

        # ----------------------------------------------------- item module -----------------------------------------------------

        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2  # 这里实际上是统一embedding维度用的

        # 先对购买目标物品的历史用户做一个sum_pooling:item_user_his_eb_sum
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')

        # 截断目标物品的购买用户的长度和时间长度：mapping_to_k处理一个给定的序列（或批量序列），使得每个序列中只保留最后 k 个元素，并生成一个对应的掩码。item_bh_seq_len_t返回的是截断后的真实长度
        # 此处的目的是为了方便修改利用的用户序列长度，而不用每次都在prepare_data中修改，prepare_data中保留较长的序列，这里可以随意控制序列的长短
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)  # [128,50,18],[128],[128,50,1]
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)

        # sequential modeling for item behaviors
        with tf.name_scope('target_item_his_items_representation'):
            att_mask_input = tf.concat([tf.cast(tf.ones([tf.shape(self.item_bh_k_user_emb)[0], ITEM_BH_CLS_CNT]), float), tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])], 1)  # [128,3],[128,50]->[128,53]
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input), axis=2), tf.expand_dims(tf.squeeze(att_mask_input), axis=1)), dtype=tf.int32)  # [128,53,1],[128,1,53]->[128,53,53]
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')  # [128,50,18]->[128,50,36]
            # -----------------------------------------------------目标物品的购买用户过Transformer_model的表示 -----------------------------------------------------
            # 直接对历史物品的用户和时间做element-wise_sum_pooling
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input') + self.item_bh_time_emb_padding
            # item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')如果上面做的不是sum_pooling而是concat这里过一个线性层转换维度
            # 拼接上多个cls的embedding
            item_bh_cls_tile = tf.tile(tf.expand_dims(self.item_bh_cls_embedding, 0), [tf.shape(item_bh_drink_trm_input)[0], 1, 1])  # [3,36]->[1,3,36]->[128,3,36]
            item_bh_drink_trm_input = tf.concat([item_bh_cls_tile, item_bh_drink_trm_input], axis=1)  # [128,3,36],[128,50,36]->[128,53,36]
            # 这里对拼接的embedding过Transformer的encoder部分
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

            # -----------------------------------------------------处理目标物品历史购买用户购买的历史物品序列的表示 -----------------------------------------------------
            item_user_his_attention_output = din_attention(tf.tile(self.target_item_emb, [1, tf.shape(self.item_user_his_eb)[1] * tf.shape(self.item_user_his_eb)[2]]),  # [128,36]->[128,50*20*36]
                                                           tf.reshape(self.item_user_his_eb, [-1, tf.shape(self.item_user_his_eb)[2], 36]),  # [128,50,20,36]->[128*50,20,36]
                                                           tf.reshape(self.item_user_his_mid_mask, [-1, tf.shape(self.item_user_his_mid_mask)[2]]),  # [128,50,20]->[128*50,20]
                                                           need_tile=False)  # 返回的形状是[128*50,1,36]

            item_user_his_att = tf.reshape(tf.reduce_sum(item_user_his_attention_output, 1), [-1, tf.shape(self.item_user_his_eb)[1], 36])  # [128*50,1,36]->[128*50,36]->[128,50,36]
            # 上面得到了目标物品的历史用户的物品序列的表示（区别于前面直接用user_emb表示）
            item_user_his_item_emb = tf.layers.dense(item_user_his_att, INC_DIM, name='item_user_his_item') + self.item_bh_time_emb_padding  # [128,50,36],[128,50,36]->[128,50,36]
            # att_mask_input1 = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])  # [128,50]
            # item_bh_self_att_mask1 = tf.cast(tf.matmul(tf.expand_dims(tf.squeeze(att_mask_input1), axis=2), tf.expand_dims(tf.squeeze(att_mask_input1), axis=1)), dtype=tf.int32)  # [128,50,1],[128,1，50]->[128,50,50]
            #
            # item_user_his_item_emb = transformer_model(item_user_his_item_emb, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask1, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_user_his_att', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            # -----------------------------------------------------获得当前用户历史购买物品的加权表征-----------------------------------------------------
            current_user_his_attention_output = din_attention(self.target_item_emb, self.his_item_emb, self.mask)  # [128,1,36]
            current_user_his_att_emb = tf.reduce_sum(current_user_his_attention_output, 1)  # [128,1,36]->[128,36]

            item_user_his_items_emb_att_sum, _ = attention_net_v1(enc=item_user_his_item_emb, sl=self.item_bh_seq_len_t,  # enc=encoder=keys [128,50,36]这里不包含cls的embedding,sl是keys的真实长度，
                                                                  dec=current_user_his_att_emb,  # dec=decoder=query[128,36]
                                                                  num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0,
                                                                  is_training=self.is_training, reuse=False, scope='item_user_his_items_emb_att_sum')

        # Decoupling
        with tf.name_scope('decoupling'):
            _, rnn_outputs1 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_user_his_uid_batch_embedded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru1")
            _, rnn_outputs2 = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_bh_time_embeeded, sequence_length=self.seq_len_u_ph, dtype=tf.float32, scope="decouple_gru2")
            decoupling_part = rnn_outputs1 + rnn_outputs2

        inp = tf.concat([inp, self.item_user_his_eb_sum, item_user_his_eb_att_sum, item_user_his_items_emb_att_sum, item_bh_cls_dot, item_bh_cls_mat, decoupling_part], 1)
        self.build_fcn_net(inp, use_dice=True)


class myModel(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(myModel, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # ---------------------------------------------------- user module--DMIN  ----------------------------------------------------
        maxlen = 20
        position_embedding_size = EMBEDDING_DIM * 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, dropout_rate=0, is_training=self.is_training)
            print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                with tf.name_scope('Attention_layer' + str(i)):
                    print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
                    print('self.target_item_emb.get_shape()', self.target_item_emb.get_shape())
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                    print('attention_output.get_shape()', attention_output.get_shape())
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)

        # ---------------------------------------------------- item module ----------------------------------------------------
        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2

        # item multi-representation net
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        # paddding
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)
        # sequential modeling for item behaviors
        with tf.name_scope('item_representation'):
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(att_mask_input, axis=2), tf.expand_dims(att_mask_input, axis=1)), dtype=tf.int32)
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
            # 直接对历史物品的用户和时间做element-wise-sum
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')
            # 运用Transformer_model而不是简单的自注意力机制（本质还是线性关系）
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            self.item_bh_time_interval_output = transformer_model(self.item_bh_time_emb_padding, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_time_interval_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            self.item_bh_drink_trm_output += self.item_bh_time_interval_output
            # 对经过Transformer的item-behavior向量运用current_user注意力
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')  # 将user_embedding变成一样的维度
            i_att, _ = attention_net_v1(self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t, dec=self.user_embedded, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=self.is_training, reuse=False, scope='item_bh_att')

        inp = tf.concat([inp, self.item_user_his_eb_sum, i_att], 1)
        self.build_fcn_net(inp, use_dice=True)


class Model_Min(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_Min, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        # ---------------------------------------------------- user module--DMIN  ----------------------------------------------------
        maxlen = 20
        position_embedding_size = EMBEDDING_DIM * 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # B,T,E

        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputs = self_multi_head_attn(self.his_item_emb, num_units=EMBEDDING_DIM * 2, num_heads=4, dropout_rate=0, is_training=self.is_training)
            print('multihead_attention_outputs.get_shape()', multihead_attention_outputs.get_shape())
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs, EMBEDDING_DIM * 4, activation=tf.nn.relu)
            multihead_attention_outputs1 = tf.layers.dense(multihead_attention_outputs1, EMBEDDING_DIM * 2)
            multihead_attention_outputs = multihead_attention_outputs1 + multihead_attention_outputs
            # multihead_attention_outputs = layer_norm(multihead_attention_outputs, name='multi_head_attention')

        inp = tf.concat([self.uid_emb, self.target_item_emb, self.item_his_eb_sum, self.target_item_emb * self.item_his_eb_sum], 1)
        with tf.name_scope("multi_head_attention"):
            multihead_attention_outputss = self_multi_head_attn_v2(multihead_attention_outputs, num_units=36, num_heads=4, dropout_rate=0, is_training=self.is_training)
            for i, multihead_attention_outputs_v2 in enumerate(multihead_attention_outputss):
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs_v2, EMBEDDING_DIM * 4, activation=tf.nn.relu)
                multihead_attention_outputs3 = tf.layers.dense(multihead_attention_outputs3, EMBEDDING_DIM * 2)
                multihead_attention_outputs_v2 = multihead_attention_outputs3 + multihead_attention_outputs_v2
                # multihead_attention_outputs_v2= layer_norm(multihead_attention_outputs_v2, name='multi_head_attention'+str(i))
                print('multihead_attention_outputs_v2.get_shape()', multihead_attention_outputs_v2.get_shape())
                with tf.name_scope('Attention_layer' + str(i)):
                    print('self.position_his_eb.get_shape()', self.position_his_eb.get_shape())
                    print('self.target_item_emb.get_shape()', self.target_item_emb.get_shape())
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_his_eb, self.mask, stag=str(i))
                    print('attention_output.get_shape()', attention_output.get_shape())
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)

        # ---------------------------------------------------- item module ----------------------------------------------------
        SEQ_USER_T = 50
        INC_DIM = EMBEDDING_DIM * 2

        # item multi-representation net
        self.item_user_his_eb_sum = tf.reduce_sum(self.item_user_his_uid_batch_embedded * tf.expand_dims(self.item_user_his_mask, -1), 1)
        self.item_user_his_eb_sum = tf.layers.dense(self.item_user_his_eb_sum, INC_DIM, activation=None, name='item_user_his_eb_sum')
        # paddding
        self.item_bh_k_user_emb, self.item_bh_seq_len_t, self.item_bh_mask_t = mapping_to_k(self.item_user_his_uid_batch_embedded, self.seq_len_u_ph, k=SEQ_USER_T)
        self.item_bh_time_emb_padding, _, _ = mapping_to_k(self.item_bh_time_embeeded, self.seq_len_u_ph, k=SEQ_USER_T)
        # sequential modeling for item behaviors
        with tf.name_scope('item_representation'):
            att_mask_input = tf.reshape(self.item_bh_mask_t, [tf.shape(self.item_bh_k_user_emb)[0], -1])
            item_bh_self_att_mask = tf.cast(tf.matmul(tf.expand_dims(att_mask_input, axis=2), tf.expand_dims(att_mask_input, axis=1)), dtype=tf.int32)
            self.item_bh_time_emb_padding = tf.layers.dense(self.item_bh_time_emb_padding, INC_DIM, name='item_bh_time_embeeded_padding_2dim')
            # 直接对历史物品的用户和时间做element-wise-sum
            item_bh_drink_trm_input = tf.layers.dense(self.item_bh_k_user_emb, INC_DIM, name='item_bh_drink_trm_input')
            # 运用Transformer_model而不是简单的自注意力机制（本质还是线性关系）
            self.item_bh_drink_trm_output = transformer_model(item_bh_drink_trm_input, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_drink_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)
            self.item_bh_time_interval_output = transformer_model(self.item_bh_time_emb_padding, hidden_size=INC_DIM, attention_mask=item_bh_self_att_mask, num_hidden_layers=1, num_attention_heads=2, intermediate_size=256, intermediate_act_fn=gelu, hidden_dropout_prob=0.2, scope='item_bh_time_interval_trm', attention_probs_dropout_prob=0.2, do_return_all_layers=False)

            self.item_bh_drink_trm_output += self.item_bh_time_interval_output
            # 对经过Transformer的item-behavior向量运用current_user注意力
            self.user_embedded = tf.layers.dense(self.uid_emb, INC_DIM, name='user_map_2dim')  # 将user_embedding变成一样的维度
            i_att, _ = attention_net_v1(self.item_bh_drink_trm_output, sl=self.item_bh_seq_len_t, dec=self.user_embedded, num_units=HIDDEN_SIZE, num_heads=4, num_blocks=1, dropout_rate=0.0, is_training=self.is_training, reuse=False, scope='item_bh_att')

        inp = tf.concat([inp, self.item_user_his_eb_sum, i_att], 1)
        self.build_fcn_net(inp, use_dice=True)
