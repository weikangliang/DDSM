# coding=utf-8
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
            self.time_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='time_his_batch_ph')  # 6

            self.item_user_his_batch_ph = tf.placeholder(tf.int32, [None, 50], name='item_user_his_batch_ph')  # 1
            self.item_user_his_time_ph = tf.placeholder(tf.int32, [None, None], name='item_user_his_time_ph')  # 2
            self.item_user_his_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_mid_batch_ph')  # 3 [128,50,20]
            self.item_user_his_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='item_user_his_cat_batch_ph')  # 4

            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')  # 当前用户的历史行为的mask，从后面填充的，被padding的部分mask值为0
            self.item_user_his_mask = tf.placeholder(tf.float32, [None, None], name='item_user_his_mask')  # 目标物品的历史购买用户的mask
            self.item_user_his_mid_mask = tf.placeholder(tf.float32, [None, None, None], name='item_user_his_mid_mask')  # [128,50,20]目标物品的历史购买用户的历史行为的mask

            self.seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')  # 历史行为的长度
            self.seq_len_u_ph = tf.placeholder(tf.int32, [None], name='seq_len_u_ph')  # 目标物品历史用户的长度
            self.target_ph = tf.placeholder(tf.float32, [None, None], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_mid_batch_ph')  # generate 3 item IDs from negative sampling.
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name='noclk_cat_batch_ph')

            self.his_item_user_list_pad = tf.placeholder(tf.int32, [None, None, None], name='his_item_user_list_pad')  # 3 [128,20,5]
            self.his_item_user_mask = tf.placeholder(tf.float32, [None, None, None], name='his_item_user_mask')  # 4 [128,20,5]

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
            self.time_his_batch_emb = tf.nn.embedding_lookup(self.time_embeddings_var, self.time_his_batch_ph)
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
                self.time_his_batch_ph: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_mid_mask: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_len_u_ph: inps[15],
                self.lr: inps[16],
                self.noclk_mid_batch_ph: inps[17],
                self.noclk_cat_batch_ph: inps[18],
                self.his_item_user_list_pad: inps[19],
                self.his_item_user_mask: inps[20],
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
                self.time_his_batch_ph: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_mid_mask: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_len_u_ph: inps[15],
                self.lr: inps[16],
                self.his_item_user_list_pad: inps[17],
                self.his_item_user_mask: inps[18],
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
                self.time_his_batch_ph: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_mid_mask: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_len_u_ph: inps[15],
                self.noclk_mid_batch_ph: inps[16],
                self.noclk_cat_batch_ph: inps[17],
                self.his_item_user_list_pad: inps[18],
                self.his_item_user_mask: inps[19],
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
                self.time_his_batch_ph: inps[6],
                self.item_user_his_batch_ph: inps[7],
                self.item_user_his_mask: inps[8],
                self.item_user_his_time_ph: inps[9],
                self.item_user_his_mid_batch_ph: inps[10],
                self.item_user_his_cat_batch_ph: inps[11],
                self.item_user_his_mid_mask: inps[12],
                self.target_ph: inps[13],
                self.seq_len_ph: inps[14],
                self.seq_len_u_ph: inps[15],
                self.his_item_user_list_pad: inps[16],
                self.his_item_user_mask: inps[17],
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

class Model_DNN_Multi_Head_with_tiv(Model):
    def __init__(self, n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling=True):
        super(Model_DNN_Multi_Head_with_tiv, self).__init__(n_uid, n_mid, n_cat, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE, use_negsampling)
        maxlen = 20
        position_embedding_size = 2
        self.position_his = tf.range(maxlen)
        self.position_embeddings_var = tf.get_variable("position_embeddings_var", [maxlen, position_embedding_size])
        self.position_his_eb = tf.nn.embedding_lookup(self.position_embeddings_var, self.position_his)  # T,E
        self.position_his_eb = tf.tile(self.position_his_eb, [tf.shape(self.his_item_emb)[0], 1])  # B*T,E
        self.position_his_eb = tf.reshape(self.position_his_eb, [tf.shape(self.his_item_emb)[0], -1, self.position_his_eb.get_shape().as_list()[1]])  # 128,20,2
        self.position_with_tiv_his_eb = tf.concat([self.position_his_eb, self.time_his_batch_emb], -1)

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
                    attention_output, attention_score, attention_scores_no_softmax = din_attention_with_position(self.target_item_emb, multihead_attention_outputs_v2, self.position_with_tiv_his_eb, self.mask, stag=str(i))
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inp = tf.concat([inp, att_fea], 1)
        # Fully connected layer
        self.build_fcn_net(inp, use_dice=True)
