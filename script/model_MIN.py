# -*- coding:utf-8 -*-
"""

Author:
    Xiaoke Li,lixiaoke7290@163.com

"""

import tensorflow as tf
from tensorflow.python.keras.layers import (Dense, Flatten)

from ..feature_column import SparseFeat, VarLenSparseFeat, VarUserLenSparseFeat, DenseFeat, build_input_features
from ..inputs import get_varlen_pooling_list, create_embedding_matrix, embedding_lookup, varlen_embedding_lookup, \
    get_dense_input
from ..layers.core import DNN, PredictionLayer
from ..layers.sequence import AttentionSequencePoolingLayer, Transformer2, AttentionSequencePoolingLayer2
from ..layers.utils import concat_func, combined_dnn_input, reduce_sum


def MIN(dnn_feature_columns, history_feature_list, itemshort=False, transformer_num=1, att_head_num=8,
        use_bn=False, dnn_hidden_units=(256, 128, 64), dnn_activation='relu', l2_reg_dnn=0,
        l2_reg_embedding=1e-6, dnn_dropout=0.0, seed=1024, task='binary'):
    """Instantiates the MIN architecture.

     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
     :param history_feature_list: list, to indicate sequence sparse field.
     :param itemshort: Whether recently or Longest Visited Sequence
     :param history_feature_list: list, to indicate sequence sparse field.
     :param transformer_num: int, the number of transformer layer.
     :param att_head_num: int, the number of heads in multi-head self attention.
     :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
     :param dnn_activation: Activation function to use in DNN
     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
     :param seed: integer ,to use as random seed.
     :param task: str, ``"binary"`` for  binary logloss or ``"regression"`` for regression loss
     :return: A Keras model instance.

     """

    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())

    user_behavior_length = features["seq_length"]
    userlist_behavior_length = features["user_seq_length"]
    userlist_length = features["user_length"]

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    user_varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarUserLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

    history_feature_columns = []
    sparse_varlen_feature_columns = []
    history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in history_fc_names:
            history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    user_history_feature_columns = []
    user_history_fc_names = list(map(lambda x: "users_" + x, history_feature_list))
    for fc in user_varlen_sparse_feature_columns:
        feature_name = fc.name
        if feature_name in user_history_fc_names:
            user_history_feature_columns.append(fc)
        else:
            sparse_varlen_feature_columns.append(fc)

    embedding_dict = create_embedding_matrix(dnn_feature_columns, l2_reg_embedding, seed, prefix="", seq_mask_zero=True)

    query_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, return_feat_list=history_feature_list, to_list=True)
    hist_emb_list = embedding_lookup(embedding_dict, features, history_feature_columns, return_feat_list=history_fc_names, to_list=True)
    user_hist_emb_list = embedding_lookup(embedding_dict, features, user_history_feature_columns, return_feat_list=user_history_fc_names, to_list=True)
    dnn_input_emb_list = embedding_lookup(embedding_dict, features, sparse_feature_columns, mask_feat_list=history_feature_list, to_list=True)
    dense_value_list = get_dense_input(features, dense_feature_columns)
    sequence_embed_dict = varlen_embedding_lookup(embedding_dict, features, sparse_varlen_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, sparse_varlen_feature_columns, to_list=True)

    dnn_input_emb_list += sequence_embed_list
    deep_input_emb = concat_func(dnn_input_emb_list)

    query_emb = concat_func(query_emb_list)

    hist_emb = concat_func(hist_emb_list)

    user_hist_emb = concat_func(user_hist_emb_list)

    hist = AttentionSequencePoolingLayer2(att_hidden_units=(80, 40), att_activation="dice", mask_reverse=itemshort, weight_normalization=False, supports_masking=False)([
        query_emb, hist_emb, user_behavior_length])

    transformer_output = hist_emb
    for _ in range(transformer_num):
        att_embedding_size = transformer_output.get_shape().as_list()[-1] // att_head_num
        transformer_layer = Transformer2(att_embedding_size=att_embedding_size, head_num=att_head_num,
                                         dropout_rate=dnn_dropout, mask_reverse=itemshort, use_positional_encoding=True,
                                         use_res=True, use_feed_forward=True, use_layer_norm=True, blinding=False, seed=seed,
                                         supports_masking=False, output_type="sum")
        transformer_output = transformer_layer([transformer_output, transformer_output, user_behavior_length, user_behavior_length])
    attn_output = query_emb * transformer_output

    transformer_outputs = user_hist_emb
    for _ in range(transformer_num):
        att_embedding_size = transformer_outputs.get_shape().as_list()[-1] // att_head_num
        U = transformer_outputs.get_shape().as_list()[1]
        tmp = []
        for i in range(U):
            transformer_layer = Transformer2(att_embedding_size=att_embedding_size, head_num=att_head_num,
                                             dropout_rate=dnn_dropout, mask_reverse=itemshort, use_positional_encoding=True,
                                             use_res=True, use_feed_forward=True, use_layer_norm=True, blinding=False, seed=seed,
                                             supports_masking=False, output_type=None)
            transformer_output_tmp = transformer_layer([transformer_outputs[:, i, :, :], transformer_outputs[:, i, :, :], userlist_behavior_length[:, i, :], userlist_behavior_length[:, i, :]])
            tmp.append(transformer_output_tmp)
        transformer_outputs = tf.stack(tmp, axis=1)

    userlist_transformer_outputs = tf.squeeze(reduce_sum(transformer_outputs, axis=-2, keep_dims=True), axis=2)
    users_attn_output = AttentionSequencePoolingLayer(att_hidden_units=(64, 16), weight_normalization=True,
                                                      supports_masking=False)([transformer_output, userlist_transformer_outputs,
                                                                               userlist_length])

    paddings = tf.zeros_like(hist_emb)
    hist_len = hist_emb.get_shape().as_list()[1]
    hist_masks = tf.sequence_mask(user_behavior_length, hist_len)
    if itemshort:
        hist_masks = tf.reverse(hist_masks, axis=[-1])
    hist_masks = tf.transpose(hist_masks, (0, 2, 1))
    hist_emb = reduce_sum(tf.where(hist_masks, hist_emb, paddings), axis=1, keep_dims=True)

    inter_emb = query_emb * hist_emb

    deep_input_emb = concat_func([deep_input_emb, hist_emb, inter_emb, hist, attn_output, users_attn_output], axis=-1)
    deep_input_emb = Flatten()(deep_input_emb)

    dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
    output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn, seed=seed)(dnn_input)
    final_logit = Dense(1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(output)
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)

    return model
