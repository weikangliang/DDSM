# coding=utf-8
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
# from tensorflow.python.ops.rnn_cell_impl import  _Linear
# from tensorflow import keras
from tensorflow.python.ops.rnn_cell import *


# from keras import backend as K

class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(QAAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
      num_units: int, The number of units in the GRU cell.
      activation: Nonlinearity to use.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
       in an existing scope.  If not `True`, and the existing scope already has
       the given variables, an error is raised.
      kernel_initializer: (optional) The initializer to use for the weight and
      projection matrices.
      bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary
    Args:
        raw_arr (TYPE): Description
    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def self_attention_ieu(x, embed_dim, att_size=72, padding_mask=None):
    # Define the linear transformations
    trans_Q = tf.layers.dense(inputs=x, units=att_size)
    trans_K = tf.layers.dense(inputs=x, units=att_size)
    trans_V = tf.layers.dense(inputs=x, units=att_size)

    # Calculate the attention scores
    attention_scores = tf.matmul(trans_Q, trans_K, transpose_b=True)

    if padding_mask is not None:
        padding_mask_expand = tf.tile(tf.expand_dims(padding_mask, 1), [1, tf.shape(x)[1], 1])  # [128,20]->[128,1,20]->[128,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand) * (-2 ** 32 + 1)  # [128,20,20]
        attention_scores = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, attention_scores)  # [128,20,20]
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    # Apply the attention scores to the values
    context = tf.matmul(attention_scores, trans_V)  # [128,20,20],[128,20,36]-> [128,20,36]

    # Projection layer
    context = tf.layers.dense(inputs=context, units=embed_dim)  # 将最后一维由att_size换成原来的embed_dim[128,20,36]->[128,20,36]
    context = context * tf.expand_dims(padding_mask, -1)  # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]
    # Optionally, you can include layer normalization and dropout here if needed
    # context = tf.contrib.layers.layer_norm(context)
    # context = tf.nn.dropout(context, rate=0.5)
    return context


def self_multi_head_attention_ieu(x, embed_dim, num_heads=4, att_size=72, padding_mask=None):
    # 确保att_size可以被num_heads整除
    assert att_size % num_heads == 0
    depth = att_size // num_heads

    def split_heads(x, batch_size):
        # 分割最后一个维度到(num_heads, depth)
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        # 重排为(batch_size, num_heads, seq_length, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    batch_size = tf.shape(x)[0]

    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=x, units=att_size)
    trans_K = tf.layers.dense(inputs=x, units=att_size)
    trans_V = tf.layers.dense(inputs=x, units=att_size)

    # 分割每个头
    trans_Q = split_heads(trans_Q, batch_size)  # [128,4,20,18]
    trans_K = split_heads(trans_K, batch_size)  # [128,4,20,18]
    trans_V = split_heads(trans_V, batch_size)  # [128,4,20,18]

    # 计算注意力分数
    attention_scores = tf.matmul(trans_Q, trans_K, transpose_b=True)  # [128,4,20,18],[128,4,18,20]->[128,4,20,20]
    attention_scores = attention_scores / tf.math.sqrt(tf.cast(depth, tf.float32))

    if padding_mask is not None:
        padding_mask_expand = tf.tile(tf.expand_dims(tf.expand_dims(padding_mask, 1), 1), [1, num_heads, tf.shape(x)[1], 1])  # [128,4,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand) * (-2 ** 32 + 1)
        attention_scores = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, attention_scores)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    # 应用注意力分数到值上
    context = tf.matmul(attention_scores, trans_V)  # [128,4,20,20],[128,4,20,18]->[128,4,20,18]

    # 重新拼接多头
    context = tf.transpose(context, perm=[0, 2, 1, 3])  # [128,4,20,18]->[128,20,4,18]
    context = tf.reshape(context, (batch_size, -1, att_size))  # [128,20,4,18]->[128,20,72]

    # 投影层
    context = tf.layers.dense(inputs=context, units=embed_dim)  # [batch_size, seq_length, embed_dim]
    if padding_mask is not None:
        context = context * tf.expand_dims(padding_mask, -1)  # # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]

    # 可选：添加层归一化和dropout
    # context = tf.contrib.layers.layer_norm(context)
    # context = tf.nn.dropout(context, rate=0.5)

    return context


def multi_layer_perceptron_prelu(input_tensor, embed_dims, dropout_rate=0):  # embd_dims[256,128,64,32]代表的是有多少层以及每一层的维度
    x = input_tensor
    for embed_dim in embed_dims:
        x = tf.layers.dense(x, units=embed_dim)  # Linear layer
        # x = tf.contrib.layers.layer_norm(x)
        x = tf.keras.layers.PReLU()(x)  # PReLU activation
        x = tf.layers.dropout(x, rate=dropout_rate)  # Dropout
    return x


def ieu(inputs, seq_length, embed_dim, padding_mask, weight_type="bit", bit_layers=1, att_size=36, mlp_layer=256):
    input_dim = seq_length * embed_dim  # [20*36]
    x_vector = self_attention_ieu(inputs, embed_dim, att_size=att_size, padding_mask=padding_mask)  # [128,20,36]
    # Contextual information extractor (CIE) unit
    inputs = inputs * tf.expand_dims(padding_mask, axis=-1)  # [128,20,36],[128,20,1]->[128,20,36]

    mlp_layers = [mlp_layer for _ in range(bit_layers)]
    x_bit = multi_layer_perceptron_prelu(tf.reshape(inputs, [-1, input_dim]), mlp_layers)  # [128,720]->[128,256]
    x_bit = tf.layers.dense(x_bit, units=embed_dim, activation=tf.nn.relu)  # [128,256]-[128,36] 将x_bit压缩成embed_dim形状
    x_bit = tf.reshape(x_bit, [-1, 1, embed_dim])  # [128,36]->[128,1,36]
    # Integration unit
    x_out = x_bit * x_vector  # [128,1,36],[128,20,36]-> [128,20,36]
    if weight_type == "vector":
        x_out = tf.reduce_sum(x_out, axis=-1, keepdims=True)  # [128,20,36]->[128,20,1]
    return x_out


def frnet(inputs, seq_length, embed_dim, padding_mask, weight_type="bit", num_layers=1, att_size=36, mlp_layer=256):
    # IEU_G computes complementary features.
    IEU_G = ieu(inputs, seq_length, embed_dim, padding_mask, weight_type="bit", bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)  # [128,20,36]->[128,20,36]
    # IEU_W computes bit-level or vector-level weights.
    IEU_W = ieu(inputs, seq_length, embed_dim, padding_mask, weight_type=weight_type, bit_layers=num_layers, att_size=att_size, mlp_layer=mlp_layer)  # [128,20,36]->[128,20,36]
    IEU_W = tf.sigmoid(IEU_W)  # [128,20,36]->[128,20,36]
    # Computing the output
    x_out = inputs * IEU_W + IEU_G * (1.0 - IEU_W)  # [128,20,36]->[128,20,36]
    return x_out


def get_context_matrix(input_context_tensor, embed_dims, dropout_rate=0., is_training=False, scope=""):
    x = input_context_tensor  # [128,756]
    for embed_dim in embed_dims:
        x = tf.layers.dense(x, units=embed_dim, activation=tf.nn.sigmoid)  # [128,162]-> [128,81]
        x = tf.layers.dropout(x, rate=dropout_rate, training=is_training)  # Dropout
    return x  # [128,81]


def context_aware_self_attention(inputs, context_weights_matrix, embed_dim=36, num_units=18, padding_mask=None, causality_mask_bool=False, is_layer_norm=True, dropout_rate=0, is_training=True, name="context_aware_self_attention"):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units)  # [128,20,36]->[128,20,18]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units)  # [128,20,36]->[128,20,18]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units)  # [128,20,36]->[128,20,18]

    matrix_shape = tf.shape(context_weights_matrix)
    identity_matrix = tf.eye(matrix_shape[1], matrix_shape[2])  # 创建单位矩阵 [18,18]
    identity_matrix = tf.tile(tf.expand_dims(identity_matrix, 0), [matrix_shape[0], 1, 1])  # [512,18,18]
    weight_context = tf.get_variable(name + "weight_context", shape=[], initializer=tf.constant_initializer(1.0), trainable=True)
    linear_combination = weight_context * context_weights_matrix + identity_matrix

    attention_scores = tf.matmul(trans_Q, linear_combination)  # [128,20,18],[128,18,18]->[128,20,18]
    attention_scores = tf.matmul(attention_scores, trans_K, transpose_b=True)  # [128,20,18],[128,18,20]->[128,20,20]
    align = attention_scores / (18 ** 0.5)  # [512,20,20]
    outputs_scores = align
    if padding_mask is not None:
        padding_mask_expand = tf.tile(tf.expand_dims(padding_mask, 1), [1, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[128,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand) * (-2 ** 32 + 1)  # [128,20,20]
        attention_scores = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, outputs_scores)  # [128,20,20]

    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    # Apply the attention scores to the values
    outputs = tf.matmul(attention_scores, trans_V)  # [128,20,20],[128,20,36]-> [128,20,36]
    # Projection layer
    outputs = tf.layers.dense(inputs=outputs, units=embed_dim)  # 将最后一维由att_size换成原来的embed_dim[128,20,36]->[128,20,36]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)  # [512,20,20]
    outputs += inputs  # Residual connection
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name + "layer_norm")  # [128,20,36]每个头用的layer_norm层不一样
    if padding_mask is not None:
        outputs = outputs * tf.expand_dims(padding_mask, -1)  # [128,20,36],[128,20]->[128,20,36],[128,20,1]->[128,20,36]
    return [outputs]


def context_aware_multi_head_self_attention(inputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,8],[128,20,8],[128,20,8],[128,20,8]]->[512,20,8]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,8],[128,20,8],[128,20,8],[128,20,8]]->[512,20,8]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,8],[128,20,8],[128,20,8],[128,20,8]]->[512,20,8]
    # 转换注意力得分矩阵
    context_weights_matrix_flatten = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], 256])  # [128,16,16]->[128,256]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix_flatten, num_heads, axis=1), axis=0)  # [128,256]->[512,64]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [128,16,16]->[512,8,8]

    scores_orign = tf.matmul(trans_Q, trans_K, transpose_b=True)  # [512,20,8],[512,8,20]->[512,20,20]

    scores_context = tf.matmul(trans_Q, context_weights_matrix)  # [512,20,8],[512,8,8]->[512,20,8]
    scores_context = tf.matmul(scores_context, trans_K, transpose_b=True)  # [512,20,8],[512,8,20]->[512,20,20]

    align_context = scores_context / (num_units ** 0.5)  # [512,20,20]
    align_orign = scores_orign / (num_units ** 0.5)

    outputs_scores_context = align_context
    outputs_scores_origin = align_orign
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores_context = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores_origin)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align_context[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align_context)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores_context = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_origin)  # [512,20,20]
    outputs_scores_context = tf.nn.softmax(outputs_scores_context, axis=-1)
    outputs_scores_origin = tf.nn.softmax(outputs_scores_origin, axis=-1)
    # 应用注意力得分到 V
    outputs_context = tf.matmul(outputs_scores_context, trans_V)  # [512,20,20],[512,20,8]->[512,20,8]
    outputs_origin = tf.matmul(outputs_scores_origin, trans_V)  # [512,20,20],[512,20,8]->[512,20,8]
    # 恢复头结构
    outputs_context = tf.concat(tf.split(outputs_context, num_heads, axis=0), axis=2)  # [512,20,16]->[[128,20,16],[128,20,16],[128,20,16],[128,20,16]]->[128,20,64]
    outputs_origin = tf.concat(tf.split(outputs_origin, num_heads, axis=0), axis=2)  # [512,20,16]->[[128,20,16],[128,20,16],[128,20,16],[128,20,16]]->[128,20,64]
    outputs_context = tf.layers.dense(inputs=outputs_context, units=inputs.get_shape().as_list()[-1])  # [128,20,64]->[128,20,36]
    outputs_origin = tf.layers.dense(inputs=outputs_origin, units=inputs.get_shape().as_list()[-1])  # [128,20,64]->[128,20,36]
    outputs = inputs + outputs_context
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # Normalize
    return outputs


def context_aware_multi_head_self_attention_ieu(inputs, context_weights_matrix, context_weights_matrix_weight, num_heads=4, num_units=9, padding_mask=None, dropout_rate=0, is_training=True, causality_mask_bool=False, is_layer_norm=False, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 全局信息矩阵
    context_weights_matrix_flatten = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], 324])  # [128,16,16]->[128,256]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix_flatten, num_heads, axis=1), axis=0)  # [128,256]->[512,64]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,64]->[512,9,9]

    scores_origin = tf.matmul(trans_Q, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]
    scores_origin = tf.layers.dropout(scores_origin, dropout_rate, training=is_training)  # [512,20,20]

    scores_context = tf.matmul(trans_Q, context_weights_matrix)  # [512,20,9],[512,9,9]->[512,20,9]
    scores_context = tf.matmul(scores_context, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]
    scores_context = tf.layers.dropout(scores_context, dropout_rate, training=is_training)  # [512,20,20]

    align_context = scores_context / (num_units ** 0.5)  # [512,20,20]
    align_origin = scores_origin / (num_units ** 0.5)

    outputs_scores_context = align_context
    outputs_scores_origin = align_origin
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask_expand = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores_context = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, outputs_scores_origin)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align_context[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align_context)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores_context = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_origin)  # [512,20,20]

    outputs_scores_context = tf.nn.softmax(outputs_scores_context, axis=-1)
    outputs_scores_origin = tf.nn.softmax(outputs_scores_origin, axis=-1)

    # 应用注意力得分到 V
    outputs_context = tf.matmul(outputs_scores_context, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs_origin = tf.matmul(outputs_scores_origin, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # 恢复头结构
    outputs_context = tf.concat(tf.split(outputs_context, num_heads, axis=0), axis=2)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]
    outputs_origin = tf.concat(tf.split(outputs_origin, num_heads, axis=0), axis=2)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]
    outputs_context *= tf.expand_dims(padding_mask, -1)  # [128,20,1],[128,20,36]->[128,20,36]
    outputs_origin *= tf.expand_dims(padding_mask, -1)  # [128,20,1],[128,20,36]->[128,20,36]
    outputs_context = tf.layers.dense(inputs=outputs_context, units=inputs.get_shape().as_list()[-1])  # [128,20,36]->[128,20,36]
    outputs_origin = tf.layers.dense(inputs=outputs_origin, units=inputs.get_shape().as_list()[-1])  # [128,20,]->[128,20,36]

    # 计算现在的部分与原来的部分重要性权重
    context_weights_matrix_flatten_weight = tf.reshape(context_weights_matrix_weight, [tf.shape(inputs)[0], 324])  # [128,16,16]->[128,256]
    x_bit = tf.layers.dense(context_weights_matrix_flatten_weight, units=num_heads * num_units, activation=tf.nn.relu)  # [128,256]-[128,36] 将x_bit压缩成embed_dim形状
    x_bit = tf.reshape(x_bit, [-1, 1, num_heads * num_units])  # [128,36]->[128,1,36]

    x_out = x_bit * outputs_origin  # [128,1,36],[128,20,36]-> [128,20,36]

    # if weight_type == "vector":
    x_out = tf.reduce_sum(x_out, axis=-1, keepdims=True)  # [128,20,36]->[128,20,1]
    IEU_W = tf.sigmoid(x_out)  # [128,20,1]->[128,20,1]
    IEU_W *= tf.expand_dims(padding_mask, -1)
    outputs_context = tf.layers.dropout(outputs_context, dropout_rate, training=is_training)  # [512,20,20]
    outputs = inputs * IEU_W + outputs_context * (1.0 - IEU_W)  # [128,20,36]->[128,20,36]
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # Normalize
    return outputs


def context_aware_multi_head_self_attention_ieu2(inputs, context_weights_matrix, context_weights_matrix_weight, num_heads=4, num_units=9, padding_mask=None, dropout_rate=0, is_training=True, causality_mask_bool=False, is_layer_norm=False, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 全局信息矩阵
    context_weights_matrix_flatten = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], 324])  # [128,16,16]->[128,256]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix_flatten, num_heads, axis=1), axis=0)  # [128,256]->[512,64]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,64]->[512,9,9]

    trans_Q_origin = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]
    trans_K_origin = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]
    trans_V_origin = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]

    scores_origin = tf.matmul(trans_Q_origin, trans_K_origin, transpose_b=True)  # [128,20,36],[128,36,20]->[128,20,20]
    scores_origin = tf.layers.dropout(scores_origin, dropout_rate, training=is_training)  # [128,20,20]

    scores_context = tf.matmul(trans_Q, context_weights_matrix)  # [512,20,9],[512,9,9]->[512,20,9]
    scores_context = tf.matmul(scores_context, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]
    scores_context = tf.layers.dropout(scores_context, dropout_rate, training=is_training)  # [512,20,20]

    align_context = scores_context / (num_units ** 0.5)  # [512,20,20]
    align_origin = scores_origin / (num_units ** 0.5)  # [128,20,20]

    outputs_scores_context = align_context
    outputs_scores_origin = align_origin
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask_expand1 = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand1) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores_context = tf.where(tf.equal(padding_mask_expand1, 0), padding_mask_val, outputs_scores_context)  # [512,20,20]
        padding_mask_expand2 = tf.tile(tf.expand_dims(padding_mask, 1), [1, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[128,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand2) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(padding_mask_expand2, 0), padding_mask_val, outputs_scores_origin)  # [128,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align_context[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask1 = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align_context)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask1) * (-2 ** 32 + 1)
        outputs_scores_context = tf.where(tf.equal(causality_mask1, 0), causality_mask_val, outputs_scores_context)  # [512,20,20]
        causality_mask2 = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs_scores_origin)[0], 1, 1])  # [128,20,20]
        causality_mask_val = tf.ones_like(causality_mask2) * (-2 ** 32 + 1)
        outputs_scores_origin = tf.where(tf.equal(causality_mask2, 0), causality_mask_val, outputs_scores_origin)  # [128,20,20]

    outputs_scores_context = tf.nn.softmax(outputs_scores_context, axis=-1)  # [512,20,20]
    outputs_scores_origin = tf.nn.softmax(outputs_scores_origin, axis=-1)  # [128,20,20]

    # 应用注意力得分到 V
    outputs_context = tf.matmul(outputs_scores_context, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs_origin = tf.matmul(outputs_scores_origin, trans_V_origin)  # [128,20,20],[128,20,36]->[128,20,36]
    # 恢复头结构
    outputs_context = tf.concat(tf.split(outputs_context, num_heads, axis=0), axis=2)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]

    outputs_context = tf.layers.dense(inputs=outputs_context, units=inputs.get_shape().as_list()[-1])  # [128,20,36]->[128,20,36]
    outputs_origin = tf.layers.dense(inputs=outputs_origin, units=inputs.get_shape().as_list()[-1])  # [128,20,36]->[128,20,36]

    # 计算现在的部分与原来的部分重要性权重
    # context_weights_matrix_flatten_weight = tf.reshape(context_weights_matrix_weight, [tf.shape(inputs)[0], 324])  # [128,18,18]->[128,324]
    x_bit = tf.layers.dense(context_weights_matrix_weight, units=324, activation=tf.nn.sigmoid)  # [128,324]-[128,36] 将x_bit压缩成embed_dim形状
    x_bit = tf.layers.dense(x_bit, units=num_heads * num_units, activation=tf.nn.relu)
    x_bit = tf.reshape(x_bit, [-1, 1, num_heads * num_units])  # [128,36]->[128,1,36]
    x_out = x_bit * outputs_origin  # [128,1,36],[128,20,36]-> [128,20,36]

    # if weight_type == "vector":
    x_out = tf.reduce_sum(x_out, axis=-1, keepdims=True)  # [128,20,36]->[128,20,1]
    IEU_W = tf.sigmoid(x_out)  # [128,20,1]->[128,20,1]
    outputs_context = tf.layers.dropout(outputs_context, dropout_rate, training=is_training)  # [512,20,20]
    outputs = inputs * IEU_W + outputs_context * (1.0 - IEU_W)  # [128,20,36]->[128,20,36]
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # Normalize
    return outputs


def context_aware_multi_head_self_attention_origin(inputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, dropout_rate=0, is_training=True, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 转换全局信息矩阵
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], -1])  # [128,18,18]->[128,324]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix, num_heads, axis=1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]
    # context_weights_matrix = tf.matmul(context_weights_matrix,context_weights_matrix,transpose_b=True)#是否需要该矩阵对称
    matrix_shape = tf.shape(context_weights_matrix)  # 获取矩阵的形状[128,9,9]

    identity_matrix = tf.eye(matrix_shape[1], matrix_shape[2])  # 创建单位矩阵 [9,9]
    identity_matrix = tf.tile(tf.expand_dims(identity_matrix, 0), [matrix_shape[0], 1, 1])  # [512,9,9]
    # 定义num_heads个可训练的权重变量
    weights_context = [tf.get_variable(name + "weights_context" + str(i), shape=[], initializer=tf.constant_initializer(1.0), trainable=True) for i in range(num_heads)]
    # 创建context的权重矩阵，形状为 [512,1,1]
    expanded_weights_context_list = [tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(w, 0), 0), 0), [tf.shape(inputs)[0], 1, 1]) for w in weights_context]  # [[128,1,1],[128,1,1],[128,1,1],[128,1,1]]
    final_context_weights = tf.concat(expanded_weights_context_list, axis=0)  # [[128,1,1],[128,1,1],[128,1,1],[128,1,1]]-> [512,1,1]
    # 对两个矩阵进行线性加和
    linear_combination = final_context_weights * context_weights_matrix + identity_matrix

    scores = tf.matmul(trans_Q, linear_combination)  # [512,20,9],[512,9,9]->[512,20,9]
    scores = tf.matmul(scores, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]

    align = scores / (9 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # [128,20,9]->[128,20,36]先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,36]每一个头都加上inputs，但是每个头都自己决定加多少上去，而不是直接用outputs_sub += inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def context_aware_multi_head_self_attention_v2_no_weight(inputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, dropout_rate=0, is_training=True, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 转换全局信息矩阵
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], -1])  # [128,324]->[128,324]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix, num_heads, axis=-1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]

    scores = tf.matmul(trans_Q, context_weights_matrix)  # [512,20,9],[512,9,9]->[512,20,9]
    scores = tf.matmul(scores, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]

    align = scores / (9 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # [128,20,9]->[128,20,36]先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += tf.layers.dense(inputs=inputs, units=inputs.get_shape().as_list()[-1])  # [128,20,36]->[128,20,36]每一个头都加上inputs，但是每个头都自己决定加多少上去，而不是直接用outputs_sub += inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def context_aware_multi_head_self_attention_v2(inputs, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, dropout_rate=0, is_training=True, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 转换全局信息矩阵
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], -1])  # [128,324]->[128,324]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix, num_heads, axis=1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]
    # context_weights_matrix = tf.matmul(context_weights_matrix,context_weights_matrix,transpose_b=True)#是否需要该矩阵对称
    matrix_shape = tf.shape(context_weights_matrix)  # 获取矩阵的形状[128,9,9]

    identity_matrix = tf.eye(matrix_shape[1], matrix_shape[2])  # 创建单位矩阵 [9,9]
    identity_matrix = tf.tile(tf.expand_dims(identity_matrix, 0), [matrix_shape[0], 1, 1])  # [512,9,9]
    # 定义num_heads个可训练的权重变量
    weights_context = [tf.get_variable(name + "weights_context" + str(i), shape=[], initializer=tf.constant_initializer(1.0), trainable=True) for i in range(num_heads)]
    # 创建context的权重矩阵，形状为 [512,1,1]
    expanded_weights_context_list = [tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(w, 0), 0), 0), [tf.shape(inputs)[0], 1, 1]) for w in weights_context]  # [[128,1,1],[128,1,1],[128,1,1],[128,1,1]]
    final_context_weights = tf.sigmoid(tf.concat(expanded_weights_context_list, axis=0))  # [[128,1,1],[128,1,1],[128,1,1],[128,1,1]]-> [512,1,1]
    # 对两个矩阵进行线性加和
    linear_combination = final_context_weights * context_weights_matrix + (1 - final_context_weights) * identity_matrix

    scores = tf.matmul(trans_Q, linear_combination)  # [512,20,9],[512,9,9]->[512,20,9]
    scores = tf.matmul(scores, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]

    align = scores / (9 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # [128,20,9]->[128,20,36]先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,36]每一个头都加上inputs，但是每个头都自己决定加多少上去，而不是直接用outputs_sub += inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def context_aware_multi_head_self_attention_v2_bit_wise(inputs, context_embs_matrix, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, dropout_rate=0, is_training=True, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 转换全局信息矩阵
    context_embs_matrix = tf.reshape(context_embs_matrix, [tf.shape(inputs)[0], -1])  # [128,324]->[128,324]
    context_embs_matrix = tf.concat(tf.split(context_embs_matrix, num_heads, axis=1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_embs_matrix = tf.reshape(context_embs_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]
    # 转换权重矩阵
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], -1])  # [128,324]->[128,324]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix, num_heads, axis=1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]

    # context_embs_matrix = tf.matmul(context_embs_matrix,context_embs_matrix,transpose_b=True)#是否需要该矩阵对称
    matrix_shape = tf.shape(context_embs_matrix)  # 获取矩阵的形状[128,9,9]

    identity_matrix = tf.eye(matrix_shape[1], matrix_shape[2])  # 创建单位矩阵 [9,9]
    identity_matrix = tf.tile(tf.expand_dims(identity_matrix, 0), [matrix_shape[0], 1, 1])  # [512,9,9]
    # 对两个矩阵进行线性加和
    linear_combination = context_weights_matrix * context_embs_matrix + (1 - context_weights_matrix) * identity_matrix

    scores = tf.matmul(trans_Q, linear_combination)  # [512,20,9],[512,9,9]->[512,20,9]
    scores = tf.matmul(scores, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]

    align = scores / (9 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # [128,20,9]->[128,20,36]先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,36]每一个头都加上inputs，但是每个头都自己决定加多少上去，而不是直接用outputs_sub += inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def context_aware_multi_head_self_attention_v2_vector_wise(inputs, context_embs_matrix, context_weights_matrix, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, is_layer_norm=False, dropout_rate=0, is_training=True, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 转换全局信息矩阵
    context_embs_matrix = tf.reshape(context_embs_matrix, [tf.shape(inputs)[0], -1])  # [128,324]->[128,324]
    context_embs_matrix = tf.concat(tf.split(context_embs_matrix, num_heads, axis=1), axis=0)  # [128,324]->[[128,81],[128,81],[128,81],[128,81]]->[512,81]
    context_embs_matrix = tf.reshape(context_embs_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]
    # 转换权重矩阵
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], -1])  # [128,36]->[128,36]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix, num_heads, axis=1), axis=0)  # [128,36]->[[128,9],[128,9],[128,9],[128,9]]->[512,9]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, 1])  # [512,9]->[512,9,1]

    matrix_shape = tf.shape(context_embs_matrix)  # 获取矩阵的形状[128,9,9]

    identity_matrix = tf.eye(matrix_shape[1], matrix_shape[2])  # 创建单位矩阵 [9,9]
    identity_matrix = tf.tile(tf.expand_dims(identity_matrix, 0), [matrix_shape[0], 1, 1])  # [512,9,9]
    # 对两个矩阵进行线性加和
    linear_combination = context_weights_matrix * context_embs_matrix + (1 - context_weights_matrix) * identity_matrix

    scores = tf.matmul(trans_Q, linear_combination)  # [512,20,9],[512,9,9]->[512,20,9]
    scores = tf.matmul(scores, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]

    align = scores / (9 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # [128,20,9]->[128,20,36]先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,36]每一个头都加上inputs，但是每个头都自己决定加多少上去，而不是直接用outputs_sub += inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def context_aware_multi_head_self_attention_v2_ieu(inputs, context_weights_matrix, context_weights_matrix_weight, num_heads=4, num_units=9, padding_mask=None, causality_mask_bool=False, dropout_rate=0, is_training=True, is_layer_norm=False, name=""):
    # 定义线性变换
    trans_Q = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_K = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    trans_V = tf.layers.dense(inputs=inputs, units=num_units * num_heads)  # [128,20,36]->[128,20,4*9]
    # 分割为多个头
    trans_Q = tf.concat(tf.split(trans_Q, num_heads, axis=2), axis=0)  # [h*N, T_q, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_K = tf.concat(tf.split(trans_K, num_heads, axis=2), axis=0)  # [h*N, T_k, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    trans_V = tf.concat(tf.split(trans_V, num_heads, axis=2), axis=0)  # [h*N, T_v, C/h][128,20,4*8]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # 全局信息矩阵
    context_weights_matrix_flatten = tf.reshape(context_weights_matrix, [tf.shape(inputs)[0], 324])  # [128,18,18]->[128,324]
    context_weights_matrix = tf.concat(tf.split(context_weights_matrix_flatten, num_heads, axis=1), axis=0)  # [128,324]->[512,81]
    context_weights_matrix = tf.reshape(context_weights_matrix, [tf.shape(trans_Q)[0], num_units, num_units])  # [512,81]->[512,9,9]

    scores_origin = tf.matmul(trans_Q, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]
    scores_origin = tf.layers.dropout(scores_origin, dropout_rate, training=is_training)  # [512,20,20]

    scores_context = tf.matmul(trans_Q, context_weights_matrix)  # [512,20,9],[512,9,9]->[512,20,9]
    scores_context = tf.matmul(scores_context, trans_K, transpose_b=True)  # [512,20,9],[512,9,20]->[512,20,20]
    scores_context = tf.layers.dropout(scores_context, dropout_rate, training=is_training)  # [512,20,20]

    align_context = scores_context / (num_units ** 0.5)  # [512,20,20]
    align_origin = scores_origin / (num_units ** 0.5)

    outputs_scores_context = align_context
    outputs_scores_origin = align_origin
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask_expand = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask_expand) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores_context = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(padding_mask_expand, 0), padding_mask_val, outputs_scores_origin)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align_context[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align_context)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores_context = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_context)  # [512,20,20]
        outputs_scores_origin = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores_origin)  # [512,20,20]

    outputs_scores_context = tf.nn.softmax(outputs_scores_context, axis=-1)  # [512,20,20]
    outputs_scores_origin = tf.nn.softmax(outputs_scores_origin, axis=-1)  # [512,20,20]

    # 应用注意力得分到 V
    outputs_context = tf.matmul(outputs_scores_context, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs_origin = tf.matmul(outputs_scores_origin, trans_V)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs_context = tf.split(outputs_context, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputs_origin = tf.split(outputs_origin, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]

    # 计算现在的部分与原来的部分重要性权重
    context_weights_matrix_flatten_weight = tf.reshape(context_weights_matrix_weight, [tf.shape(inputs)[0], 324])  # [128,18,18]->[128,324]
    x_bit = tf.layers.dense(context_weights_matrix_flatten_weight, units=num_heads * num_units, activation=tf.nn.relu)  # [128,324]-[128,36] 将x_bit压缩成embed_dim形状
    x_bit = tf.reshape(x_bit, [-1, 1, num_heads * num_units])  # [128,36]->[128,1,36]

    x_out = x_bit * outputs_origin  # [128,1,36],[128,20,36]-> [128,20,36]
    x_out = tf.concat(tf.split(x_out, num_heads, axis=2), axis=0)  # [128,20,36]->[512,20,9]
    IEU_W = tf.sigmoid(x_out)  # [512,20,9]->[512,20,9]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs_context):  # 下面每个头都使用不同的dense层
        outputs_sub = inputs * IEU_W + outputs_sub * (1.0 - IEU_W)  # [512,20,9]->[512,20,9]
        outputs_sub = tf.layers.dense(outputs_sub, inputs.get_shape().as_list()[-1])  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += inputs  # 每一个头都加上inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def user_similarity(user_feat, item_user_bhvs, need_tile=True):  # [128,54],[128,50,54]
    user_feats = tf.tile(user_feat, [1, tf.shape(item_user_bhvs)[1]]) if need_tile else user_feat  # [128,54]->[128,50*54]
    user_feats = tf.reshape(user_feats, tf.shape(item_user_bhvs))  # [128,50*54]->[128,50,54]
    # ------------------------------------分别计算当前用户的表示和目标物品历史用户表示各自的模------------------------------------
    pooled_len_1 = tf.sqrt(tf.reduce_sum(user_feats * user_feats, -1))  # [128,50,54],[128,50,54]->[128,50,54]->[128,50]
    pooled_len_2 = tf.sqrt(tf.reduce_sum(item_user_bhvs * item_user_bhvs, -1))  # [128,50,54],[128,50,54]->[128,50,54]->[128,50]
    # ------------------------------------分别计算当前用户的表示和目标物品历史用户表示的内积------------------------------------
    pooled_mul_12 = tf.reduce_sum(user_feats * item_user_bhvs, -1)  # [128,50,54]*[128,50,54]->[128,50,54]->[128,50]
    # ------------------------------------分别计算当前用户的表示和目标物品历史用户表示的余弦相似度------------------------------------
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")  # [128,50]
    return score  # [128,50]


def din_attention(query, facts, mask, stag='null', mode='SUM', softmax_stag=1, need_tile=True):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    queries = tf.tile(query, [1, tf.shape(facts)[1]]) if need_tile else query  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]->[128,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag, reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag, reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag, reuse=tf.AUTO_REUSE)  # [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,20,1]->[128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,20]->[128,1,20]    [128*50,20]->[128*50,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,1,20]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,1,20],[128,20,36]->[128,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])  # [128,20]
        output = facts * tf.expand_dims(scores, -1)  # [128,20,36],[128,20,1]->[128,20,36]
        output = tf.reshape(output, tf.shape(facts))  # [128,20,36]->[128,20,36]
    return output  # [128,1,36]or[128,20,36]


def IUI_attention(query, facts, mask=None, mode='SUM',name='null', softmax_stag=1):
    queries = tf.tile(tf.expand_dims(query,axis=2), [1, 1,tf.shape(facts)[1],1])  # [128,3,36]->[128,3,20,36]
    facts_ex = tf.tile(tf.expand_dims(facts,axis=1), [1, tf.shape(query)[1],1,1])  # [128,20,36]->[128,3,20,36]

    din_all = tf.concat([queries, facts_ex, queries - facts_ex, queries * facts_ex], axis=-1)  # [[128,3,20,36],[128,3,20,36],[128,3,20,36],[128,3,20,36]]->[128,3,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name= name + 'f1_att', reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name= name + 'f2_att', reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name= name + 'f3_att', reuse=tf.AUTO_REUSE) # [128,3,20,1]
    scores = tf.squeeze(d_layer_3_all, axis=-1)  # [128,3,20,1]->[128,3,20]
    # Mask
    if mask is not None:
        padding_mask = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(query)[1], 1])  # [128,20]->[128,1,20]->[128,3,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [128,3,20]
        scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, scores)  # [128,3,20]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,3,20]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,3,20],[128,20,36]->[128,3,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])  # [128,20]
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output  # [128,3,36]


def din_attention_DIB_with_user(query, facts, mask, stag='null', mode='SUM', softmax_stag=1, need_tile=True):  # [128,18],[128,20,5,36],[128,20,5]
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    query = tf.layers.dense(query, facts.get_shape().as_list()[-1], activation=None, name=stag + 'din_attention_DIB_query')  # [128,18]->[128,36]
    queries = tf.tile(query, [1, tf.shape(facts)[1] * tf.shape(facts)[2]]) if need_tile else query  # [128,36]->[128,20*5*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*5*36]->[128,20,5,36]

    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [[128,20,5,36],[128,20,5,36],[128,20,5,36],[128,20,5,36]]->[128,20,5,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag, reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag, reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag, reuse=tf.AUTO_REUSE)  # [128,20,5,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [tf.shape(facts)[0], -1, 1, tf.shape(facts)[2]])  # [128,20,5,1]->[128,20,1,5]
    scores = d_layer_3_all  # [128,20,1,5]
    # Mask
    key_masks = tf.expand_dims(mask, 2)  # [128,20,5]->[128,20,1,5]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,20,1,5]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,20,1,5]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,20,1,5],[128,20,5,36]->[128,20,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1], tf.shape(facts)[2]])  # [128,20,5]
        output = facts * tf.expand_dims(scores, -1)  # [128,20,5,36],[128,20,5,1]->[128,20,5,36]
        output = tf.reshape(output, tf.shape(facts))  # [128,20,5,36]->[128,20,5,36]
    return output  # [128,20,1,36]or[128,20,5,36]


def din_attention_DIB_with_item(query, facts, mask, stag='null', mode='SUM', softmax_stag=1, need_tile=True):  # [128,20,36],[128,20,5,36],[128,20,5]
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20,5] 转换成tf.bool
    query = tf.layers.dense(query, facts.get_shape().as_list()[-1], activation=None, name=stag + 'din_attention_DIB_query')  # [128,36]->[128,36]
    queries = tf.tile(query, [1, tf.shape(facts)[2], 1]) if need_tile else query  # [128,20,36]->[128,20*5,36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*5,36]->[128,20,5,36]

    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [[128,20,5,36],[128,20,5,36],[128,20,5,36],[128,20,5,36]]->[128,20,5,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag, reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag, reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag, reuse=tf.AUTO_REUSE)  # [128,20,5,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [tf.shape(facts)[0], -1, 1, tf.shape(facts)[2]])  # [128,20,5,1]->[128,20,1,5]
    scores = d_layer_3_all  # [128,20,1,5]
    # Mask
    key_masks = tf.expand_dims(mask, 2)  # [128,20,5]->[128,20,1,5]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,20,1,5]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,20,1,5]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,20,1,5],[128,20,5,36]->[128,20,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1], tf.shape(facts)[2]])  # [128,20,5]
        output = facts * tf.expand_dims(scores, -1)  # [128,20,5,36],[128,20,5,1]->[128,20,5,36]
        output = tf.reshape(output, tf.shape(facts))  # [128,20,5,36]->[128,20,5,36]
    return output  # [128,20,1,36]or[128,20,5,36]


def din_attention_DIB_V2(queries, facts, mask, stag='null', mode='SUM', softmax_stag=1, need_tile=True):  # [128,18],[128,20,10,36],[128,20,10]
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    queries = tf.layers.dense(queries, facts.get_shape().as_list()[-1], activation=None, name=stag + 'din_attention_DIB_query')  # [128,18]->[128,36]
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [[128,20,10,36],[128,20,10,36],[128,20,10,36],[128,20,10,36]]->[128,20,10,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag, reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag, reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag, reuse=tf.AUTO_REUSE)  # [128,20,10,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [tf.shape(facts)[0], -1, 1, tf.shape(facts)[2]])  # [128,20,10,1]->[128,20,1,10]
    scores = d_layer_3_all  # [128,20,1,10]
    # Mask
    key_masks = tf.expand_dims(mask, 2)  # [128,20,10]->[128,20,1,10]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,20,1,10]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,20,1,10]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,20,1,10],[128,20,10,36]->[128,20,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1], tf.shape(facts)[2]])  # [128,20,10]
        output = facts * tf.expand_dims(scores, -1)  # [128,20,10,36],[128,20,10,1]->[128,20,10,36]
        output = tf.reshape(output, tf.shape(facts))  # [128,20,10,36]->[128,20,10,36]
    return output  # [128,20,1,36]or[128,20,10,36]


def din_attention_mask_top_k(query, facts, mask, stag='null', mode='SUM', softmax_stag=1, need_tile=True):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    queries = tf.tile(query, [1, tf.shape(facts)[1]]) if need_tile else query  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]

    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]->[128,144]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag, reuse=tf.AUTO_REUSE)  # 这里设置的是reuse=tf.AUTO_REUSE，即只要name一样就可以重复使用
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag, reuse=tf.AUTO_REUSE)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag, reuse=tf.AUTO_REUSE)  # [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,20,1]->[128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,20]->[128,1,20]    [128*50,20]->[128*50,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    scores_top_k, indices = tf.nn.top_k(scores, k=20)
    third_largest = scores_top_k[:, :, -1]
    scores_sub_kth = scores - tf.expand_dims(third_largest, -1)
    mask_except_top_k = tf.cast(tf.less(scores_sub_kth, 0), tf.float32)
    scores = tf.where(tf.equal(mask_except_top_k, 1), paddings, scores)

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,1,20]

    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [128,1,20],[128,20,36]->[128,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])  # [128,20]
        output = facts * tf.expand_dims(scores, -1)  # [128,20,36],[128,20,1]->[128,20,36]
        output = tf.reshape(output, tf.shape(facts))  # [128,20,36]->[128,20,36]
    return output  # [128,1,36]or[128,20,36]


def din_attention_with_context_origin(query, facts, context_embedding, mask, stag='null', mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    queries = tf.tile(query, [1, tf.shape(facts)[1]])  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]
    # --------------------------------相比原始DIN多了拼接context_embedding的过程--------------------------------
    if context_embedding is None:
        queries = queries
    else:
        queries = tf.concat([queries, context_embedding], axis=-1)  # [128,20,36],[128,20,36]->[128,20,72]
    queries = tf.layers.dense(queries, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_queries')  # [128,20,72]->[128,20,36] 需要保证units是int型而不是tensor
    # ---------------------------------------------------------------------------------------------
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [128,20,36],[128,20,36],[128,20,36],[128,20,36]->[128,20,44]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [128,20,44]->[128,20,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [128,20,80]->[128,20,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [128,20,40]-> [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_attention_with_context(query, facts, context_embedding, mask, stag='null', mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    query = tf.layers.dense(query, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_query')  # 将query的维度变得跟facts一样的维度
    queries = tf.tile(query, [1, tf.shape(facts)[1]])  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]
    # --------------------------------相比原始DIN多了拼接context_embedding的过程--------------------------------
    if context_embedding is None:
        facts_new = facts
    else:
        facts_new = tf.concat([facts, context_embedding], axis=-1)  # [128,20,36],[128,20,36]->[128,20,72]
    facts_new = tf.layers.dense(facts_new, units=facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_facts')  # [128,20,72]->[128,20,36] 需要保证units是int型而不是tensor
    # ---------------------------------------------------------------------------------------------
    din_all = tf.concat([queries, facts_new, queries - facts_new, queries * facts_new], axis=-1)  # [128,20,36],[128,20,36],[128,20,36],[128,20,36]->[128,20,44]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [128,20,44]->[128,20,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [128,20,80]->[128,20,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [128,20,40]-> [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)  # [128,1,20]

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [128,1,20]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts_new)  # [128,1,20]*[128,20,36]->[128,1,36]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])  # [128,1,20]
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_attention_with_context_with_user(query, facts, context_embedding, user_embedding, mask, stag='null', mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    query = tf.layers.dense(query, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_query')  # 将query的维度变得跟facts一样的维度
    queries = tf.tile(query, [1, tf.shape(facts)[1]])  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]
    # --------------------------------相比原始DIN多了拼接context_embedding的过程--------------------------------
    if context_embedding is None:
        facts_new = facts
    else:
        facts_new = tf.concat([facts, context_embedding], axis=-1)  # [128,20,36],[128,20,36]->[128,20,72]
    facts_new = tf.layers.dense(facts_new, units=facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_facts')  # [128,20,72]->[128,20,36] 需要保证units是int型而不是tensor
    # ---------------------------------------------------------------------------------------------
    user_embedding = tf.tile(tf.expand_dims(user_embedding, 1), [1, tf.shape(facts)[1], 1])
    din_all = tf.concat([user_embedding, queries, facts_new, queries - facts_new, queries * facts_new], axis=-1)  # [128,20,36],[128,20,36],[128,20,36],[128,20,36]->[128,20,44]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [128,20,44]->[128,20,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [128,20,80]->[128,20,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [128,20,40]-> [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts_new)  # [B, 1, H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_attention_with_context_test(query, facts, context_embedding, mask, stag='null', mode='SUM', softmax_stag=1):
    mask = tf.equal(mask, tf.ones_like(mask))  # [128,20] 转换成tf.bool
    query = tf.layers.dense(query, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_query')  # 将query的维度变得跟facts一样
    queries = tf.tile(query, [1, tf.shape(facts)[1]])  # [128,36]->[128,20*36]
    queries = tf.reshape(queries, tf.shape(facts))  # [128,20*36]->[128,20,36]
    # --------------------------------相比原始DIN多了拼接context_embedding的过程--------------------------------
    # if context_embedding is None:
    #     facts_new = facts
    # else:
    #     facts_new = tf.concat([facts, context_embedding], axis=-1)  # [128,20,36],[128,20,36]->[128,20,72]
    # facts_new = tf.layers.dense(facts_new, units=facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr_facts')  # [128,20,72]->[128,20,36] 需要保证units是int型而不是tensor
    # ---------------------------------------------------------------------------------------------
    if context_embedding is not None:
        din_all = tf.concat([queries, facts, context_embedding, queries - facts, queries * facts], axis=-1)  # [128,20,36],[128,20,36],[128,20,36],[128,20,36]->[128,20,44]
    else:
        din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [128,20,44]->[128,20,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [128,20,80]->[128,20,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [128,20,40]-> [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    key_masks = tf.expand_dims(mask, 1)  # [128,1,20]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [128,1,20]

    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, return_alphas=False, forCnn=False):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    mask = tf.equal(mask, tf.ones_like(mask))  # 转换成tf.bool类型
    facts_size = facts.get_shape().as_list()[-1]  # 36
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)  # [128,20,36],[128,20,36],[128,20,36],[128,20,36]->[128,20,44]
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)  # [128,20,44]->[128,20,80]
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)  # [128,20,80]->[128,20,40]
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)  # [128,20,40]-> [128,20,1]
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])  # [128,1,20]
    scores = d_layer_3_all  # [128,1,20]
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    if softmax_stag:  # Activation
        scores = tf.nn.softmax(scores)  # [B, 1, T]
    if mode == 'SUM':  # Weighted sum
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output


def self_multi_head_attn(inputs, num_units, num_heads, padding_mask=None, causality_mask_bool=False, dropout_rate=0, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units, name=name + "dense1")  # 先将传进来的过一个线性层，最后一维转换成num_units,[128,20,36]->[128,20,108]
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [128,20,108]->[[128,20,36],[128,20,36],[128,20,36]]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # Scaled Dot-Product Attention
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [512,20,9],[512,9,20]->[512,20,20]
    align = scores / (num_units ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, V_)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]
    # output linear
    outputs = tf.layers.dense(outputs, num_units, name=name + "dense2")  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,36]->[128,20,36]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)  # drop_out before residual and layernorm
    outputs += inputs  # Residual connection

    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)# [128,20,36]->[128,20,36]
    return outputs  # [128,20,36]


def self_multi_head_attn_context(inputs, num_units, num_heads, padding_mask=None, causality_mask_bool=True, dropout_rate=0, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units,[128,20,36]->[128,20,108]
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [128,20,108]->[[128,20,36],[128,20,36],[128,20,36]]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # Scaled Dot-Product Attention
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [512,20,9],[512,9,20]->[512,20,20]
    align = scores / (num_units ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, V_)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]
    # output linear
    outputs = tf.layers.dense(outputs, num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,36]->[128,20,36]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)  # drop_out before residual and layernorm
    return outputs  # [128,20,36]


def self_multi_head_attn_v1(inputs, num_units, num_heads, padding_mask=None, causality_mask_bool=True, dropout_rate=0, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units,[128,20,36]->[128,20,108]
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [128,20,108]->[[128,20,36],[128,20,36],[128,20,36]]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # Scaled Dot-Product Attention
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [512,20,9],[512,9,20]->[512,20,20]
    align = scores / (36 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # 创建一个与单个注意力分数矩阵同形状的全1矩阵
        causality_mask = tf.linalg.band_part(diag_val, 0, -1)  # 创建上三角掩码
        causality_mask = tf.tile(tf.expand_dims(causality_mask, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)  # 使用非常小的数填充掩码以避免在softmax中被选中
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # 应用掩码

    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, V_)  # [512,20,20],[512,20,9]->[512,20,9]
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # [[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[128,20,36]
    # output linear
    outputs = tf.layers.dense(outputs, num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,36]->[128,20,36]
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)  # drop_out before residual and layernorm
    outputs += inputs  # Residual connection
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # Normalize
    return outputs  # [128,20,36]


def self_multi_head_attn_v2(inputs, num_units, num_heads, padding_mask=None, causality_mask_bool=False, dropout_rate=0, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units,[128,20,36]->[128,20,108]
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [128,20,108]->[[128,20,36],[128,20,36],[128,20,36]]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # Scaled Dot-Product Attention
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [512,20,9],[512,9,20]->[512,20,20]
    align = scores / (36 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:  # [128,20]
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20],会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, V_)  # [512,20,20],[512,20,9]->[512,20,9]
    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += inputs  # 每一个头都加上inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    return outputss  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]

def cosine_similarity(tensor_a, tensor_b):  # [128,20,36],[128,20,36]
    """计算两个张量之间的余弦相似度，希望余弦相似度尽可能地小"""
    tensor_a_norm = tf.nn.l2_normalize(tensor_a, axis=-1)  # [128,20,36]->[128,20,36]
    tensor_b_norm = tf.nn.l2_normalize(tensor_b, axis=-1)  # [128,20,36]->[128,20,36]
    dot_product = tf.reduce_mean(tf.multiply(tensor_a_norm, tensor_b_norm), axis=-1)  # [128,20,36],[128,20,36]->[128,20,1]
    regularization = tf.square(dot_product)  # [128,20,1]->[128,20,1]
    return tf.reduce_mean(regularization)


def self_multi_head_attn_v2_with_loss(inputs, num_units, num_heads, padding_mask=None, causality_mask_bool=True, dropout_rate=0, name="", is_training=True, is_layer_norm=True):
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units,[128,20,36]->[128,20,108]
    Q, K, V = tf.split(Q_K_V, 3, -1)  # [128,20,108]->[[128,20,36],[128,20,36],[128,20,36]]

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)[128,20,36]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]->[512,20,9]
    # Scaled Dot-Product Attention
    scores = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [512,20,9],[512,9,20]->[512,20,20]
    align = scores / (36 ** 0.5)  # [512,20,20]

    outputs_scores = align
    # Apply padding mask
    if padding_mask is not None:
        padding_mask = tf.tile(tf.expand_dims(padding_mask, 1), [num_heads, tf.shape(inputs)[1], 1])  # [128,20]->[128,1,20]->[512,20,20]
        padding_mask_val = tf.ones_like(padding_mask) * (-2 ** 32 + 1)  # [512,20,20]
        outputs_scores = tf.where(tf.equal(padding_mask, 0), padding_mask_val, outputs_scores)  # [512,20,20]
    # Apply causality mask
    if causality_mask_bool:  # 后面的行为不能影响前面的行为（模拟RNN）
        diag_val = tf.ones_like(align[0, :, :])  # [20, 20]
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [20, 20]会将diag_val的上半部分变成0
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense()  # [20, 20] for tensorflow1.4
        causality_mask = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])  # [512,20,20]
        causality_mask_val = tf.ones_like(causality_mask) * (-2 ** 32 + 1)
        outputs_scores = tf.where(tf.equal(causality_mask, 0), causality_mask_val, outputs_scores)  # [512,20,20]
    outputs_scores = tf.nn.softmax(outputs_scores)  # [512,20,20]
    outputs_scores = tf.layers.dropout(outputs_scores, dropout_rate, training=is_training)  # [512,20,20]
    outputs = tf.matmul(outputs_scores, V_)  # [512,20,20],[512,20,9]->[512,20,9]

    # Restore shape
    outputs = tf.split(outputs, num_heads, axis=0)  # [512,20,9]->[[128,20,9],[128,20,9],[128,20,9],[128,20,9]]
    outputss = []
    for head_index, outputs_sub in enumerate(outputs):  # 下面每个头都使用不同的dense层
        outputs_sub = tf.layers.dense(outputs_sub, num_units)  # 先将传进来的过一个线性层，最后一维转换成num_units[128,20,9]->[128,20,36]
        outputs_sub = tf.layers.dropout(outputs_sub, dropout_rate, training=is_training)
        outputs_sub += inputs  # 每一个头都加上inputs
        if is_layer_norm:
            outputs_sub = layer_norm(outputs_sub, name=name + str(head_index))  # [128,20,36]每个头用的layer_norm层不一样
        outputss.append(outputs_sub)
    # 计算不同头输出之间的余弦距离作为辅助损失
    auxiliary_loss = 0
    num_heads = len(outputss)
    for i in range(1, num_heads):  # 这里避开了对第一个头进行计算
        for j in range(i + 1, num_heads):
            cos_similarity = cosine_similarity(outputss[i], outputss[j])  # [128,20,36],[128,20,36]
            auxiliary_loss += cos_similarity
    return outputss, tf.div(auxiliary_loss, tf.cast(tf.pow(num_heads, 2), tf.float32))  # [[128,20,36],[128,20,36],[128,20,36],[128,20,36]]


def soft_max_weighted_sum(align, value, key_masks, drop_out, is_training, future_binding=False):
    """
    :param align:           [batch_size, None, time]
    :param value:           [batch_size, time, units]
    :param key_masks:       [batch_size, None, time]
                            2nd dim size with align
    :param drop_out:
    :param is_training:
    :param future_binding:  TODO: only support 2D situation at present
    :return:                weighted sum vector
                            [batch_size, None, units]
    """
    # exp(-large) -> 0
    paddings = tf.fill(tf.shape(align), float('-inf'))
    # [batch_size, None, time]
    align = tf.where(key_masks, align, paddings)

    if future_binding:
        length = tf.reshape(tf.shape(value)[1], [-1])
        # [time, time]
        lower_tri = tf.ones(tf.concat([length, length], axis=0))
        # [time, time]
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
        # [batch_size, time, time]
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        # [batch_size, time, time]
        align = tf.where(tf.equal(masks, 0), paddings, align)

    # soft_max and dropout
    # [batch_size, None, time]
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, drop_out, training=is_training)
    # weighted sum
    # [batch_size, None, units]
    return tf.matmul(align, value)


def layer_norm(inputs, name, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)# [128,20,36]->[128,20,1]
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))# [128,20,36]

    params_shape = inputs.get_shape()[-1:] # 36
    gamma = tf.get_variable(name + 'gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable(name + 'beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs


def get_angles(pos, i, d_model):
    angle_rates = 1 / tf.pow(10000.0, (2 * tf.cast(i // 2, tf.float32)) / tf.cast(d_model, tf.float32))
    pos = tf.cast(pos, tf.float32)  # Ensure pos is also of type tf.float32
    return pos * angle_rates


def positional_encoding(position, d_model, new_axis=False):
    angle_rads = get_angles(tf.range(position)[:, tf.newaxis], tf.range(d_model)[tf.newaxis, :], d_model)
    sines = tf.sin(angle_rads[:, 0::2])
    cosines = tf.cos(angle_rads[:, 1::2])
    angle_rads = tf.concat([sines, cosines], axis=-1)
    pos_encoding = angle_rads[tf.newaxis, ...]
    if new_axis:
        pos_encoding = pos_encoding[tf.newaxis, ...]
    return pos_encoding


def mapping_to_k(sequence, seq_len, k=5):  # seq_len参数非常重要，因为它告诉函数每个序列中有多少个元素需要被逆序，而不是直接从后面逆序，所以这是从后面填充的原因
    reverse_seq = tf.reverse_sequence(input=sequence, seq_lengths=seq_len, seq_axis=1, batch_axis=0)  # [128,50,36],[[[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 0, 0]],[[1, 2, 3, 4], [5, 6, 7, 9], [9, 10, 11, 12]]]->[[[5, 6, 7, 8], [1, 2, 3, 4], [0, 0, 0, 0]],[[9, 10, 11, 12], [5, 6, 7, 9], [1, 2, 3, 4],  ]]
    reverse_seq_k = reverse_seq[:, :k, :]  # [128,k,36]
    seq_len_k = tf.clip_by_value(seq_len, 0, k)
    sequence_k = tf.reverse_sequence(input=reverse_seq_k, seq_lengths=seq_len_k, seq_axis=1, batch_axis=0)  # [128,k,36]，因为这里同样也有seq_lengths参数，所以旋转完之后填充的0也在后面
    mask = tf.sequence_mask(lengths=seq_len_k, maxlen=tf.shape(sequence_k)[1], dtype=tf.float32)  # [128,k]
    mask = tf.expand_dims(mask, -1)  # [128,k]-># [128,k,1]
    return sequence_k, seq_len_k, mask


def attention_net_v1(enc, sl, dec, num_units, num_heads, num_blocks, dropout_rate, is_training, reuse, scope='all', value=None):
    with tf.variable_scope(scope, reuse=reuse):
        dec = tf.expand_dims(dec, 1)  # [128,36]->[128,1,36]
        with tf.variable_scope("item_feature_group"):
            for i in range(num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    dec, att_vec = multihead_attention_v1(queries=dec, queries_length=tf.ones_like(dec[:, 0, 0], dtype=tf.int32), keys=enc, keys_length=sl, num_units=num_units, num_heads=num_heads, dropout_rate=dropout_rate, is_training=is_training, scope="vanilla_attention", value=value)
                    ## Feed Forward
                    dec = feedforward_v1(dec, num_units=[num_units // 2, num_units], scope="feed_forward", reuse=reuse)
        dec = tf.reshape(dec, [-1, num_units])
        return dec, att_vec


def multihead_attention_v1(queries, queries_length, keys, keys_length, num_units=None, num_heads=8, dropout_rate=0, is_training=True, scope="multihead_attention", reuse=None, first_n_att_weight_report=20, value=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      queries_length: A 1d tensor with shape of [N].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      keys_length:  A 1d tensor with shape of [N].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        Q = tf.layers.dense(queries, num_units, activation=None)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        if value is None:
            V = tf.layers.dense(keys, num_units, activation=None)  # (N, T_k, C)
        else:
            V = tf.layers.dense(value, num_units, activation=None)
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        from tensorflow.contrib import layers
        Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
        K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
        outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # (h*N, T_q, T_k)
        # Causality = Future blinding: No use, removed
        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)
        # Query Masking
        query_masks = tf.sequence_mask(queries_length, tf.shape(queries)[1], dtype=tf.float32)  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (h*N, T_q, T_k)
        # Attention vector
        att_vec = outputs
        # Dropouts
        # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # summary
        keys_masks_tmp = tf.reshape(tf.cast(key_masks, tf.float32), [-1, tf.shape(keys)[1]])
        defined_length = tf.constant(first_n_att_weight_report, dtype=tf.float32, name="%s_defined_length" % (scope))
        greater_than_define = tf.cast(tf.greater(tf.reduce_sum(keys_masks_tmp, axis=1), defined_length), tf.float32)
        greater_than_define_exp = tf.tile(tf.expand_dims(greater_than_define, -1), [1, tf.shape(keys)[1]])

        weight = tf.reshape(outputs, [-1, tf.shape(keys)[1]]) * greater_than_define_exp
        weight_map = tf.reshape(weight, [-1, tf.shape(queries)[1], tf.shape(keys)[1]])  # BxL1xL2
        greater_than_define_exp_map = tf.reshape(greater_than_define_exp, [-1, tf.shape(queries)[1], tf.shape(keys)[1]])  # BxL1xL2
        weight_map_mean = tf.reduce_sum(weight_map, 0) / (tf.reduce_sum(greater_than_define_exp_map, axis=0) + 1e-5)  # L1xL2
        report_image = tf.expand_dims(tf.expand_dims(weight_map_mean, -1), 0)  # 1xL1xL2x1
        tf.summary.image("%s_attention" % (scope), report_image[:, :first_n_att_weight_report, :first_n_att_weight_report, :])  # 1x10x10x1
        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        # Residual connection
        # outputs += queries
        # Normalize
        # outputs = normalize(outputs)  # (N, T_q, C)
    return outputs, att_vec


def feedforward_v1(inputs, num_units=[2048, 512], scope="feedforward", reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        # Residual connection
        outputs += inputs
        # Normalize
        # outputs = normalize(outputs)
    return outputs
