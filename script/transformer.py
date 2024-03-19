# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import re

import six
import tensorflow as tf
from tensorflow.contrib import layers


# from tensorflow.contrib.framework.python.ops import variables as contrib_variables


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
      input_tensor: float Tensor to perform activation.

    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def dropout(input_tensor, dropout_prob):
    """Perform dropout.

    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).

    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def attention_layer(from_tensor, to_tensor, attention_mask=None, num_attention_heads=1, size_per_head=512,
                    query_act=None, key_act=None, value_act=None,
                    attention_probs_dropout_prob=0.0, do_return_2d_tensor=False,
                    batch_size=None, from_seq_length=None, to_seq_length=None,
                    variables_collections=None, outputs_collections=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention is all you Need".
    If `from_tensor` and `to_tensor` are the same, then this is self-attention.
    Each timestep in `from_tensor` attends to the corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and `to_tensor` into "key" and "value" tensors.
    These are (effectively) a list of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are softmaxed to obtain attention probabilities.
    The value tensors are then interpolated by these probabilities, then concatenated back to a single tensor and returned.

    In practice, the multi-headed attention are done with transposes and reshapes rather than actual separate tensors.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length, from_width].
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size, from_seq_length, to_seq_length].
      The values should be 1 or 0. The attention scores will effectively be set to -infinity for any positions in the mask that are 0,
      and will be unchanged for positions that are 1.
      num_attention_heads: int. Number of attention heads.
      size_per_head: int. Size of each attention head.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the attention probabilities.
      initializer_range: float. Range of the weight initializer.
      do_return_2d_tensor: bool. If True, the output will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head].
      If False, the output will be of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
      batch_size: (Optional) int. If the input is 2D, this might be the batch size of the 3D version of the `from_tensor` and `to_tensor`.
      from_seq_length: (Optional) If the input is 2D, this might be the seq length of the 3D version of the `from_tensor`.
      to_seq_length: (Optional) If the input is 2D, this might be the seq length of the 3D version of the `to_tensor`.

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads * size_per_head].
      (If `do_return_2d_tensor` is true, this will be of shape [batch_size * from_seq_length, num_attention_heads * size_per_head]).

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width):
        output_tensor = tf.reshape(input_tensor, [batch_size, seq_length, num_attention_heads, width])
        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError("When passing in rank 2 tensors to attention_layer, the values for `batch_size`, `from_seq_length`, and `to_seq_length' must all be specified.")

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = layers.fully_connected(from_tensor_2d, num_attention_heads * size_per_head, activation_fn=query_act, scope="query", variables_collections=variables_collections, outputs_collections=outputs_collections)

    # `key_layer` = [B*T, N*H]
    key_layer = layers.fully_connected(to_tensor_2d, num_attention_heads * size_per_head, activation_fn=key_act, scope="key", variables_collections=variables_collections, outputs_collections=outputs_collections)

    # `value_layer` = [B*T, N*H]
    value_layer = layers.fully_connected(to_tensor_2d, num_attention_heads * size_per_head, activation_fn=value_act, scope="value", variables_collections=variables_collections, outputs_collections=outputs_collections)

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder
    # Normalize the attention scores to probabilities.`attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)
    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(value_layer, [batch_size, to_seq_length, num_attention_heads, size_per_head])
    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)
    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(context_layer, [batch_size, from_seq_length, num_attention_heads * size_per_head])
    return context_layer


def transformer_model(input_tensor, attention_mask=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
                      intermediate_act_fn=gelu, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, do_return_all_layers=False,
                      scope="transformer", reuse=None, variables_collections=None, outputs_collections=None):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".
    This is almost an exact implementation of the original Transformer encoder.
    See the original paper:
    https://arxiv.org/abs/1706.03762
    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py
    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length, seq_length],
      with 1 for positions that can be attended to and 0 in positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated normal).
      do_return_all_layers: Whether to also return all layers or just the final layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    with tf.variable_scope(scope, reuse=reuse):
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, num_attention_heads))

        attention_head_size = int(hidden_size / num_attention_heads)
        input_shape = get_shape_list(input_tensor, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]

        # The Transformer performs sum residuals on all layers so the input needs to be the same as the hidden size.
        if input_width != hidden_size:
            raise ValueError("The width of the input tensor (%d) != hidden size (%d)" % (input_width, hidden_size))

        # We keep the representation as a 2D tensor to avoid re-shaping it back and
        # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
        # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
        # help the optimizer.
        prev_output = reshape_to_matrix(input_tensor)

        all_layer_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = attention_layer(from_tensor=layer_input, to_tensor=layer_input, attention_mask=attention_mask,
                                                         num_attention_heads=num_attention_heads, size_per_head=attention_head_size, attention_probs_dropout_prob=attention_probs_dropout_prob,
                                                         do_return_2d_tensor=True, batch_size=batch_size, from_seq_length=seq_length, to_seq_length=seq_length, variables_collections=variables_collections, outputs_collections=outputs_collections)
                        attention_heads.append(attention_head)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)

                    # Run a linear projection of `hidden_size` then add a residual with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = layers.fully_connected(attention_output, hidden_size, activation_fn=None, variables_collections=variables_collections, outputs_collections=outputs_collections)
                        attention_output = dropout(attention_output, hidden_dropout_prob)
                        # attention_output = layer_norm(attention_output + layer_input)
                        attention_output = attention_output + layer_input

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = layers.fully_connected(attention_output, intermediate_size, activation_fn=intermediate_act_fn, variables_collections=variables_collections, outputs_collections=outputs_collections)

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    layer_output = layers.fully_connected(intermediate_output, hidden_size, activation_fn=None, variables_collections=variables_collections, outputs_collections=outputs_collections)
                    layer_output = dropout(layer_output, hidden_dropout_prob)
                    # layer_output = layer_norm(layer_output + attention_output)
                    layer_output = layer_output + attention_output
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)
        final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" % (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
