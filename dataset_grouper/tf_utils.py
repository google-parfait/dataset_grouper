# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorFlow utils."""

from dataset_grouper import types
import tensorflow as tf


def _get_single_tensor_byte_size(tensor: types.Tensor) -> tf.Tensor:
  """Computes the size (in bytes) of a single tensor."""
  dtype = tensor.dtype
  if dtype is tf.string:
    byte_size = tf.strings.length(tensor)
  else:
    shape = tensor.shape
    total_dim = tf.math.reduce_prod(shape)
    byte_size = total_dim * dtype.size
  # We use a 64-bit integer here, so that when summing up byte sizes, we
  # avoid overflow.
  return tf.cast(byte_size, tf.int64)


def get_tensor_byte_size(nested_tensor: types.NestedTensor) -> int:
  """Computes the size (in bytes) of a structure of tensors."""
  nested_byte_size = tf.nest.map_structure(
      _get_single_tensor_byte_size, nested_tensor
  )
  flat_byte_size = tf.nest.flatten(nested_byte_size)
  total_byte_size = tf.math.reduce_sum(flat_byte_size)
  return int(total_byte_size.numpy())


def _get_single_tensor_num_words(tensor: types.Tensor) -> tf.Tensor:
  if tensor.dtype != tf.string:
    return tf.constant(0, dtype=tf.int64)
  else:
    text = tensor.numpy()
    split_text = text.decode('utf-8').strip().split(' ')
    num_words = len(split_text)
    return tf.constant(num_words, dtype=tf.int64)


def get_tensor_num_words(nested_tensor: types.NestedTensor) -> int:
  nested_num_words = tf.nest.map_structure(
      _get_single_tensor_num_words, nested_tensor
  )
  flat_num_words = tf.nest.flatten(nested_num_words)
  total_num_words = tf.math.reduce_sum(flat_num_words)
  return int(total_num_words.numpy())
