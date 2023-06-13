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
"""Utilities for tokenizing and preprocessing text datasets."""

import collections
from collections.abc import Callable
from typing import Union

import jax.numpy as jnp
import tensorflow as tf
import tensorflow_text as text


OrderedDict = collections.OrderedDict
RawExampleType = OrderedDict[str, Union[str, tf.Tensor]]
ProcessedExampleType = OrderedDict[str, tf.Tensor]


def get_placeholder_batch(
    batch_size: int, sequence_length: int
) -> OrderedDict[str, Union[jnp.ndarray, OrderedDict[str, jnp.ndarray]]]:
  """Creates a placeholder batch in JAX to initialize the model."""
  input_ids = jnp.ones([batch_size, sequence_length - 1], dtype=jnp.int32)
  input_paddings = jnp.zeros([batch_size, sequence_length - 1], dtype=jnp.int32)
  class_ids = jnp.ones([batch_size, sequence_length - 1], dtype=jnp.int32)
  class_weights = jnp.ones([batch_size, sequence_length - 1], dtype=jnp.float32)
  labels = OrderedDict(
      [('class_ids', class_ids), ('class_weights', class_weights)]
  )
  return OrderedDict([
      ('input_ids', input_ids),
      ('input_paddings', input_paddings),
      ('labels', labels),
  ])


def build_preprocess_fn(
    tf_tokenizer: text.BertTokenizer,
    num_epochs: int,
    batch_size: int,
    max_elements: int,
    sequence_length: int,
    dataset_text_field: str,
    shuffle: bool = True,
):
  """Return a function to preprocess each group's dataset."""
  tokenize_fn = build_tokenize_function(tf_tokenizer, dataset_text_field)
  group_and_split_input_target_fn = build_group_and_split_function(
      sequence_length
  )

  def preprocess_fn(input_dataset: tf.data.Dataset) -> tf.data.Dataset:
    if shuffle:
      input_dataset = input_dataset.shuffle(max_elements)
    processed_dataset = (
        input_dataset.map(tokenize_fn)
        .apply(tf.data.experimental.dense_to_ragged_batch(max_elements))
        # We are creating a large batch; we must be careful not to OOM.
        .map(
            group_and_split_input_target_fn,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .unbatch()
        .repeat(num_epochs)
        .take(max_elements)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x: OrderedDict([('x', x), ('y', ())]))
    )
    return processed_dataset

  return preprocess_fn


def build_tokenize_function(
    tf_tokenizer: text.BertTokenizer,
    dataset_text_field: str,
) -> Callable[[RawExampleType], ProcessedExampleType]:
  """Return a function that can tokenize text."""

  def tokenize_fn(example):
    tokenized_text = tf_tokenizer.tokenize(
        example[dataset_text_field]
    ).merge_dims(-2, -1)
    # The tokenizer returns a tf.RaggedTensor of shape (1, None).
    tokenized_text = tokenized_text[0]  # tf.Tensor of shape (num_tokens,)
    # Padding is a 0/1 tensor, with 1 denoting padding.
    # Since we do not have any padding yet, we return all zeros.
    padding = tf.zeros_like(tokenized_text)
    # The EOS (= PAD) token is 0 for our tokenizer: this is hard-coded.
    tokenized_text = tf.concat(
        [tokenized_text, tf.zeros(1, dtype=tf.int64)], axis=0
    )
    padding = tf.concat(
        [padding, tf.ones(1, dtype=tf.int64)], axis=0
    )  # Add a 1 to denote the padding
    return OrderedDict([('input_ids', tokenized_text), ('padding', padding)])

  return tokenize_fn


def build_group_and_split_function(
    sequence_length: int,
) -> Callable[[ProcessedExampleType], ProcessedExampleType]:
  """Return a function that can group all examples and split input/target."""

  def group_and_split_input_target_fn(examples):
    # Our input is dict with keys 'input_ids', 'padding',
    # where each is a RaggedTensor.
    # We concatenate all the texts and split them into BLOCK_SIZE-sized chunks.
    block_size = tf.constant(sequence_length + 1)
    # We add 1 because we lose 1 due to the right shift later.
    input_ids = examples['input_ids'].flat_values
    padding = examples['padding'].flat_values
    # We drop the small remainder instead of adding some padding.
    # NOTE: need to use tf.shape(tensor) rather than tensor.shape in graph mode.
    total_length = (
        tf.truncatediv(tf.shape(input_ids)[0], block_size) * block_size
    )
    # Split by chunks of max_len: shape = (n, block_size).
    input_ids = tf.reshape(input_ids[:total_length], (-1, block_size))
    padding = tf.reshape(padding[:total_length], (-1, block_size))
    # Labels: we need to right shift the input_ids
    class_ids = tf.identity(input_ids)[:, 1:]  # right shift on a copy
    input_ids = input_ids[:, :-1]  # drop the last token
    padding = padding[:, :-1]
    # Assign a class weight of 0 to prediction of the padding tokens, else 1.
    class_weights = tf.cast(tf.math.not_equal(class_ids, 0), dtype=tf.float32)
    labels = OrderedDict(
        [('class_ids', class_ids), ('class_weights', class_weights)]
    )
    return OrderedDict([
        ('input_ids', input_ids),
        ('input_paddings', padding),
        ('labels', labels),
    ])

  return group_and_split_input_target_fn
