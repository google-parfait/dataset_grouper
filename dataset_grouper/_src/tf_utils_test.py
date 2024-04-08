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
"""Tests for tf_utils."""

from absl.testing import parameterized
from dataset_grouper._src import tf_utils
import tensorflow as tf


class CountBytesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('float32', tf.float32),
      ('int16', tf.int16),
      ('float16', tf.float16),
      ('int64', tf.int64),
      ('float64', tf.float64),
      ('uint8', tf.uint8),
  )
  def test_get_single_tensor_byte_size_non_string(self, tensor_type):
    shape = [3, 4, 5]
    num_entries = 3 * 4 * 5
    tensor = tf.ones(shape=shape, dtype=tensor_type)
    actual_result = tf_utils._get_single_tensor_byte_size(tensor)
    expected_result = num_entries * tensor_type.size
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('length_1', 1),
      ('length_2', 2),
      ('length_3', 3),
      ('length_100', 100),
  )
  def test_get_single_tensor_byte_size_string(self, string_length):
    tensor = tf.constant('a' * string_length, dtype=tf.string)
    actual_result = tf_utils._get_single_tensor_byte_size(tensor)
    self.assertEqual(actual_result, string_length)

  def test_get_tensor_byte_size(self):
    tensors = [
        tf.ones(shape=(3, 4), dtype=tf.int32),
        tf.ones(shape=(5,), dtype=tf.uint8),
        tf.constant('a' * 7, dtype=tf.string),
    ]
    structure = {
        'a': {'a0': tensors[0], 'a1': tensors[1]},
        'b': tensors[2],
    }
    structure_size = tf_utils.get_tensor_byte_size(structure)
    tensor_sizes = [
        tf_utils._get_single_tensor_byte_size(tensor) for tensor in tensors
    ]
    expected_structure_size = sum(tensor_sizes)
    self.assertEqual(structure_size, expected_structure_size)
    expected_structure_size = 12 * 4 + 5 * 1 + 7
    self.assertEqual(structure_size, expected_structure_size)


class CountWordsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('int32', tf.int32),
      ('float32', tf.float32),
      ('int16', tf.int16),
      ('float16', tf.float16),
      ('int64', tf.int64),
      ('float64', tf.float64),
      ('uint8', tf.uint8),
  )
  def test_get_single_tensor_num_words_non_string(self, tensor_type):
    shape = [3, 4, 5]
    tensor = tf.ones(shape=shape, dtype=tensor_type)
    actual_result = tf_utils._get_single_tensor_num_words(tensor)
    expected_result = tf.constant(0, dtype=tf.int64)
    self.assertEqual(actual_result, expected_result)

  @parameterized.named_parameters(
      ('1_word', 1),
      ('2_words', 2),
      ('3_words', 3),
      ('100_words', 100),
  )
  def test_get_single_tensor_num_words_string(self, num_words):
    tensor = tf.constant('foo ' * num_words, dtype=tf.string)
    actual_result = tf_utils._get_single_tensor_num_words(tensor)
    expected_result = tf.constant(num_words, dtype=tf.int64)
    self.assertEqual(actual_result, expected_result)

  def test_get_tensor_byte_size(self):
    tensors = [
        tf.ones(shape=(3, 4), dtype=tf.int32),
        tf.ones(shape=(5,), dtype=tf.uint8),
        tf.constant('this has four words', dtype=tf.string),
    ]
    structure = {
        'a': {'a0': tensors[0], 'a1': tensors[1]},
        'b': tensors[2],
    }
    total_num_words = tf_utils.get_tensor_num_words(structure)
    num_words = [
        tf_utils._get_single_tensor_num_words(tensor) for tensor in tensors
    ]
    expected_total_num_words = sum(num_words)
    self.assertEqual(total_num_words, expected_total_num_words)
    expected_total_num_words = 4
    self.assertEqual(total_num_words, expected_total_num_words)


if __name__ == '__main__':
  tf.test.main()
