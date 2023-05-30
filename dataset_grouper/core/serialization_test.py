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
"""Tests for serialization.py."""

from absl.testing import parameterized
from dataset_grouper.core import serialization
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class SerializationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('len1_list', [b'01']),
      ('len3_list', [b'01', b'02', b'03']),
      ('len0_list', []),
  )
  def test_create_sequence_example_has_expected_fields(self, input_bytes):
    sequence_example = serialization._create_sequence_example(input_bytes)
    self.assertIsInstance(sequence_example, tf.train.SequenceExample)
    self.assertTrue(sequence_example.HasField('feature_lists'))
    feature_list = sequence_example.feature_lists.feature_list
    expected_keys = [serialization.BYTES_FEATURE_NAME]
    self.assertEqual(list(feature_list.keys()), expected_keys)
    bytes_list = feature_list[serialization.BYTES_FEATURE_NAME]
    self.assertLen(bytes_list.feature, len(input_bytes))

  def test_bytes_limited_generator_with_limit(self):
    # 32-bit integers are made up of 4 bytes each
    examples = [tf.constant(i, dtype=tf.int32) for i in range(5)]
    bytes_limited_generator = serialization._bytes_limited_generator(
        examples, bytes_limit=9
    )
    output = list(bytes_limited_generator)
    expected_output = [tf.constant(0), tf.constant(1)]
    self.assertAllEqual(output, expected_output)

  def test_bytes_limited_generator_without_limit(self):
    # 32-bit integers are made up of 4 bytes each
    examples = [tf.constant(i, dtype=tf.int32) for i in range(5)]
    bytes_limited_generator = serialization._bytes_limited_generator(
        examples, bytes_limit=10000
    )
    output = list(bytes_limited_generator)
    self.assertAllEqual(output, examples)

  def test_sequence_example_with_compatible_features_dict(self):
    features_dict = tfds.features.FeaturesDict({'a': np.int32, 'b': np.float32})
    examples = [
        {'a': tf.constant(1), 'b': tf.constant(1.0)},
        {'a': tf.constant(2), 'b': tf.constant(2.0)},
        {'a': tf.constant(3), 'b': tf.constant(3.0)},
    ]
    sequence_example = serialization.sequence_example_from_features_dict(
        examples, features_dict
    )
    self.assertIsInstance(sequence_example, tf.train.SequenceExample)

  def test_sequence_example_with_incompatible_features_dict(self):
    features_dict = tfds.features.FeaturesDict({'a': np.int32, 'b': np.float32})
    examples = [
        {'a': tf.constant(1), 'b': tf.constant(1.0)},
        {'a': tf.constant(2), 'b': tf.constant(2.0), 'c': tf.constant('2.0')},
        {'a': tf.constant(3), 'b': tf.constant(3.0)},
    ]
    with self.assertRaisesRegex(
        KeyError, 'Found a mismatch between the provided features_dict'
    ):
      serialization.sequence_example_from_features_dict(examples, features_dict)


if __name__ == '__main__':
  tf.test.main()
