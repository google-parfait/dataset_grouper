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
from dataset_grouper._src import serialization
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class SerializationTest(tf.test.TestCase, parameterized.TestCase):

  def test_serialize_tfds_example_with_compatible_features_dict(self):
    example = {'a': tf.constant(1), 'b': tf.constant(1.0)}
    features_dict = tfds.features.FeaturesDict({'a': np.int32, 'b': np.float32})
    serialized_example = serialization.serialize_tfds_example(
        example, features_dict
    )
    self.assertIsInstance(serialized_example, bytes)

  def test_serialize_tfds_example_with_incompatible_features_dict(self):
    example = {
        'a': tf.constant(2),
        'b': tf.constant(2.0),
        'c': tf.constant('2.0'),
    }
    features_dict = tfds.features.FeaturesDict({'a': np.int32, 'b': np.float32})
    with self.assertRaisesRegex(
        KeyError, 'Found a mismatch between the provided features_dict'
    ):
      serialization.serialize_tfds_example(example, features_dict)

  @parameterized.named_parameters(
      ('len1_list', [b'01']),
      ('len3_list', [b'01', b'02', b'03']),
      ('len0_list', []),
  )
  def test_create_sequence_example_has_expected_fields(self, input_bytes):
    sequence_example = serialization.create_sequence_example(input_bytes)
    self.assertIsInstance(sequence_example, tf.train.SequenceExample)
    self.assertTrue(sequence_example.HasField('feature_lists'))
    feature_list = sequence_example.feature_lists.feature_list
    expected_keys = [serialization.BYTES_FEATURE_NAME]
    self.assertEqual(list(feature_list.keys()), expected_keys)
    bytes_list = feature_list[serialization.BYTES_FEATURE_NAME]
    self.assertLen(bytes_list.feature, len(input_bytes))


if __name__ == '__main__':
  tf.test.main()
