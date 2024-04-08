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
"""Utilities for serializing examples."""

from dataset_grouper import types
import tensorflow as tf
import tensorflow_datasets as tfds

BYTES_FEATURE_NAME = 'serialized_bytes'


def serialize_tfds_example(
    example: types.Example, features_dict: tfds.features.FeaturesDict
) -> bytes:
  """Serializes a TFDS example, and computes the number of bytes.

  Args:
    example: A nested structure of `tf.Tensor`s.
    features_dict: A `tfds.features.FeaturesDict` matching the structure of
      `example`.

  Returns:
    A `bytes` object.

  Raises:
    KeyError: If `example` and `features_dict` do not have matching structure.
  """
  numpy_example = tfds.as_numpy(example)
  try:
    serialized_example = features_dict.serialize_example(numpy_example)
  except KeyError as serialization_error:
    raise KeyError(
        'Found a mismatch between the provided features_dict and an example.'
        ' Please make sure that features_dict matches the structure of *all*'
        ' examples being serialized.'
    ) from serialization_error
  return serialized_example


def create_sequence_example(
    bytes_list: list[bytes],
) -> tf.train.SequenceExample:
  bytes_features = [
      tf.train.Feature(bytes_list=tf.train.BytesList(value=[a]))
      for a in bytes_list
  ]
  feature_list = tf.train.FeatureList(feature=bytes_features)
  feature_lists = tf.train.FeatureLists(
      feature_list={BYTES_FEATURE_NAME: feature_list}
  )
  return tf.train.SequenceExample(feature_lists=feature_lists)
