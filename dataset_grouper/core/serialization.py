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

from collections.abc import Generator, Iterable

from dataset_grouper.core import tf_utils
from dataset_grouper.core import types
import tensorflow as tf
import tensorflow_datasets as tfds

# This is a general protobuf limit size.
BYTES_LIMIT = 2e9
BYTES_FEATURE_NAME = 'serialized_bytes'


def _create_sequence_example(
    bytes_list=list[bytes],
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


def _bytes_limited_generator(
    examples: Iterable[types.Example],
    bytes_limit: float = BYTES_LIMIT,
) -> Generator[types.Example, None, None]:
  """Creates an iterator that only yields examples up to some bytes limit."""
  bytes_sum = 0.0
  for ex in examples:
    # This is probably an over-estimate of the serialized tensor size (as it
    # doesn't account for variable-length coding) but it does not include the
    # protobuf overhead. We would likely need to change this call if we
    # encounter problems when pushing the examples into a proto.
    num_bytes = tf_utils.get_tensor_byte_size(ex)
    bytes_sum += num_bytes
    if bytes_sum >= bytes_limit:
      return
    yield ex


def sequence_example_from_features_dict(
    examples: Iterable[types.Example],
    features_dict: tfds.features.FeaturesDict,
) -> tf.train.SequenceExample:
  """Creates a `tf.train.SequenceExample` from TFDS examples."""
  # `FeaturesDict.serialize_example` requires numpy-like structures
  serialized_examples: list[bytes] = []
  for example in _bytes_limited_generator(examples):
    numpy_example = tfds.as_numpy(example)
    try:
      serialized_example = features_dict.serialize_example(numpy_example)
    except KeyError as serialization_error:
      raise KeyError(
          'Found a mismatch between the provided features_dict and an example.'
          ' Please make sure that features_dict matches the structure of *all*'
          ' examples being serialized.'
      ) from serialization_error
    serialized_examples.append(serialized_example)
  return _create_sequence_example(serialized_examples)
