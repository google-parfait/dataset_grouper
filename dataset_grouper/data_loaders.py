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
"""Classes for loading datasets partitioned across groups."""

from typing import Optional, Union

from dataset_grouper import serialization
import tensorflow as tf
import tensorflow_datasets as tfds


SEQUENCE_FEATURES = {
    serialization.BYTES_FEATURE_NAME: tf.io.FixedLenSequenceFeature(
        shape=[], dtype=tf.string
    )
}
BYTES_FEATURE_INDEX = 1


class PartitionedDataset:
  """A dataset partitioned across groups, backed by TFRecords files.

  These TFRecords files are assumed to be generated via
  `tfds_pipelines.create_tfds_to_tfrecords_pipeline`.
  """

  def __init__(
      self,
      file_pattern: Union[str, list[str], tf.Tensor],
      tfds_features: Union[tfds.features.FeaturesDict, str],
  ):
    """Initializes a `PartitionedDataset`.

    Args:
      file_pattern: : A string, a list of strings, or a `tf.Tensor` of string
        type representing the wildcard patterns to be matched by the TFRecords
        files associated to partitioned dataset.
      tfds_features: Either a `tfds.features.FeaturesDict` representing the
        features of the non-partitioned TFDS dataset, or a string representing
        the name of this TFDS dataset. This must match the TFDS dataset used to
        partition the dataset.
    """
    self.file_pattern = file_pattern
    if isinstance(tfds_features, str):
      builder = tfds.builder(tfds_features)
      features_dict = builder.info.features
    else:
      features_dict = tfds_features
    self.features_dict = features_dict

    def decode_bytes(grouped_bytes: tf.Tensor) -> tf.Tensor:
      parsed_bytes = tf.io.parse_sequence_example(
          grouped_bytes, sequence_features=SEQUENCE_FEATURES
      )
      return parsed_bytes[BYTES_FEATURE_INDEX][serialization.BYTES_FEATURE_NAME]

    self.decode_bytes = decode_bytes

  def build_group_stream(
      self,
      buffer_size: Optional[int] = None,
      num_parallel_reads: Optional[int] = None,
      shuffle_files: bool = True,
      shuffle_seed: Optional[int] = None,
  ) -> tf.data.Dataset:
    """Builds a `tf.data.Dataset` that yields group-level `tf.data.Datasets`.

    Each of the yielded datasets contain the examples held by one the groups
    formed in the original partition of the dataset.

    Args:
      buffer_size: An optional 64-bit integer representing the number of bytes
        in the TFRecords read buffer. If set to `None`, this is inferred
        automatically. See `tf.data.TFRecordDataset` for more information.
      num_parallel_reads: An optional 64-bit integer representing the number of
        TFRecords files to read in parallel. If greater than one, the records
        are interleaved and read in parallel. If set to one, the files are read
        sequentially. If set to `None`, this will be set via `tf.data.AUTOTUNE`.
      shuffle_files: Whether to shuffle the file names when reading them.
        Defaults to `True`.
      shuffle_seed: If `shuffle_files=True` and this is not set to `None`, then
        this seed is used to seed the filename shuffling.

    Returns:
      A `tf.data.Dataset`.
    """
    filenames = tf.data.Dataset.list_files(
        self.file_pattern, shuffle=shuffle_files, seed=shuffle_seed
    )

    def serialized_tensor_to_dataset(
        serialized_tensor: tf.Tensor,
    ) -> tf.data.Dataset:
      decoded_tensor = self.decode_bytes(serialized_tensor)
      bytes_ds = tf.data.Dataset.from_tensor_slices(decoded_tensor)
      # This takes the decoded examples, and applies the appropriate TFDS
      # deserialization on each example to coerce them into their original form,
      # matching `self.features_dict`.
      tfds_ds = bytes_ds.map(
          self.features_dict.deserialize_example,
          num_parallel_calls=tf.data.AUTOTUNE,
      )
      return tfds_ds

    if num_parallel_reads is None:
      num_parallel_reads = tf.data.AUTOTUNE
    serialized_tensor_tensor_stream = tf.data.TFRecordDataset(
        filenames,
        buffer_size=buffer_size,
        num_parallel_reads=num_parallel_reads,
    )
    return serialized_tensor_tensor_stream.map(
        serialized_tensor_to_dataset, num_parallel_calls=tf.data.AUTOTUNE
    )
