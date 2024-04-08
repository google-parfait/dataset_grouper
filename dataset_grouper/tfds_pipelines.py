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
"""Beam pipelines that partition datasets from TFDS."""

from typing import Optional

import apache_beam as beam
from dataset_grouper import beam_transforms
from dataset_grouper import types
import tensorflow as tf
import tensorflow_datasets as tfds


def tfds_to_tfrecords(
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    file_path_prefix: str,
    get_key_fn: types.GetKeyFn,
    file_name_suffix: str = '',
    num_shards: Optional[int] = None,
) -> beam.Pipeline:
  """Builds a Beam pipeline that partitions a TFDS partitions to TFRecords.

  The pipeline will partition the dataset across groups. Each group corresponds
  to a distinct output of the `get_key_fn`, which is applied to each tensor of
  the TFDS dataset specified. The TFDS examples for each group are serialized
  in a single `tf.train.SequenceExample`, and written to TFRecords.

  Args:
    dataset_builder: A `tfds.core.DatasetBuilder` to get examples from.
    split: Which split of the dataset should be used.
    file_path_prefix: The file path to write to. The TFRecords files written
      will begin with this prefix, followed by a shard identifier, and a common
      extension (if `file_name_suffix` is provided).
    get_key_fn: A function that takes as input an example from the TFDS dataset
      specified, and that returns bytes identifying the group it belongs to.
    file_name_suffix: A common suffix for the files written.
    num_shards: The number of files (shards) used for output. If not set, the
      number of shards will be automatically set.

  Returns:
    A `beam.Pipeline`.
  """
  features_dict = dataset_builder.info.features

  def pipeline(root) -> None:
    examples = root | 'ReadTFDS' >> tfds.beam.ReadFromTFDS(
        dataset_builder, split=split
    )
    keyed_sequence_examples = beam_transforms.to_keyed_sequence_examples(
        examples, get_key_fn, features_dict
    )
    sequence_examples = keyed_sequence_examples | 'RemoveKey' >> beam.Map(
        lambda x: x[1]
    )
    _ = (
        sequence_examples
        | 'WriteTFRecords'
        >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix,
            coder=beam.coders.ProtoCoder(tf.train.SequenceExample),
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
        )
    )

  return pipeline


def tfds_group_counts(
    dataset_builder: tfds.core.DatasetBuilder,
    split: str,
    file_path_prefix: str,
    get_key_fn: types.GetKeyFn,
    file_name_suffix: str = '',
    num_shards: Optional[int] = None,
    delimiter: str = ',',
) -> beam.Pipeline:
  """Builds a Beam pipeline that computes various statistics about the dataset.

  This beam pipeline will write to (possibly multiple) text files. Each line
  represents a distinct group (where groups are possible outputs of
  `get_key_fn`). Each line will be a of the form:
  ```
  group_id,num_examples,num_bytes,num_words
  ```
  where `group_id` is some output of `get_key_fn` on the dataset, `num_examples`
  is the total number of examples that belong to this group, `num_bytes` are
  the total number of bytes of examples in this group, and `num_words` is the
  total number of words in the concatenation of all examples in the group. We
  only consider string-valued features when computing this statistic, and assume
  that words are separated by ' '.

  Note that you can delimit statistics by something other than a comma via the
  `delimiter` argument.

  Args:
    dataset_builder: A `tfds.core.DatasetBuilder`.
    split: Which split of the dataset should be used.
    file_path_prefix: The file path to write to. The output files written will
      begin with this prefix, followed by a shard identifier, and a common
      extension (if `file_name_suffix` is provided).
    get_key_fn: A function that takes as input an example from the TFDS dataset
      specified, and that returns bytes identifying the group it belongs to.
    file_name_suffix: A common suffix for the files written.
    num_shards: The number of files (shards) used for output. If not set, the
      number of shards will be automatically set.
    delimiter: A string representing the delimiter to use when writing the
      statistics of a group.

  Returns:
    A `beam.Pipeline`.
  """
  header = 'group_id,num_examples,num_bytes,num_words'

  def pipeline(root) -> None:
    examples = root | 'ReadTFDS' >> tfds.beam.ReadFromTFDS(
        dataset_builder, split=split
    )
    formatted_group_counts = beam_transforms.compute_group_counts(
        examples, get_key_fn, delimiter=delimiter
    )
    _ = formatted_group_counts | 'WriteToText' >> beam.io.WriteToText(
        file_path_prefix=file_path_prefix,
        file_name_suffix=file_name_suffix,
        num_shards=num_shards,
        header=header,
    )

  return pipeline
