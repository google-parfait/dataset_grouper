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
"""Transformations for loading, grouping, and serializing partitioned data."""

from collections.abc import Iterable

import apache_beam as beam
from dataset_grouper import count_utils
from dataset_grouper import serialization
from dataset_grouper import types
import tensorflow as tf
import tensorflow_datasets as tfds


# This is a general protobuf limit size.
BYTES_LIMIT = 2_000_000_000
MergeAccumulator = tuple[list[bytes], int]


class MergeWithLimitFn(beam.CombineFn):
  """Creates a list of bytes objects whose total size is at most `BYTES_LIMIT`."""

  def __init__(self, limit: int = BYTES_LIMIT):
    super().__init__()
    self._limit = limit

  def create_accumulator(self) -> MergeAccumulator:
    accum: list[bytes] = []
    return accum, 0

  def add_input(
      self, accumulator: MergeAccumulator, element: bytes
  ) -> MergeAccumulator:
    total_bytes_list, total_bytes = accumulator
    num_bytes = len(element)
    if total_bytes + num_bytes >= self._limit:
      return accumulator
    else:
      total_bytes += num_bytes
      total_bytes_list.append(element)
      return total_bytes_list, total_bytes

  def merge_accumulators(
      self, accumulators: Iterable[MergeAccumulator]
  ) -> MergeAccumulator:
    total_bytes_list: list[bytes] = []
    total_bytes = 0
    for accumulator in accumulators:
      bytes_list, num_bytes = accumulator
      if total_bytes + num_bytes < self._limit:
        total_bytes += num_bytes
        total_bytes_list += bytes_list
      else:
        elements, _ = accumulator
        for element in elements:
          num_bytes = len(element)
          if total_bytes + num_bytes >= self._limit:
            return total_bytes_list, total_bytes
          else:
            total_bytes_list.append(element)
            total_bytes += num_bytes

    return total_bytes_list, total_bytes

  def extract_output(self, accumulator: MergeAccumulator) -> list[bytes]:
    examples, _ = accumulator
    return examples


def to_keyed_sequence_examples(
    examples: beam.PCollection[types.Example],
    get_key_fn: types.GetKeyFn,
    features_dict: tfds.features.FeaturesDict,
) -> beam.PCollection[tuple[bytes, tf.train.SequenceExample]]:
  """Partitions a PCollection of examples as keyed `tf.train.SequenceExample`s."""

  def serialize_keyed_example(
      keyed_example: tuple[bytes, types.Example]
  ) -> tuple[bytes, bytes]:
    key, example = keyed_example
    serialized_example = serialization.serialize_tfds_example(
        example, features_dict
    )
    return key, serialized_example

  def serialize_group(
      grouped_examples: tuple[bytes, list[bytes]]
  ) -> tuple[bytes, tf.train.SequenceExample]:
    group_id, bytes_list = grouped_examples
    return group_id, serialization.create_sequence_example(bytes_list)

  keyed = examples | 'KeyExamples' >> beam.Map(lambda x: (get_key_fn(x), x))
  serialized = keyed | 'Serialize' >> beam.Map(serialize_keyed_example)
  grouped = serialized | 'CombineByKey' >> beam.CombinePerKey(
      MergeWithLimitFn()
  )
  serialized_groups = grouped | 'SerializeGroups' >> beam.Map(serialize_group)
  return serialized_groups


def compute_group_counts(
    examples: beam.PCollection[types.Example],
    get_key_fn: types.GetKeyFn,
    delimiter: str = ',',
) -> beam.PCollection[str]:
  """Computes various counts of a PCollection, grouped by a key."""
  keyed_examples = examples | 'KeyExamples' >> beam.Map(
      lambda x: (get_key_fn(x), x)
  )
  keyed_counts = keyed_examples | 'ComputeCounts' >> beam.Map(
      lambda x: (x[0], count_utils.get_group_count(x[1]))
  )
  grouped_counts = keyed_counts | 'GroupCounts' >> beam.GroupByKey()
  merged_counts = grouped_counts | 'MergeCounts' >> beam.ParDo(
      count_utils.MergeGroupCounts()
  )
  formatted_counts = merged_counts | 'FormatCounts' >> beam.ParDo(
      count_utils.FormatGroupCount(), delimiter=delimiter
  )
  return formatted_counts
