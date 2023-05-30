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

import functools
from typing import Optional

import apache_beam as beam
from dataset_grouper.core import count_utils
from dataset_grouper.core import serialization
from dataset_grouper.core import types
import tensorflow_datasets as tfds


def key_and_group(
    examples: beam.PCollection[types.Example],
    get_key_fn: types.GetKeyFn,
    filter_fn: Optional[types.FilterFn] = None,
) -> beam.PCollection[types.GroupedExamples]:
  """Groups a PCollection of examples by key, and filters out groups."""
  keyed_examples = examples | 'KeyExamples' >> beam.Map(
      lambda x: (get_key_fn(x), x)
  )
  grouped_examples = keyed_examples | 'GroupExamples' >> beam.GroupByKey()
  if filter_fn is not None:
    grouped_examples = grouped_examples | 'FilterGroups' >> beam.Filter(
        filter_fn
    )
  return grouped_examples


def to_keyed_sequence_examples(
    examples: beam.PCollection[types.Example],
    get_key_fn: types.GetKeyFn,
    features_dict: tfds.features.FeaturesDict,
    filter_fn: Optional[types.FilterFn] = None,
) -> beam.PCollection[types.KeyedSequenceExample]:
  """Partitions a PCollection of examples as keyed `tf.train.SequenceExample`s."""
  to_sequence_example = functools.partial(
      serialization.sequence_example_from_features_dict,
      features_dict=features_dict,
  )

  def sequence_map(
      grouped_examples: types.GroupedExamples,
  ) -> types.KeyedSequenceExample:
    group_id, examples = grouped_examples
    sequence_example = to_sequence_example(examples)
    return group_id, sequence_example

  grouped_examples = key_and_group(examples, get_key_fn, filter_fn=filter_fn)
  return grouped_examples | 'SequenceMap' >> beam.Map(sequence_map)


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
