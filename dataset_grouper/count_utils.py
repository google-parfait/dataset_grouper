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
"""Functions for getting aggregate counts of dataset statistics."""

import collections
from collections.abc import Iterable

import apache_beam as beam
from dataset_grouper import tf_utils
from dataset_grouper import types


GroupCount = collections.namedtuple(
    'GroupCount', ['num_examples', 'num_bytes', 'num_words']
)


@beam.typehints.with_input_types(tuple[bytes, Iterable[GroupCount]])
@beam.typehints.with_output_types(tuple[bytes, GroupCount])
class MergeGroupCounts(beam.DoFn):
  """A class for merging `GroupStats`s grouped by key."""

  def process(self, keyed_group_counts):
    """Merges an iterable of `GroupStats`s."""
    group_id, group_counts = keyed_group_counts
    num_examples = 0
    num_bytes = 0
    num_words = 0
    for group_count in group_counts:
      num_examples += group_count.num_examples
      num_bytes += group_count.num_bytes
      num_words += group_count.num_words
    merged_group_count = GroupCount(
        num_examples=num_examples, num_bytes=num_bytes, num_words=num_words
    )
    yield group_id, merged_group_count


@beam.typehints.with_input_types(tuple[bytes, GroupCount])
@beam.typehints.with_output_types(str)
class FormatGroupCount(beam.DoFn):
  """Formats a keyed `GroupCount` to a string for output."""

  def process(self, keyed_group_count, delimiter=','):
    group_id, group_count = keyed_group_count
    group_count_to_str = delimiter.join(str(x) for x in group_count)
    yield group_id.decode() + delimiter + group_count_to_str


def get_group_count(example: types.Example) -> GroupCount:
  num_examples = 1
  num_bytes = tf_utils.get_tensor_byte_size(example)
  num_words = tf_utils.get_tensor_num_words(example)
  return GroupCount(num_examples, num_bytes, num_words)
