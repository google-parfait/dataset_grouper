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
"""Tests for count_utils.py."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from dataset_grouper import count_utils


class CountUtilsTest(absltest.TestCase):

  def test_merge_client_records(self):
    test_counts = [
        (
            b'a',
            [
                count_utils.GroupCount(1, 2, 3),
                count_utils.GroupCount(2, 3, 4),
                count_utils.GroupCount(3, 7, 10),
            ],
        ),
        (
            b'b',
            [
                count_utils.GroupCount(2, 0, 0),
                count_utils.GroupCount(3, 3, 1),
            ],
        ),
        (
            b'c',
            [
                count_utils.GroupCount(7, 1, 2),
            ],
        ),
    ]
    with beam.Pipeline() as root:
      keyed_counts = root | beam.Create(test_counts)
      merged_counts = keyed_counts | beam.ParDo(count_utils.MergeGroupCounts())

    expected_counts = [
        (b'a', count_utils.GroupCount(6, 12, 17)),
        (b'b', count_utils.GroupCount(5, 3, 1)),
        (b'c', count_utils.GroupCount(7, 1, 2)),
    ]
    assert_that(merged_counts, equal_to(expected_counts))

  def test_format_group_counts(self):
    test_records = [
        (b'a', count_utils.GroupCount(1, 2, 3)),
        (b'b', count_utils.GroupCount(2, 3, 5)),
        (b'c', count_utils.GroupCount(3, 7, 10)),
    ]
    with beam.Pipeline() as root:
      keyed_records = root | beam.Create(test_records)
      actual_result = keyed_records | beam.ParDo(count_utils.FormatGroupCount())

    expected_result = [
        'a,1,2,3',
        'b,2,3,5',
        'c,3,7,10',
    ]
    assert_that(actual_result, equal_to(expected_result))

  def test_format_group_counts_with_delimiter(self):
    test_records = [
        (b'a', count_utils.GroupCount(1, 2, 3)),
        (b'b', count_utils.GroupCount(2, 3, 5)),
        (b'c', count_utils.GroupCount(3, 7, 10)),
    ]
    with beam.Pipeline() as root:
      keyed_records = root | beam.Create(test_records)
      actual_result = keyed_records | beam.ParDo(
          count_utils.FormatGroupCount(), delimiter='+'
      )

    expected_result = [
        'a+1+2+3',
        'b+2+3+5',
        'c+3+7+10',
    ]
    assert_that(actual_result, equal_to(expected_result))


if __name__ == '__main__':
  absltest.main()
