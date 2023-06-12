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
"""Tests for beam_transforms.py."""

from absl.testing import absltest
import apache_beam as beam
from apache_beam.testing import test_pipeline
from apache_beam.testing import util
from dataset_grouper.core import beam_transforms
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


class MergeWithLimitFnTest(absltest.TestCase):

  def test_add_input_under_limit(self):
    accumulator = ([b'one', b'two'], 6)
    element = b'three'
    merger = beam_transforms.MergeWithLimitFn(limit=12)
    output = merger.add_input(accumulator, element)
    expected_output = ([b'one', b'two', b'three'], 6 + 5)
    self.assertEqual(output, expected_output)

  def test_add_input_over_limit(self):
    accumulator = ([b'one', b'two'], 6)
    element = b'four'
    merger = beam_transforms.MergeWithLimitFn(limit=8)
    output = merger.add_input(accumulator, element)
    self.assertEqual(output, accumulator)

  def test_merge_two_accumulators_under_limit(self):
    accumulator1 = ([b'one', b'two'], 6)
    accumulator2 = ([b'three', b'four'], 9)
    merger = beam_transforms.MergeWithLimitFn(limit=20)
    output = merger.merge_accumulators([accumulator1, accumulator2])
    expected_output = ([b'one', b'two', b'three', b'four'], 6 + 9)
    self.assertEqual(output, expected_output)

  def test_merge_two_accumulators_over_limit(self):
    accumulator1 = ([b'one', b'two'], 6)
    accumulator2 = ([b'three', b'four'], 9)
    merger = beam_transforms.MergeWithLimitFn(limit=12)
    output = merger.merge_accumulators([accumulator1, accumulator2])
    expected_output = ([b'one', b'two', b'three'], 6 + 5)
    self.assertEqual(output, expected_output)

  def test_merge_three_accumulators_over_limit(self):
    accumulator1 = ([b'one', b'two'], 6)
    accumulator2 = ([b'three', b'four'], 9)
    accumulator3 = ([b'five', b'six'], 7)
    merger = beam_transforms.MergeWithLimitFn(limit=20)
    output = merger.merge_accumulators(
        [accumulator1, accumulator2, accumulator3]
    )
    expected_output = ([b'one', b'two', b'three', b'four', b'five'], 6 + 9 + 4)
    self.assertEqual(output, expected_output)

  def test_beam_merges_up_to_bytes_limit_for_no_groups(self):
    examples = [
        (b'group1', b'a'),
        (b'group1', b'b'),
        (b'group1', b'c'),
        (b'group2', b'dd'),
        (b'group2', b'ee'),
    ]
    expected_output = [
        (b'group1', [b'a', b'b', b'c']),
        (b'group2', [b'dd', b'ee']),
    ]

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = pcoll | beam.CombinePerKey(
          beam_transforms.MergeWithLimitFn(limit=5)
      )
      util.assert_that(output, util.equal_to(expected_output))

  def test_beam_merges_up_to_bytes_limit_for_some_groups(self):
    examples = [
        (b'group1', b'a'),
        (b'group1', b'b'),
        (b'group1', b'c'),
        (b'group2', b'dd'),
        (b'group2', b'ee'),
    ]
    expected_output = [(b'group1', [b'a', b'b', b'c']), (b'group2', [b'dd'])]

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = pcoll | beam.CombinePerKey(
          beam_transforms.MergeWithLimitFn(limit=4)
      )
      util.assert_that(output, util.equal_to(expected_output))

  def test_beam_merges_up_to_bytes_limit_for_all_groups(self):
    examples = [
        (b'group1', b'a'),
        (b'group1', b'b'),
        (b'group1', b'c'),
        (b'group2', b'dd'),
        (b'group2', b'ee'),
    ]
    expected_output = [(b'group1', [b'a', b'b']), (b'group2', [b'dd'])]

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = pcoll | beam.CombinePerKey(
          beam_transforms.MergeWithLimitFn(limit=3)
      )
      util.assert_that(output, util.equal_to(expected_output))


# TODO(b/285414270): Figure out how to re-configure these tests to use some
# kind of `mock` operation on the serialization, rather than just testing that
# the type of the serialized object is correct.
class ToKeyedSequenceExamplesTest(tf.test.TestCase):

  def test_with_single_group(self):
    features_dict = tfds.features.FeaturesDict(
        {'a': np.str_, 'b': np.float32}
    )
    examples = [
        {'a': tf.constant('foo'), 'b': tf.constant(1.0)},
        {'a': tf.constant('foo'), 'b': tf.constant(2.0)},
        {'a': tf.constant('bar'), 'b': tf.constant(3.0)},
        {'a': tf.constant('bar'), 'b': tf.constant(4.0)},
        {'a': tf.constant('baz'), 'b': tf.constant(5.0)},
    ]
    get_key_fn = lambda _: b'group'
    expected_output = [(b'group', tf.train.SequenceExample)]

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = beam_transforms.to_keyed_sequence_examples(
          pcoll, get_key_fn, features_dict
      )
      output_to_type = output | beam.Map(lambda x: (x[0], type(x[1])))
      util.assert_that(output_to_type, util.equal_to(expected_output))

  def test_with_multiple_groups(self):
    features_dict = tfds.features.FeaturesDict({'a': np.str_, 'b': np.float32})
    examples = [
        {'a': tf.constant('foo'), 'b': tf.constant(1.0)},
        {'a': tf.constant('foo'), 'b': tf.constant(2.0)},
        {'a': tf.constant('bar'), 'b': tf.constant(3.0)},
        {'a': tf.constant('bar'), 'b': tf.constant(4.0)},
        {'a': tf.constant('baz'), 'b': tf.constant(5.0)},
    ]
    get_key_fn = lambda x: x['a'].numpy()
    expected_output = [
        (b'foo', tf.train.SequenceExample),
        (b'bar', tf.train.SequenceExample),
        (b'baz', tf.train.SequenceExample),
    ]

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = beam_transforms.to_keyed_sequence_examples(
          pcoll, get_key_fn, features_dict
      )
      output_to_type = output | beam.Map(lambda x: (x[0], type(x[1])))
      util.assert_that(output_to_type, util.equal_to(expected_output))


class ComputeGroupCountsTest(absltest.TestCase):

  def test_compute_group_counts_with_single_groups(self):
    get_key_fn = lambda x: b'group'
    examples = [
        tf.constant('foo'),
        tf.constant('bar bar'),
        tf.constant('baz baz baz'),
    ]
    # There are 3 examples, with a total byte size of 21 and 6 words.
    expected_output = ['group,3,21,6']

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = beam_transforms.compute_group_counts(pcoll, get_key_fn)
      util.assert_that(output, util.equal_to(expected_output))

  def test_compute_group_counts_with_delimiter(self):
    get_key_fn = lambda x: b'group'
    examples = [
        tf.constant('foo'),
        tf.constant('bar bar'),
        tf.constant('baz baz baz'),
    ]
    # There are 3 examples, with a total byte size of 21 and 6 words.
    expected_output = ['group+3+21+6']

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = beam_transforms.compute_group_counts(
          pcoll, get_key_fn, delimiter='+'
      )
      util.assert_that(output, util.equal_to(expected_output))

  def test_compute_group_counts_with_two_groups(self):
    get_key_fn = lambda x: x['a'].numpy()
    examples = [
        {'a': tf.constant('1'), 'b': tf.constant('foo')},
        {'a': tf.constant('1'), 'b': tf.constant('bar bar')},
        {'a': tf.constant('2'), 'b': tf.constant('baz baz baz')},
        {'a': tf.constant('2'), 'b': tf.constant('bat bat bat bat')},
    ]
    # Group '1' has 2 examples, 12 bytes, and 5 words.
    # Group '2' has 2 examples, 28 bytes, and 9 words.
    expected_output = ['1,2,12,5', '2,2,28,9']

    with test_pipeline.TestPipeline() as p:
      pcoll = p | beam.Create(examples)
      output = beam_transforms.compute_group_counts(pcoll, get_key_fn)
      util.assert_that(output, util.equal_to(expected_output))


if __name__ == '__main__':
  absltest.main()
