# Copyright 2024 Google LLC
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
"""Tests for data_loaders.py."""

from absl.testing import absltest
from dataset_grouper import data_loaders
from dataset_grouper import test_utils
import tensorflow as tf


class DataLoadersTest(absltest.TestCase):

  def test_create_partitioned_dataset(self):
    dataset_builder, dataset_files = test_utils.prepare_test_tfrecord_dataset()
    features_dict = dataset_builder.info.features
    partitioned_dataset = data_loaders.PartitionedDataset(
        dataset_files, features_dict
    )
    group_stream = partitioned_dataset.build_group_stream()
    group_stream_element_spec = group_stream.element_spec
    self.assertIsInstance(group_stream_element_spec, tf.data.DatasetSpec)
    dataset_element_spec = group_stream_element_spec.element_spec
    self.assertDictEqual(dataset_element_spec, features_dict.get_tensor_spec())


if __name__ == "__main__":
  absltest.main()
