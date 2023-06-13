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

import os.path

import apache_beam as beam
from dataset_grouper.core import data_loaders
from dataset_grouper.core import tfds_pipelines
import tensorflow as tf
import tensorflow_datasets as tfds


class IntegrationTest(tf.test.TestCase):

  def test_pipeline_runs_and_data_loads(self):
    temp_dir = self.get_temp_dir()
    data_dir = os.path.join(temp_dir, 'data')
    dataset_builder = tfds.testing.DummyMnist(data_dir=data_dir)
    dataset_builder.download_and_prepare()

    save_dir = os.path.join(temp_dir, 'save')
    file_path_prefix = os.path.join(save_dir, 'mnist_test.tfrecord')

    mnist_pipeline = tfds_pipelines.tfds_to_tfrecords(
        dataset_builder=dataset_builder,
        split='test',
        get_key_fn=lambda _: b'test_client',
        file_path_prefix=file_path_prefix,
        num_shards=1,
    )
    with beam.Pipeline() as root:
      mnist_pipeline(root)

    written_files = tf.io.gfile.listdir(save_dir)
    self.assertEqual(written_files, ['mnist_test.tfrecord-00000-of-00001'])

    file_pattern = [os.path.join(save_dir, x) for x in written_files]
    partitioned_dataset = data_loaders.PartitionedDataset(
        file_pattern=file_pattern, tfds_features=dataset_builder.info.features
    )
    group_stream = partitioned_dataset.build_group_stream()
    self.assertIsInstance(group_stream, tf.data.Dataset)
    group_element_spec = group_stream.element_spec
    self.assertIsInstance(group_element_spec, tf.data.DatasetSpec)
    mnist_dataset = dataset_builder.as_dataset(split='train')
    self.assertDictEqual(
        group_element_spec.element_spec, mnist_dataset.element_spec
    )


if __name__ == '__main__':
  tf.test.main()
