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
"""Utilities for testing Beam pipelines and dataset loading."""

import glob
import os.path
import tempfile

import apache_beam as beam
from dataset_grouper import tfds_pipelines
import tensorflow_datasets as tfds


def prepare_test_tfrecord_dataset() -> (
    tuple[tfds.core.DatasetBuilder, list[str]]
):
  """Prepares a test TFRecord dataset for use with `PartitionedDataset`.

  This uses a test dataset with `tfds_to_tfrecords`, writing all examples to
  a single client's dataset.

  Returns:
    A prepared `tfds.core.DatasetBuilder`, and a list of file paths.
  """

  data_dir = tempfile.mkdtemp()
  dataset_builder = tfds.testing.DummyDataset(data_dir=data_dir)
  dataset_builder.download_and_prepare()
  file_path_prefix = os.path.join(data_dir, 'testing', 'train')
  file_name_suffix = '.tfrecord'
  test_pipeline = tfds_pipelines.tfds_to_tfrecords(
      dataset_builder=dataset_builder,
      split='train',
      file_path_prefix=file_path_prefix,
      file_name_suffix=file_name_suffix,
      get_key_fn=lambda x: b'client',
  )
  with beam.Pipeline() as root:
    test_pipeline(root)

  prepared_files = glob.glob(file_path_prefix + '*' + file_name_suffix)
  return dataset_builder, prepared_files
