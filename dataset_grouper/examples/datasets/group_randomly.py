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
"""Partition a TFDS dataset randomly across a set of groups."""

from collections.abc import Sequence
import os.path
import random

from absl import app
from absl import flags
import apache_beam as beam
import dataset_grouper as dsgp
import tensorflow_datasets as tfds


_TFDS_NAME = flags.DEFINE_string(
    name='tfds_name',
    default=None,
    help='The name of the dataset in TFDS.',
    required=True,
)
_DATA_DIR = flags.DEFINE_string(
    name='data_dir',
    default=None,
    help='The data directory for TFDS.',
)
_NUM_GROUPS = flags.DEFINE_integer(
    name='num_groups',
    default=None,
    help='The number of groups to partition over.',
    required=True,
)
_OUTPUT_PATH = flags.DEFINE_string(
    name='output_path',
    default=None,
    help='The output file path.',
    required=True,
)
_FILE_PREFIX = flags.DEFINE_string(
    name='file_prefix',
    default=None,
    help='The base name for saved TFRecords files.',
    required=True,
)
_SPLIT = flags.DEFINE_string(
    name='split',
    default=None,
    help=(
        'Which split to write. This can include TFDS slicing operations, as in '
        'https://www.tensorflow.org/datasets/splits.'
    ),
    required=True,
)
_NUM_SHARDS = flags.DEFINE_integer(
    name='num_shards',
    default=None,
    help=(
        'Number of shards. If set to `None`, this will be inferred '
        'automatically.'
    ),
)


def get_key_fn(example: dsgp.Example) -> bytes:
  """A function that returns a random group ID."""
  del example  # This partitioning is agnostic to the example
  group_id = random.randint(0, _NUM_GROUPS.value - 1)
  return str(group_id).encode('utf-8')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_builder = tfds.builder(
      name=_TFDS_NAME.value, data_dir=_DATA_DIR.value
  )
  dataset_builder.download_and_prepare()
  file_path_prefix = os.path.join(_OUTPUT_PATH.value, _FILE_PREFIX.value)
  partition_pipeline = dsgp.tfds_to_tfrecords(
      dataset_builder=dataset_builder,
      split=_SPLIT.value,
      file_path_prefix=file_path_prefix,
      get_key_fn=get_key_fn,
      num_shards=_NUM_SHARDS.value,
  )
  with beam.Pipeline() as p:
    partition_pipeline(p)


if __name__ == '__main__':
  app.run(main)
