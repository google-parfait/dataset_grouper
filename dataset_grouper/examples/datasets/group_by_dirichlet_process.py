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
"""Partition a TFDS dataset by a Dirichlet-governed process.

This Dirichlet process is similar to Latent Dirichlet Allocation (LDA), and is
used to partition examples in a dataset across some number of groups. For each
possible label in a user-specified feature, we generate a categorical
distribution over groups, by sampling from a Dirichlet. Each example is
partitioned randomly according to the categorical distribution corresponding to
its label.

For example, say there are 3 possible labels (0, 1, and 2) and we'd like to
partition examples over 4 groups. After sampling three times from a Dirichlet
distribution, we might get the following categorical distributions:

0 - [0.1, 0.6, 0.2, 0.1]
1 - [0.7, 0.1, 0.0, 0.2]
2 - [0.0, 0.5, 0.5, 0.0]

When we assign an example to a group, we look at its label. Suppose the label is
1. We sample over the categorical distribution [0.7, 0.1, 0.0, 0.2], and assign
the example to that group. This means the example is likely to be assigned to
group 0, and will never be assigned to group 2.

This is similar to the Dirichlet scheme from https://arxiv.org/abs/1909.06335,
but the distributions are over groups, not labels. This allows us to do the
partitioning in a completely parallelizable fashion.
"""

from collections.abc import Callable, Sequence
import os.path

from absl import app
from absl import flags
import apache_beam as beam
import dataset_grouper as dsgp
import numpy as np
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
_DIRICHLET_PARAMETER = flags.DEFINE_float(
    name='dirichlet_parameter',
    default=None,
    help=(
        'The concentration parameter of the (symmetric) Dirichlet. Higher'
        ' values result in a more homogeneous partitioning.'
    ),
    required=True,
)
_GROUP_FEATURE = flags.DEFINE_string(
    name='group_feature',
    default=None,
    help=(
        'The name of the feature used to create groups. This must be an integer'
        ' feature, whose range is specified by the `feature_min` and'
        ' `feature_max` flags.'
    ),
    required=True,
)
_FEATURE_MIN = flags.DEFINE_integer(
    name='feature_min',
    default=0,
    help='The minimum value of the integer feature governing the partition.',
)
_FEATURE_MAX = flags.DEFINE_integer(
    name='feature_max',
    default=None,
    help='The maximum value of the integer feature governing the partition.',
    required=True,
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


def build_get_key_fn() -> Callable[[dsgp.Example], bytes]:
  """Builds a function that partitions examples by a Dirichlet process."""
  rng = np.random.default_rng()
  num_labels = _FEATURE_MAX.value - _FEATURE_MIN.value + 1
  alpha = [_DIRICHLET_PARAMETER.value] * _NUM_GROUPS.value
  labels_to_group_probs = rng.dirichlet(alpha=alpha, size=num_labels)
  all_groups = list(range(_NUM_GROUPS.value))

  def get_key_fn(example: dsgp.Example) -> bytes:
    """A function that extracts the base domain from a C4 example."""
    label = example[_GROUP_FEATURE.value].numpy()
    pvals = labels_to_group_probs[label - _FEATURE_MIN.value, :]
    group_id = rng.choice(all_groups, p=pvals)
    return str(group_id).encode('utf-8')

  return get_key_fn


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_builder = tfds.builder(
      name=_TFDS_NAME.value, data_dir=_DATA_DIR.value
  )
  dataset_builder.download_and_prepare()
  file_path_prefix = os.path.join(_OUTPUT_PATH.value, _FILE_PREFIX.value)
  get_key_fn = build_get_key_fn()
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
