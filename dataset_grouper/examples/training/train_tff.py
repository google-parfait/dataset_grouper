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
"""Train a transformer on the FedC4 dataset using TFF."""

from collections.abc import Sequence

from absl import app
from absl import flags
from absl import logging
import dataset_grouper as dsgp
from dataset_grouper.examples.training import dataset_utils
from dataset_grouper.examples.training import model_utils
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff
import tensorflow_text as text


# Dataset configuration
_DATASET_SHARD_PATTERN = flags.DEFINE_string(
    'dataset_shard_pattern', None, 'Glob pattern for federated C4 shards.'
)
_CLIENTS_PER_TRAIN_ROUND = flags.DEFINE_integer(
    'clients_per_train_round',
    10,
    'How many clients to sample at each training round.',
)
_TRAIN_BATCH_SIZE = flags.DEFINE_integer(
    'train_batch_size', 4, 'Batch size on train clients.'
)
_MAX_ELEMENTS = flags.DEFINE_integer(
    'max_elements', None, 'Maximum number of examples to use at every client.',
    required=True,
)

# Training configuration
_USE_FEDSGD = flags.DEFINE_bool(
    'use_fedsgd', False, 'If True, use FedSGD; else, use standard FedAvg.'
)
_CLIENT_LR = flags.DEFINE_float('client_lr', 0.01, 'Client learning rate.')
_SERVER_LR = flags.DEFINE_float('server_lr', 0.001, 'Server learning rate.')
_TOTAL_ROUNDS = flags.DEFINE_integer(
    'total_rounds', 100, 'Number of total training rounds.'
)

# Model configuration
_BASE_RANDOM_SEED = flags.DEFINE_integer(
    'base_random_seed', 0, 'A random seed governing model initialization.',
)
_MODEL_CONFIG = flags.DEFINE_enum(
    'model_config',
    None,
    model_utils.MODEL_CONFIG_NAMES,
    'Model configuration for the Transformer model.',
)
_VOCAB_SIZE = flags.DEFINE_integer(
    'vocab_size', None, 'Size of the model vocabulary', required=True,
)
_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path', None, 'Path to load the tokenizer vocabulary',
    required=True,
)
_MAX_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'max_sequence_length', 257, 'Maximum sequence length'
)  # Note: This sould be 1 plus a power of 2 (for input/target split)


_DATASET_NAME = 'c4'
_DATASET_TEXT_FIELD = 'text'


def build_cohort_stream(
    dataset_shard_pattern: str,
    tokenizer_path: str,
    batch_size: int,
    max_elements: int,
    sequence_length: int,
    num_clients_per_round: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
  """Creates a preprocessed version of the federated C4 dataset."""
  tf_tokenizer = text.BertTokenizer(tokenizer_path, lower_case=True)
  preprocess_fn = dataset_utils.build_preprocess_fn(
      tf_tokenizer=tf_tokenizer,
      num_epochs=1,
      batch_size=batch_size,
      max_elements=max_elements,
      sequence_length=sequence_length,
      dataset_text_field=_DATASET_TEXT_FIELD,
      shuffle=shuffle,
  )
  dataset_builder = tfds.builder(_DATASET_NAME)
  partitioned_dataset = dsgp.SSTablePartitionedDataset(
      file_pattern=dataset_shard_pattern,
      tfds_features=dataset_builder.info.features,
  )
  group_stream = partitioned_dataset.build_group_stream()
  group_stream = group_stream.map(
      preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE
  )
  cohort_stream = group_stream.window(
      num_clients_per_round, drop_remainder=True
  )
  return cohort_stream


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Step 1: Load the dataset.
  train_cohort_stream = build_cohort_stream(
      dataset_shard_pattern=_DATASET_SHARD_PATTERN.value,
      tokenizer_path=_TOKENIZER_PATH.value,
      batch_size=_TRAIN_BATCH_SIZE.value,
      max_elements=_MAX_ELEMENTS.value,
      sequence_length=_MAX_SEQUENCE_LENGTH.value,
      num_clients_per_round=_CLIENTS_PER_TRAIN_ROUND.value,
      shuffle=True,
  )
  element_spec = train_cohort_stream.element_spec.element_spec.element_spec
  placeholder_batch = dataset_utils.get_placeholder_batch(
      _TRAIN_BATCH_SIZE.value, _MAX_SEQUENCE_LENGTH.value
  )
  logging.info('Setup client datasets. element spec = %s', element_spec)

  cohort_iter = iter(train_cohort_stream)
  def training_selection_fn(round_num: int) -> list[tf.data.Dataset]:
    del round_num
    cohort = next(cohort_iter)
    return list(cohort)

  # Step 2: Load the model.
  functional_model = model_utils.load_transformer_lm_as_tff_functional_model(
      vocab_size=_VOCAB_SIZE.value,
      model_config=_MODEL_CONFIG.value,
      batch_spec=element_spec,
      placeholder_batch=placeholder_batch,
      random_seed=_BASE_RANDOM_SEED.value,
  )

  # Step 3: Set up the learning algorithm.
  client_optimizer = tff.learning.optimizers.build_sgdm(
      learning_rate=_CLIENT_LR.value)
  server_optimizer = tff.learning.optimizers.build_adam(
      learning_rate=_SERVER_LR.value
  )
  if _USE_FEDSGD.value:
    learning_process = tff.learning.algorithms.build_fed_sgd(
        model_fn=functional_model,
        server_optimizer_fn=server_optimizer,
        model_aggregator=tff.learning.robust_aggregator(),
    )
  else:
    learning_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=functional_model,
        client_optimizer_fn=client_optimizer,
        server_optimizer_fn=server_optimizer,
        model_aggregator=tff.learning.robust_aggregator(),
    )

  # Step 4: Run!
  tff.simulation.run_training_process(
      training_process=learning_process,
      training_selection_fn=training_selection_fn,
      total_rounds=_TOTAL_ROUNDS.value,
      metrics_managers=[tff.program.LoggingReleaseManager()],
  )


if __name__ == '__main__':
  app.run(main)
