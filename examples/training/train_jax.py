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
"""Train a transformer on the FedC4 dataset using JAX."""

from collections.abc import Sequence
import time

from absl import app
from absl import flags
from absl import logging
import dataset_grouper as dsgp
from examples.training import dataset_utils
from examples.training import jax_fed_algs
from examples.training import model_utils
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from praxis import base_layer
from praxis import py_utils
import tensorflow as tf
import tensorflow_text as text


_BATCH_SIZE = flags.DEFINE_integer('batch_size', 16, 'Batch size.')
_CKPT_FREQ = flags.DEFINE_integer('ckpt_freq', 500, 'Checkpoint frequency.')
_CLIENT_LR = flags.DEFINE_float('client_lr', 0.01, 'Client learning rate.')
_COHORT_SIZE = flags.DEFINE_integer('cohort_size', 16, 'Cohort size.')
_DATASET_SHARD_PATTERN = flags.DEFINE_string(
    'dataset_shard_pattern', None, 'Glob pattern for federated C4 shards.'
)
_MAX_ELEMENTS = flags.DEFINE_integer('max_elements', 1024, 'Max elements.')
_MODEL_CONFIG = flags.DEFINE_enum(
    'model_config',
    None,
    model_utils.MODEL_CONFIG_NAMES,
    'Model configuration for the Transformer model.',
)
_NUM_STEPS = flags.DEFINE_integer('num_steps', 200000, 'Number of steps.')
_ORBAX_DIR = flags.DEFINE_string('orbax_dir', None, 'Orbax directory.')
_SEQUENCE_LEN = flags.DEFINE_integer('sequence_len', 128, 'Sequence length.')
_SERVER_LR = flags.DEFINE_float('server_lr', 0.001, 'Server learning rate.')
_SERVER_SCHED = flags.DEFINE_enum(
    'server_sched',
    None,
    ['constant', 'linear_exp', 'linear_cosine'],
    'Which learning rate schedule to use.',
)
_TOKENIZER_PATH = flags.DEFINE_string(
    'tokenizer_path',
    None,
    'Path to load the tokenizer vocabulary',
    required=True,
)

_DATASET_NAME = 'c4'
_DATASET_TEXT_FIELD = 'text'


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf_tokenizer = text.BertTokenizer(_TOKENIZER_PATH.value, lower_case=True)
  vocab_size = tf_tokenizer.submodules[1].vocab_size().numpy().item()
  lm = model_utils.get_praxis_transformer_model(
      vocab_size=vocab_size,
      model_config=_MODEL_CONFIG.value,
  )

  context_p = base_layer.JaxContext.HParams(do_eval=False)
  def loss_fn(params, batch, prng_key):
    batch_dict = py_utils.NestedMap.FromNestedDict(batch)
    with base_layer.JaxContext.new_context(hparams=context_p):
      output = lm.apply(
          params,
          batch_dict['x']['input_ids'],
          batch_dict['x']['input_paddings'],
          labels=batch_dict['x']['labels'],
          rngs={'dropout': prng_key},
      )
      loss = output.total_loss
    return loss

  input_ids = jax.random.randint(
      key=jax.random.PRNGKey(1234),
      shape=[_BATCH_SIZE.value, _SEQUENCE_LEN.value],
      minval=0,
      maxval=vocab_size,
  )
  input_paddings = jnp.zeros([_BATCH_SIZE.value, _SEQUENCE_LEN.value])
  init_prng_key = jax.random.PRNGKey(123)
  params = lm.init(init_prng_key, input_ids, input_paddings)

  preprocess_fn = dataset_utils.build_preprocess_fn(
      tf_tokenizer=tf_tokenizer,
      num_epochs=_MAX_ELEMENTS.value,  # Ensures all clients have _MAX_ELEMENTS
      batch_size=_BATCH_SIZE.value,
      max_elements=_MAX_ELEMENTS.value,
      sequence_length=_SEQUENCE_LEN.value,
      dataset_text_field=_DATASET_TEXT_FIELD,
      shuffle=False,
  )

  def preprocess_and_batch(client_ds: tf.data.Dataset) -> tf.data.Dataset:
    client_ds = preprocess_fn(client_ds)
    return client_ds.batch(
        _MAX_ELEMENTS.value // _BATCH_SIZE.value, drop_remainder=True
    )

  partitioned_dataset = dsgp.PartitionedDataset(
      file_pattern=_DATASET_SHARD_PATTERN.VALUE, tfds_features=_DATASET_NAME
  )

  num_rounds = (
      _NUM_STEPS.value // (_MAX_ELEMENTS.value // _BATCH_SIZE.value) + 1
  )
  client_optimizer = optax.sgd(learning_rate=_CLIENT_LR.value)
  if _SERVER_SCHED.value == 'linear_exp':
    learning_rate = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=_SERVER_LR.value,
        warmup_steps=int(num_rounds * 0.1),
        transition_steps=num_rounds,
        decay_rate=1.0
    )
  elif _SERVER_SCHED.value == 'linear_cosine':
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=_SERVER_LR.value,
        warmup_steps=int(num_rounds * 0.1),
        decay_steps=num_rounds,
        end_value=0.0
    )
  else:
    learning_rate = _SERVER_LR.value

  server_optimizer = optax.adam(learning_rate=learning_rate)
  server_opt_state = server_optimizer.init(params)
  fed_avg = jax_fed_algs.build_fed_avg(
      loss_fn,
      client_optimizer,
      server_optimizer,
  )
  orbax_options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=3)
  orbax_mngr = orbax.checkpoint.CheckpointManager(
      _ORBAX_DIR.value, orbax.checkpoint.PyTreeCheckpointer(), orbax_options
  )
  fed_avg_state = {'params': params, 'server_opt_state': server_opt_state}

  client_stream = partitioned_dataset.build_group_stream(shuffle_seed=123)
  batch_stream = client_stream.flat_map(preprocess_and_batch)
  cohort_stream = batch_stream.batch(_COHORT_SIZE.value, drop_remainder=True)
  if orbax_mngr.latest_step() is not None:
    round_num = orbax_mngr.latest_step()
    fed_avg_state = orbax_mngr.restore(round_num, fed_avg_state)
    params = fed_avg_state['params']
    server_opt_state = fed_avg_state['server_opt_state']
    cohort_stream = cohort_stream.skip(round_num)
  else:
    round_num = 0

  cohort_iter = iter(cohort_stream.as_numpy_iterator())
  while round_num < num_rounds:
    cohort_iter_start = time.time()
    cohort = next(cohort_iter)
    cohort_iter_time = time.time() - cohort_iter_start

    fed_avg_start = time.time()
    prng_key = jax.random.PRNGKey(round_num)
    params, server_opt_state, loss = fed_avg(
        params, cohort, server_opt_state, prng_key
    )
    # Uncomment the following line if you would like accurate timing info.
    # Without this, JAX's asynch dispatch means the training time is not as
    # expected.
    # loss.block_until_ready()
    fed_avg_time = time.time() - fed_avg_start

    logging.info(
        'Round %s, cohort_iter %s, fed_avg %s, loss %s',
        round_num,
        cohort_iter_time,
        fed_avg_time,
        loss,
    )
    round_num += 1
    if round_num % _CKPT_FREQ.value == 0:
      fed_avg_state = {'params': params, 'server_opt_state': server_opt_state}
      orbax_mngr.save(round_num, fed_avg_state)


if __name__ == '__main__':
  app.run(main)
