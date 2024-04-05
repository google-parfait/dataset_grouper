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
"""Utilities to create transformer models in JAX and TFF."""

import collections
from collections.abc import Callable
from typing import Any, Optional, Union

import jax
from jax.experimental import jax2tf
import jax.numpy as jnp
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
import tensorflow as tf
import tensorflow_federated as tff


PyTree = Any  # Type annotation for a Jax PyTree.
OrderedDict = collections.OrderedDict
JaxExampleType = OrderedDict[
    str, Union[jnp.ndarray, OrderedDict[str, jnp.ndarray]]
]
ModelWeightsType = Any
TfExampleType = OrderedDict[str, Union[tf.Tensor, OrderedDict[str, tf.Tensor]]]
FunctionalBatchOutput = collections.namedtuple(
    'FunctionalBatchOutput',
    ['logits', 'loss', 'next_token_accuracy', 'total_weight'],
)

# Model configs
HALF_BASE_MODEL = 'half_base'  # 6 layers & heads, 384 dim.
BASE_MODEL = 'base'  # 12 layers & heads, 768 dim.
MODEL_CONFIG_NAMES = [BASE_MODEL, HALF_BASE_MODEL]


def get_praxis_transformer_model(
    vocab_size: int,
    model_config: str,
) -> layers.transformer_models.TransformerLm:
  """Create a praxis transformer model."""
  if model_config == HALF_BASE_MODEL:
    num_layers = num_attention_heads = 6
    hidden_dim = 384
  elif model_config == BASE_MODEL:
    num_layers = num_attention_heads = 12
    hidden_dim = 768
  else:
    raise ValueError(f'Found unexpected model_config {model_config}.')

  p = pax_fiddle.Config(
      layers.transformer_models.TransformerLm,
      name='lm',
      model_type=layers.transformer_models.LanguageModelType.CAUSAL,
      model_dims=hidden_dim,
      vocab_size=vocab_size,
  )
  stacked_transformer_tpl = p.stacked_transformer_tpl
  stacked_transformer_tpl.model_dims = hidden_dim
  stacked_transformer_tpl.hidden_dims = 4 * hidden_dim
  stacked_transformer_tpl.num_heads = num_attention_heads
  stacked_transformer_tpl.num_layers = num_layers
  p.softmax_tpl.scale_sqrt_depth = True
  return base_layer.instantiate(p)


def get_predict_on_batch_function(
    language_model: layers.transformer_models.TransformerLm,
    tree_structure: PyTree,
) -> Callable[[ModelWeightsType, TfExampleType, bool], FunctionalBatchOutput]:
  """Build a function that performs a forward pass on a transformer model."""

  def predict_on_batch_jax(trainable_weights, input_batch, training):
    del training
    input_batch = py_utils.NestedMap.FromNestedDict(input_batch)
    flat_trainable_weights, _ = trainable_weights
    unflattened_weights = jax.tree_util.tree_unflatten(
        tree_structure, flat_trainable_weights
    )
    # Note: input_batch is the 'x' component of the dataset spec
    output = language_model.apply(
        unflattened_weights,
        input_batch.input_ids,
        input_batch.input_paddings,
        labels=input_batch.labels,
    )
    # The relevant fields are:
    # * output.logits: (batch, seq_len, vocab_size)
    # * output.total_loss: scalar avg. cross entropy + auxiliary loss (if any)
    # * output.per_example_argmax: (batch, seq_len)
    # * output.total_weight: total number of predictions in the batch
    next_token_accuracy = jnp.dot(
        input_batch.labels.class_weights.reshape(-1),
        (output.per_example_argmax == input_batch.labels.class_ids).reshape(-1),
    ) / jnp.float32(input_batch.labels.class_weights.sum())
    return FunctionalBatchOutput(
        output.logits,
        output.total_loss,
        next_token_accuracy,
        output.total_weight,
    )

  predict_on_batch_tf = tf.function(jax2tf.convert(predict_on_batch_jax))
  return predict_on_batch_tf


@tf.function
def loss_fn(
    functional_batch_output: FunctionalBatchOutput,
    label: Any,
    sample_weights: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Wrapper to return the loss from FunctionalBatchOutput."""
  del label, sample_weights
  return functional_batch_output.loss


@tf.function
def initialize_metrics() -> OrderedDict[str, tuple[float]]:
  """Metrics helper: initialize."""
  return OrderedDict(num_examples=(0.0,), loss=(0.0,))


@tf.function
def update_metrics_state(
    state: OrderedDict[str, tuple[float]],
    labels: Any,
    batch_output: tff.learning.models.BatchOutput,
    sample_weight: Optional[tf.Tensor] = None,
) -> OrderedDict[str, float]:
  """Metrics helper: update."""
  del labels, sample_weight  # Unused.
  avg_loss = state['loss'][0]
  num_examples = state['num_examples'][0]
  batch_size = tf.cast(batch_output.num_examples, tf.float32)
  this_loss = batch_output.loss
  num_examples_new = batch_size + num_examples
  avg_loss_new = (
      this_loss * batch_size + avg_loss * num_examples
  ) / num_examples_new

  new_dict = OrderedDict(
      num_examples=(num_examples_new,),
      loss=(avg_loss_new,),
  )
  return new_dict


@tf.function
def finalize_metrics(
    state: OrderedDict[str, tuple[float]]
) -> OrderedDict[str, float]:
  """Metrics helper: finalize."""
  return OrderedDict(
      num_examples=state['num_examples'][0],
      loss=state['loss'][0],
  )


def load_transformer_lm_as_tff_functional_model(
    vocab_size: int,
    model_config: str,
    batch_spec: OrderedDict[str, Any],
    placeholder_batch: JaxExampleType,
    random_seed: int,
) -> tff.learning.models.FunctionalModel:
  """Create a FunctionalModel backed by a Praxis Transformer."""
  language_model = get_praxis_transformer_model(vocab_size, model_config)
  # Initialize the weights, using a placeholder batch for input sizes.
  prng_key = jax.random.PRNGKey(seed=random_seed)
  initial_variables = language_model.init(
      prng_key,
      placeholder_batch['input_ids'],
      placeholder_batch['input_paddings'],
  )
  flat_initial_variables, tree_structure = jax.tree_util.tree_flatten(
      initial_variables
  )
  predict_on_batch_fn = get_predict_on_batch_function(
      language_model, tree_structure
  )
  functional_model = tff.learning.models.FunctionalModel(
      initial_weights=(flat_initial_variables, ()),
      predict_on_batch_fn=predict_on_batch_fn,
      metrics_fns=(initialize_metrics, update_metrics_state, finalize_metrics),
      loss_fn=loss_fn,
      input_spec=batch_spec,
  )
  return functional_model
