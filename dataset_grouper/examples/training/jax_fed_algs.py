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
"""An implementation of FedAvg in JAX backed by Optax optimizers."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax

PRNGKey = jax.Array
Params = Any
Loss = jnp.float32
ClientOptState = optax.OptState
ServerOptState = optax.OptState
Batch = Any
Batches = Any
Cohort = Any
Carry = tuple[Params, ClientOptState, PRNGKey]


def build_fed_avg(
    loss_fn: Callable[[Params, Batch, PRNGKey], Loss],
    client_optimizer: optax.GradientTransformation,
    server_optimizer: optax.GradientTransformation,
) -> Callable[
    [Params, Cohort, ServerOptState, PRNGKey],
    tuple[Params, ServerOptState, Loss],
]:
  """Builds a function that performs a round of FedAvg.

  Args:
    loss_fn: A function (params, batch, prng_key) -> loss
    client_optimizer: An optax optimizer.
    server_optimizer: An optax optimizer.

  Returns:
    A function (params, cohort, opt_state, prng_key) -> (params, opt_state,
    loss)
  """
  grad_fn = jax.value_and_grad(loss_fn)

  @jax.jit
  def train_step(
      params: Params,
      opt_state: ClientOptState,
      batch: Batch,
      prng_key: PRNGKey,
  ) -> tuple[Params, ClientOptState, Loss]:
    loss, grad = grad_fn(params, batch, prng_key)
    updates, opt_state = client_optimizer.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

  def carry_fn(carry: Carry, batch: Batch) -> tuple[Carry, Loss]:
    params, opt_state, prng_key = carry
    params, opt_state, loss = train_step(params, opt_state, batch, prng_key)
    return (params, opt_state, prng_key), loss

  @jax.jit
  def client_train(
      params: Params, batches: Batches, prng_key: PRNGKey
  ) -> tuple[Params, Loss]:
    opt_state = client_optimizer.init(params)
    state = (params, opt_state, prng_key)
    state, losses = jax.lax.scan(carry_fn, state, batches)
    final_params, _, _ = state
    average_loss = jnp.mean(losses)
    model_delta = jax.tree_util.tree_map(
        lambda x, y: x - y, params, final_params
    )
    return model_delta, average_loss

  parallel_client_train = jax.vmap(client_train, in_axes=(None, 0, None))

  @jax.jit
  def server_update(
      params: Params, average_delta: Params, server_opt_state: ServerOptState
  ) -> tuple[Params, ServerOptState]:
    update, server_opt_state = server_optimizer.update(
        average_delta, server_opt_state)
    params = optax.apply_updates(params, update)
    return params, server_opt_state

  @jax.jit
  def fed_avg(
      params: Params,
      client_batches: Cohort,
      server_opt_state: ServerOptState,
      prng_key: PRNGKey,
  ) -> tuple[Params, ServerOptState, Loss]:
    model_deltas, losses = parallel_client_train(
        params, client_batches, prng_key
    )
    average_delta = jax.tree_util.tree_map(jnp.mean, model_deltas)
    average_loss = jnp.mean(losses)
    params, server_opt_state = server_update(
        params, average_delta, server_opt_state
    )
    return params, server_opt_state, average_loss

  return fed_avg
