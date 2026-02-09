from typing import Callable, Optional, Tuple, List, Optional, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import inspect
import optax
import paramax
from tqdm.auto import tqdm

from eqxlearn.train.losses import MeanSquaredError

def fit(
    model: eqx.Module,
    optimizer: Optional[optax.GradientTransformation] = None,
    X: Optional[jnp.ndarray] = None,
    y: Optional[jnp.ndarray] = None,
    loss_fn: Optional[Union[Callable, eqx.Module]] = None,
    key: Optional[jax.random.PRNGKey] = None,
    max_steps: int = 1000,
    patience: int = 10,
    show_progress: bool = True,
) -> Tuple[eqx.Module, List[float]]:
    if optimizer is None:
        optimizer = optax.adam(learning_rate=0.1)
    
    # 1. Partition the model
    # We filter out non-trainable leaves using Paramax
    params, static = eqx.partition(
        model,
        eqx.is_inexact_array,
        is_leaf=lambda leaf: isinstance(leaf, paramax.NonTrainable),
    )
    opt_state = optimizer.init(params)

    # 2. Logic to handle "External Data" vs "Internal Loss"
    # -------------------------------------------------------------------------
    
    # CASE A: External Data Provided (e.g. Linear Regression, Neural Network)
    if X is not None and y is not None:
        
        # Guardrail: Don't accidentally train a GP with external data logic
        # if it was designed to solve its own loss.
        if hasattr(model, 'loss'):
            # Check signature to see if .loss() takes arguments. 
            # If it takes 0 args (self), it's likely an Internal Loss model.
            sig = inspect.signature(model.loss)
            if len(sig.parameters) == 0:
                 raise ValueError(
                     "You provided X and y, but the model has a parameter-less .loss() method "
                     "(indicating it holds its own data, like a GP). "
                     "Please set X=None, y=None to use the internal loss."
                 )

        # Default Loss
        if loss_fn is None:
            loss_fn = MeanSquaredError()

        # Introspection: Does the model's __call__ need a key?
        # We check the original model (combined) to inspect the signature
        model_signature = inspect.signature(model.__call__)
        model_needs_key = 'key' in model_signature.parameters

        def compute_loss(params, static, X_batch, y_batch, key):
            # Unwrap is safe here to handle any NonTrainable parts in standard models too
            model = paramax.unwrap(eqx.combine(params, static))
            
            # Conditionally pass key only if model asks for it
            if model_needs_key and key is not None:
                pred = jax.vmap(model, in_axes=(0, None))(X_batch, key)
            else:
                pred = jax.vmap(model)(X_batch)
            
            return loss_fn(pred, y_batch)

        @eqx.filter_jit
        def step(params, opt_state, X_batch, y_batch, key):
            loss_val, grads = eqx.filter_value_and_grad(compute_loss)(
                params, static, X_batch, y_batch, key
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss_val

    # CASE B: Internal Loss (e.g. Gaussian Process)
    else:
        if not hasattr(model, 'loss'):
            raise ValueError("If X, y are not provided, model must have a .loss() method.")

        # Introspection: Does the .loss() method need a key?
        loss_signature = inspect.signature(model.loss)
        loss_needs_key = 'key' in loss_signature.parameters

        def compute_loss(params, static, key):
            model = paramax.unwrap(eqx.combine(params, static))
            
            if loss_needs_key and key is not None:
                return model.loss(key=key)
            return model.loss()

        @eqx.filter_jit
        def step(params, opt_state, key):
            loss_val, grads = eqx.filter_value_and_grad(compute_loss)(params, static, key)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss_val

    # -------------------------------------------------------------------------

    # 3. Training Loop
    loop = tqdm(range(max_steps), disable=not show_progress)
    losses = []
    
    # We prepare the arguments for the step function
    # If X is None, step_args will be empty.
    base_args = (X, y) if (X is not None) else ()

    for _ in loop:
        # Split key if provided
        if key is not None:
            step_key, key = jax.random.split(key)
        else:
            step_key = None

        # Call step. We unpack base_args (X, y or empty) and pass key explicitly.
        # Note: step signature is either (params, opt, X, y, key) or (params, opt, key)
        if X is not None:
            params, opt_state, loss = step(params, opt_state, X, y, step_key)
        else:
            params, opt_state, loss = step(params, opt_state, step_key)
        
        loss_val = loss.item()
        losses.append(loss_val)

        # Early Stopping Logic
        min_idx = jnp.argmin(jnp.array(losses)).item()
        if len(losses) - min_idx - 1 > patience:
            loop.set_postfix_str(f"Early Stopping (Best: {losses[min_idx]:.4f})")
            break
        
        if _ % 10 == 0:
            loop.set_postfix_str(f"Loss: {loss_val:.4f}")

    return eqx.combine(params, static), losses