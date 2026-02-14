from typing import Callable, Optional, Tuple, List, Union
import jax
import jax.numpy as jnp
import equinox as eqx
import inspect
import optax
from tqdm.auto import tqdm
from dataclasses import replace

from eqxlearn.metrics import mean_squared_error
from eqxlearn.base import BaseModel

def fit(
    model: BaseModel,
    X: Optional[jnp.ndarray] = None,
    y: Optional[jnp.ndarray] = None,
    *,
    solution: str = 'default', 
    learning_rate: float | None = None,
    optimizer: Optional[optax.GradientTransformation] = None,
    loss_fn: Optional[Union[Callable, eqx.Module]] = None,
    key: Optional[jax.random.PRNGKey] = None,
    max_iter: int = 1000,
    patience: int = 10,
    show_progress: bool = True,
) -> Tuple[eqx.Module, List[float]]:
    # Input validation
    if optimizer is not None and learning_rate is not None:
        raise ValueError("Cannot pass 'optimizer' and 'learning_rate' to 'fit'")    
    
    # Defaults
    if learning_rate is not None:
        optimizer = optax.adam(learning_rate=learning_rate)
    
    # Condition and solve on the data, if the model supports its.
    if X is not None and y is not None:
        # Condition the model. Used by Bayesian models (GPs) to update belief state/store data.
        if hasattr(model, 'condition'):
            model = model.condition(X, y)

        # Solve the model. Used by Analytical models (LinearReg, Scalers) to compute exact solutions.
        # Note: A model could theoretically do both, e.g. Bayesian LinReg computing MAP.
        if hasattr(model, 'solve'):
            model = model.solve(X, y)
            
    # Determine if we should run an optimization loop
    should_iterate = False
    if solution == 'iterative':
        should_iterate = True
    elif solution == 'analytic':
        should_iterate = False
    elif solution == 'default':
        has_loss = hasattr(model, 'loss')
        user_wants_opt = optimizer is not None
        if has_loss or user_wants_opt:
            should_iterate = True
    if not should_iterate:
        return model, []

    # Setup the optimization
    if optimizer is None:
        optimizer = optax.adam(learning_rate=0.1)
    
    # --- Partitioning (Freezing Data) ---
    filter_spec = jax.tree.map(lambda x: eqx.is_inexact_array(x), model)
    # Freeze X
    if hasattr(model, 'X') and model.X is not None:
        filter_spec = eqx.tree_at(lambda m: m.X, filter_spec, replace=False)
    # Freeze y
    if hasattr(model, 'y') and model.y is not None:
        filter_spec = eqx.tree_at(lambda m: m.y, filter_spec, replace=False)

    params, static = eqx.partition(model, filter_spec)
    opt_state = optimizer.init(params)
    
    # --- Introspection ---
    model_call_sig = inspect.signature(model.__call__)
    call_needs_key = 'key' in model_call_sig.parameters
    
    has_internal_loss = hasattr(model, 'loss')
    loss_needs_key = False
    if has_internal_loss:
        loss_needs_key = 'key' in inspect.signature(model.loss).parameters

    if loss_fn is None:
        loss_fn = mean_squared_error

    def compute_loss(params, static, X_batch, y_batch, key):
        model = eqx.combine(params, static)
        
        # Scenario 1: Internal Loss (GP) - Data is inside
        if has_internal_loss:
            return model.loss(key=key) if loss_needs_key and key is not None else model.loss()
            
        # Scenario 2: External Loss (NN) - Need data
        # Prioritize batch data, fallback to internal model data
        if X_batch is not None:
            X_in, y_true = X_batch, y_batch
        elif hasattr(model, 'X') and hasattr(model, 'y') and model.X is not None:
            X_in, y_true = model.X, model.y
        else:
            raise ValueError("Iterative training requested but no data available (neither passed nor internal).")

        if call_needs_key and key is not None:
            pred = jax.vmap(model, in_axes=(0, None))(X_in, key)
        else:
            pred = jax.vmap(model)(X_in)
        return loss_fn(pred, y_true)

    @eqx.filter_jit
    def step(params, opt_state, X_batch, y_batch, key):
        loss_val, grads = eqx.filter_value_and_grad(compute_loss)(
            params, static, X_batch, y_batch, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_val

    # Loop
    loop = tqdm(range(max_iter), disable=not show_progress)
    losses = []

    for i in loop:
        step_key = None
        if key is not None:
            step_key, key = jax.random.split(key)

        # We pass X and y (which might be None if injected) to step
        # compute_loss logic handles finding the data
        params, opt_state, loss = step(params, opt_state, X, y, step_key)
        
        loss_val = loss.item()
        losses.append(loss_val)

        min_idx = jnp.argmin(jnp.array(losses)).item()
        if len(losses) - min_idx - 1 > patience:
            loop.set_postfix_str(f"Early Stopping (Best: {losses[min_idx]:.4f})")
            break
        
        if i % 10 == 0:
            loop.set_postfix_str(f"Loss: {loss_val:.4f}")

    return eqx.combine(params, static), losses