from typing import Callable, Optional, Tuple, List, Union

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import jaxopt
from jaxopt.base import IterativeSolver
from tqdm.auto import tqdm

from eqxlearn.metrics import mean_squared_error
from eqxlearn.base import BaseModel

def fit(
    model: BaseModel,
    X: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    optimizer: Union[IterativeSolver, optax.GradientTransformation] = None,
    strategy: str = 'default',
    loss_fn: Optional[Union[Callable, eqx.Module]] = None,
    key: Optional[jax.random.PRNGKey] = None,
    max_iter: int | None = None,
    patience: int = 10,
    tol: float = 1e-4,
    show_progress: bool = True,
) -> Tuple[eqx.Module, List[float]]:
    # Inspect model
    has_condition = hasattr(model, 'condition')
    has_solve = hasattr(model, 'solve')
    has_loss = hasattr(model, 'loss')
    
    # Resolve defaults
    if strategy == 'default':
        strategy = model.strategy
        if optimizer is not None and strategy == 'analytical':
            strategy = 'internal-loss' if has_loss else 'external-loss'
    if loss_fn is None:
        loss_fn = mean_squared_error
    if optimizer is None:
        optimizer = jaxopt.LBFGS        

    # Condition and solve the model
    if X is not None:
        if has_condition:
            model = model.condition(X, y)
        if has_solve:
            model = model.solve(X, y)
          
    # Return the model if we don't need to iterate
    if strategy == 'analytical':
        return model, []
        
    # Create the base filter spec, which freezes any analytical models
    def is_analytical_model(node):
        return isinstance(node, BaseModel) and node.strategy == 'analytical'
    def build_filter_spec(node):
        if is_analytical_model(node):
            return False
        return eqx.is_inexact_array(node)
    filter_spec = jax.tree.map(build_filter_spec, model, is_leaf=is_analytical_model)

    # Explicitly freeze X and y (in case they exist inside a trainable model)
    if hasattr(model, 'X') and model.X is not None:
        filter_spec = eqx.tree_at(lambda m: m.X, filter_spec, replace=False)
    if hasattr(model, 'y') and model.y is not None:
        filter_spec = eqx.tree_at(lambda m: m.y, filter_spec, replace=False)
        
    # Partition the model
    params, static = eqx.partition(model, filter_spec)

    # Define the loss function wrapper
    def loss_wrapper_fn(params, X, y=None, key=None):
        model = eqx.combine(params, static)
        if has_loss:
            loss_val = model.loss(key=key)
        else:
            y_pred = jax.vmap(model, in_axes=(0, None))(X, key)
            loss_val = loss_fn(y_pred, y)

        # if jnp.iscomplex(loss_val):
        #     raise Exception("Loss cannot return a complex value")
        return jnp.real(loss_val)
    loss_wrapper_fn_jit = jax.jit(loss_wrapper_fn)

    # Run the specific optimizer backend
    if isinstance(optimizer, optax.GradientTransformation):
        params, losses = _fit_optax(optimizer, params, X, y, loss_wrapper_fn_jit, max_iter=max_iter, patience=patience, show_progress=show_progress, key=key)
    else:
        params, losses = _fit_jaxopt(optimizer, params, X, y, loss_wrapper_fn_jit, max_iter=max_iter, tol=tol, show_progress=show_progress, key=key)

    return eqx.combine(params, static), losses
    
def _fit_jaxopt(solver, params, X, y, loss_fn, *, max_iter=None, tol=1e-4, show_progress=True, key=None):
    solver_kwargs = {'tol': tol}
    if max_iter is not None:
        solver_kwargs['maxiter'] = max_iter
    solver = solver(fun=loss_fn, **solver_kwargs)
    res = solver.run(params, X, y, key)
    final_params = res.params
    final_loss = res.state.value 
    return final_params, [final_loss]
    
def _fit_optax(optimizer, params, X, y, loss_fn, *, max_iter=1000, patience=10, show_progress=True, key=None):
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def step(params, opt_state, X_batch, y_batch, key):
        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(
            params, X_batch, y_batch, key
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss_val

    loop = tqdm(range(max_iter), disable=not show_progress)
    losses = []

    for i in loop:
        step_key = None
        if key is not None:
            step_key, key = jax.random.split(key)

        params, opt_state, loss = step(params, opt_state, X, y, step_key)
        
        loss_val = loss.item()
        losses.append(loss_val)

        # Early stopping
        min_idx = jnp.argmin(jnp.array(losses)).item()
        if len(losses) - min_idx - 1 > patience:
            loop.set_postfix_str(f"Early Stopping (Best: {losses[min_idx]:.4f})")
            break
        
        if i % 10 == 0:
            loop.set_postfix_str(f"Loss: {loss_val:.4f}")

    return params, losses    
