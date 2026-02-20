from typing import Callable, Optional, Tuple, List, Union
import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import jaxopt
from jaxopt.base import IterativeSolver
from tqdm.auto import tqdm

from eqxlearn.metrics import mean_squared_error
from eqxlearn.base import BaseModel, Transformer

logger = logging.getLogger(__file__)

def fit(
    model: BaseModel,
    X: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    *,
    bounds: Optional[tuple[BaseModel, BaseModel]] = None,
    optimizer = None,
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
        optimizer = jaxopt.ScipyBoundedMinimize

    # Condition and solve the model
    if X is not None:
        if has_condition:
            model = model.condition(X, y)
        if has_solve:
            model = model.solve(X, y)
          
    # Return the model if we don't need to iterate
    if strategy == 'analytical':
        return model, []
    
    # TODO: improve. Gemini has suggested the magical _inject_bound() below but afraid to use it at the moment.
    # If we have bounds, condition and solve them too. Not sure of a better way to do this at the moment
    # since .condition(X, y) and .solve(X, y) can fundamentally change the model structure.
    if bounds is not None and X is not None:
        if has_condition:
            bounds = (bounds[0].condition(X, y), bounds[1].condition(X, y))
        if has_solve:
            bounds = (bounds[0].solve(X, y), bounds[1].solve(X, y))
        
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
        
    # Partition the model and bounds
    params, static = eqx.partition(model, filter_spec)
    if bounds is not None:
        lb_params, _ = eqx.partition(bounds[0], filter_spec)
        ub_params, _ = eqx.partition(bounds[1], filter_spec)
        bounds = (lb_params, ub_params)

    # Define the loss function wrapper
    def loss_wrapper_fn(params, X, y=None, key=None):
        model = eqx.combine(params, static)
        if has_loss:
            loss_val = model.loss(key=key)
        else:
            y_pred = jax.vmap(model, in_axes=(0, None))(X, key)
            loss_val = loss_fn(y_pred, y)
        return jnp.real(loss_val)
    loss_wrapper_fn_jit = jax.jit(loss_wrapper_fn)    

    # Run the specific optimizer backend
    if isinstance(optimizer, optax.GradientTransformation):
        if bounds is not None:
            raise ValueError("Optax does not support bounds")
        
        params, losses = _fit_optax(optimizer, params, X, y, loss_wrapper_fn_jit, max_iter=max_iter, patience=patience, show_progress=show_progress, key=key)
    else:
        params, losses = _fit_jaxopt(optimizer, params, X, y, loss_wrapper_fn_jit, bounds=bounds, max_iter=max_iter, tol=tol, key=key)
        
    # Check for bounds saturation
    if bounds is not None:
        lb_params, ub_params = bounds
        saturated_paths = []

        def check_boundaries(path, p, lb, ub):
            if p is None:
                return
            
            # Use a tiny tolerance to account for floating-point inaccuracies
            hit_lower = jnp.any(jnp.isclose(p, lb))
            hit_upper = jnp.any(jnp.isclose(p, ub))
            
            # Pull the boolean back to CPU to evaluate in standard Python
            if jax.device_get(hit_lower) or jax.device_get(hit_upper):
                # Format the PyTree path cleanly (e.g., "kernel.lengthscale")
                path_str = ".".join(
                    str(getattr(k, 'name', getattr(k, 'key', getattr(k, 'idx', '?')))) 
                    for k in path
                )
                saturated_paths.append(path_str)

        # Map over the tree to populate the squashed_paths list
        jax.tree_util.tree_map_with_path(check_boundaries, params, lb_params, ub_params)

        if saturated_paths:
            logger.warning(f"Some parameters were found to be close to their bounds. Paths: {saturated_paths}.")    

    return eqx.combine(params, static), losses

def fit_transform(
    model: Transformer,
    X: jnp.ndarray,
    y: Optional[jnp.ndarray] = None,
    **kwargs
) -> jnp.ndarray:
    fitted_model, _ = fit(model, X, y, **kwargs)
    return fitted_model.transform(X)
    
def _fit_jaxopt(solver, params, X, y, loss_fn, *, bounds=None, max_iter=None, tol=1e-4, show_progress=True, key=None):
    solver_kwargs = {'tol': tol}
    if max_iter is not None:
        solver_kwargs['maxiter'] = max_iter
        
    opt = solver(fun=loss_fn, **solver_kwargs)
    
    # ScipyBoundedMinimize takes `bounds` as the second argument
    if bounds is not None:
        res = opt.run(params, bounds, X, y, key=key)
    else:
        res = opt.run(params, X, y, key=key)
        
    final_params = res.params
    # Handle JAXopt's inconsistent state naming between native and Scipy solvers
    final_loss = getattr(res.state, 'fun_val', getattr(res.state, 'value', None))
    
    return final_params, [final_loss]
    
def _fit_optax(optimizer, params, X, y, loss_fn, *, max_iter=None, patience=None, show_progress=True, key=None):
    if max_iter is None:
        raise Exception("Must pass max_iter when using optax")
    if patience is None:
        raise Exception("Must pass patience when using optax")
    
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

def _inject_bound(params_skeleton, bound_model, default_val):
    # "Magic" solution which attempts to match paths that "look" the same in the model and bounds model
    
    # 1. Flatten the bounds and extract pure attribute paths
    bound_leaves, _ = jax.tree_util.tree_flatten_with_path(bound_model)
    
    bound_dict = {}
    for path, leaf in bound_leaves:
        if leaf is None:
            continue
            
        # Extract string names, completely ignoring sequence indices (like [0] or batch dims)
        names = []
        for p in path:
            if isinstance(p, jax.tree_util.GetAttrKey):
                names.append(p.name)
            elif isinstance(p, jax.tree_util.DictKey):
                names.append(str(p.key))
                
        if not names:
            continue
            
        # Reverse to create a leaf-to-root sequence (e.g. ('lengthscale', 'kernel', 'estimator'))
        suffix = tuple(reversed(names))
        bound_dict[suffix] = leaf

    # 2. Map over the conditioned skeleton using suffix matching
    def map_fn(path, leaf):
        if leaf is None:
            return None

        # Extract skeleton path names in the exact same way
        names = []
        for p in path:
            if isinstance(p, jax.tree_util.GetAttrKey):
                names.append(p.name)
            elif isinstance(p, jax.tree_util.DictKey):
                names.append(str(p.key))
                
        target_suffix = tuple(reversed(names))
        
        best_match_val = None
        best_match_score = 0
        
        # Find the bound with the deepest matching leaf-to-root path
        for bound_suffix, bound_val in bound_dict.items():
            score = 0
            for ts, bs in zip(target_suffix, bound_suffix):
                if ts == bs:
                    score += 1
                else:
                    break
                    
            # Ensure at least the actual parameter name matches (score > 0)
            if score > best_match_score:
                best_match_score = score
                best_match_val = bound_val
                
        if best_match_val is not None:
            # Automatically broadcasts scalar bounds to stacked/vmapped arrays!
            return jnp.broadcast_to(best_match_val, jnp.shape(leaf))
            
        return jnp.full_like(leaf, default_val)

    return jax.tree_util.tree_map_with_path(map_fn, params_skeleton)