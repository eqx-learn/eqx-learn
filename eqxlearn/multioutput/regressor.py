import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Union, Tuple

from eqxlearn.base import Regressor

class _MultiOutputBase(Regressor):
    """
    Base class for Multi-Output regression. 
    Handles prediction and forward pass, but NOT loss.
    """
    batched_model: eqx.Module

    def __init__(
        self, 
        X: jnp.ndarray, 
        Y: jnp.ndarray, 
        make_regressor_fn: Callable[[jnp.ndarray, jnp.ndarray], eqx.Module]
    ):
        N, M = Y.shape
        # Instantiate M independent models
        list_of_models = [make_regressor_fn(X, Y[:, i]) for i in range(M)]
        
        # Stack them into a single Batched PyTree
        # The resulting object looks like the original model, 
        # but every array leaf has an extra leading dimension (M).
        self.batched_model = jax.tree.map(lambda *args: jnp.stack(args), *list_of_models)

    def __call__(self, x: jnp.ndarray, **kwargs):
        """
        Forward pass for a SINGLE data point x.
        Returns: (M,) vector.
        """
        # We vmap over the models (axis 0 of batched_model), 
        # but broadcast 'x' (None) so it feeds into all models.
        return jax.vmap(lambda m: m(x, **kwargs))(self.batched_model)

    def predict(self, X: jnp.ndarray, **kwargs):
        """
        Returns stacked predictions for all outputs.
        Output Shape: (N, M)
        """
        # 1. Run prediction. Result shape is (M, N) because M is the batch dim
        outs = jax.vmap(lambda m: m.predict(X, **kwargs))(self.batched_model)
        
        # 2. Helper to transpose (M, N) -> (N, M)
        def _transpose(arr):
            if isinstance(arr, jnp.ndarray) and arr.ndim == 2:
                return arr.T
            return arr

        # 3. Handle Tuple outputs (e.g. mean, var)
        if isinstance(outs, tuple):
            return tuple(_transpose(o) for o in outs)
            
        return _transpose(outs)


class _MultiOutputInternal(_MultiOutputBase):
    """
    Subclass specifically for models that have an internal loss (like GPs).
    This exposes the .loss() method so the fit() function can find it.
    """
    def loss(self):
        """Computes the sum of losses across all M estimators."""
        def single_loss(model):
            return model.loss()
            
        # vmap over the M models
        all_losses = jax.vmap(single_loss)(self.batched_model)
        return jnp.sum(all_losses)


def MultiOutputRegressor(
    X: jnp.ndarray, 
    Y: jnp.ndarray, 
    make_regressor_fn: Callable[[jnp.ndarray, jnp.ndarray], eqx.Module]
) -> Union[_MultiOutputBase, _MultiOutputInternal]:
    """
    Factory function that returns the correct MultiOutput wrapper.
    
    - If the base model has a .loss() method (e.g. GP), it returns a wrapper with .loss().
    - If the base model does NOT (e.g. LinearReg), it returns a wrapper without it.
    """
    # 1. Instantiate a dummy model to inspect it
    dummy_model = make_regressor_fn(X, Y[:, 0])
    
    # 2. Check if it implements the Internal Loss protocol
    has_internal_loss = hasattr(dummy_model, 'loss')
    
    # 3. Return the appropriate class instance
    if has_internal_loss:
        return _MultiOutputInternal(X, Y, make_regressor_fn)
    else:
        return _MultiOutputBase(X, Y, make_regressor_fn)