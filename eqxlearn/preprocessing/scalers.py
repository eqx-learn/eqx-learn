import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple

from eqxlearn.base import Transformer

class StandardScaler(Transformer):
    mean: jnp.ndarray
    scale: jnp.ndarray
    
    # Static configuration
    with_mean: bool = eqx.field(static=True)
    with_std: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray, 
        with_mean: bool = True, 
        with_std: bool = True, 
        eps: float = 1e-8
    ):
        """
        Args:
            X: The data used to compute mean and standard deviation.
        """
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = eps

        # 1. Compute Mean
        if with_mean:
            self.mean = jnp.mean(X, axis=0)
        else:
            self.mean = jnp.zeros(X.shape[1])

        # 2. Compute Scale
        if with_std:
            std = jnp.std(X, axis=0)
            # Avoid division by zero
            self.scale = jnp.where(std < eps, 1.0, std)
        else:
            self.scale = jnp.ones(X.shape[1])

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.mean) / self.scale

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X * self.scale) + self.mean
    
class MinMaxScaler(Transformer):
    min_val: jnp.ndarray
    max_val: jnp.ndarray
    scale: jnp.ndarray
    min_: jnp.ndarray
    
    feature_range: Tuple[float, float] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray, 
        feature_range: Tuple[float, float] = (0.0, 1.0), 
        eps: float = 1e-8
    ):
        self.feature_range = feature_range
        self.eps = eps
        
        # 1. Compute Statistics from X
        self.min_val = jnp.min(X, axis=0)
        self.max_val = jnp.max(X, axis=0)
        
        # 2. Pre-compute scaling factors (optimization)
        data_range = self.max_val - self.min_val
        data_range = jnp.where(data_range < eps, 1.0, data_range)
        
        self.scale = (feature_range[1] - feature_range[0]) / data_range
        self.min_ = feature_range[0] - self.min_val * self.scale

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return X * self.scale + self.min_

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X - self.min_) / self.scale