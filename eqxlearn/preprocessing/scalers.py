import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Union, Self
from dataclasses import replace

from eqxlearn.base import InvertibleTransformer

EPSILON = 1E-8

class StandardScaler(InvertibleTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    # Learned State
    mean: Optional[jnp.ndarray] = None
    scale: Optional[jnp.ndarray] = None
    
    eps: float = eqx.field(static=True)

    def __init__(self, mean=None, scale=None, eps: float = 1e-8):
        self.mean = mean
        self.scale = scale
        self.eps = eps

    def solve(self, X: jnp.ndarray, y: jnp.ndarray = None) -> Self:
        """
        Computes mean and std from X (N, D).
        """
        # Calculate stats over batch dim (0)
        mu = jnp.mean(X, axis=0)
        # Simple epsilon for stability
        sigma = jnp.std(X, axis=0) + self.eps

        return replace(self, mean=mu, scale=sigma)

    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs):
        """
        Forward transformation (Single Sample).
        x: (D,) Array OR ((D,), (D,)) Tuple
        """
        mu = self.mean
        sigma = self.scale

        # 1. Array Input
        if isinstance(x, jnp.ndarray):
            return (x - mu) / sigma
            
        # 2. Tuple Input (Mean, Variance)
        elif isinstance(x, tuple) and len(x) == 2:
            val, var = x
            new_val = (val - mu) / sigma
            new_var = var / (sigma ** 2)
            return new_val, new_var
            
        else:
            raise ValueError(f"Unknown input type: {type(x)}")
    
    def inverse(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs):
        """
        Inverse transformation (Single Sample).
        """
        mu = self.mean
        sigma = self.scale

        # 1. Array Input
        if isinstance(x, jnp.ndarray):
            return (x * sigma) + mu
        
        # 2. Tuple Input
        elif isinstance(x, tuple) and len(x) == 2:
            val, var = x
            new_val = (val * sigma) + mu
            new_var = var * (sigma ** 2)
            return new_val, new_var
        
        raise ValueError(f"Unknown input type: {type(x)}")

    def inverse_scaler(self) -> "StandardScaler":
        """
        Returns a NEW StandardScaler that performs the inverse operation.
        Math: x_new = (x_old - (-mu/sigma)) / (1/sigma)
        """
        mu = self.mean
        sigma = self.scale
        
        inv_scale = 1.0 / sigma
        inv_mean = -mu / sigma
        
        return replace(self, mean=inv_mean, scale=inv_scale)


class MinMaxScaler(InvertibleTransformer):
    """
    Transform features by scaling each feature to a given range.
    """
    # Learned State
    scale: Optional[jnp.ndarray] = None
    min: Optional[jnp.ndarray] = None
    
    # Configuration
    feature_range: Tuple[float, float] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self,
        feature_range: Tuple[float, float] = (0.0, 1.0), 
        eps: float = 1e-8,
        scale=None,
        min=None
    ):
        self.feature_range = feature_range
        self.eps = eps
        self.scale = scale
        self.min = min

    def solve(self, X: jnp.ndarray, y: jnp.ndarray = None) -> Self:
        # 1. Compute Statistics
        data_min = jnp.min(X, axis=0)
        data_max = jnp.max(X, axis=0)
        
        # 2. Compute Scaling Factors
        data_range = data_max - data_min
        data_range = jnp.where(data_range < self.eps, 1.0, data_range)
        
        scale_val = (self.feature_range[1] - self.feature_range[0]) / data_range
        min_val = self.feature_range[0] - data_min * scale_val
        
        return replace(self, scale=scale_val, min=min_val)

    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs):
        """
        Forward transformation (Single Sample).
        """
        s = self.scale
        m = self.min
        
        if isinstance(x, jnp.ndarray):
            return x * s + m

        elif isinstance(x, tuple) and len(x) == 2:
            mean, var = x
            new_mean = mean * s + m
            new_var = var * (s ** 2)
            return new_mean, new_var
            
        else:
            raise ValueError(f"MinMaxScaler received unknown input type: {type(x)}")

    def inverse(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs):
        """
        Inverse transformation (Single Sample).
        """
        s = self.scale
        m = self.min
        
        # Inverse of y = x*s + m  =>  x = (y - m) / s
        
        if isinstance(x, jnp.ndarray):
            return (x - m) / (s + self.eps)
            
        elif isinstance(x, tuple) and len(x) == 2:
            mean, var = x
            new_mean = (mean - m) / (s + self.eps)
            new_var = var / ((s ** 2) + self.eps)
            return new_mean, new_var

        raise ValueError(f"MinMaxScaler received unknown input type: {type(x)}")

    def inverse_scaler(self) -> "MinMaxScaler":
        """
        Returns a NEW scaler that performs the inverse operation.
        """
        s = self.scale
        m = self.min

        inv_scale = 1.0 / (s + self.eps)
        inv_min = -m * inv_scale
        
        return replace(self, scale=inv_scale, min=inv_min)