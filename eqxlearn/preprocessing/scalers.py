import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Union

from paramax import non_trainable, unwrap

from eqxlearn.base import Transformer

class StandardScaler(Transformer):
    mean: jnp.ndarray
    scale: jnp.ndarray

    def __init__(self, data: jnp.ndarray = None, mean=None, scale=None):
        """
        Can be initialized from data (calculates stats) OR manually (for inverse).
        """
        if data is not None:
            # Calculate stats over batch dim (0)
            self.mean = non_trainable(jnp.mean(data, axis=0))
            # simple epsilon for stability
            self.scale = non_trainable(jnp.std(data, axis=0) + 1e-6)
        else:
            self.mean = non_trainable(mean)
            self.scale = non_trainable(scale)

    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], **kwargs):
        mu = unwrap(self.mean)
        sigma = unwrap(self.scale)

        # 1. Handle Standard Input (Array)
        if isinstance(x, jnp.ndarray):
            return (x - mu) / sigma
            
        # 2. Handle GP Output (Tuple: Mean, Variance)
        #    This is crucial for the "Inverse Scaler" step at the end.
        elif isinstance(x, tuple) and len(x) == 2:
            val, var = x
            
            # Scale the Mean: (y - mu) / sigma  OR  y*sigma + mu (if inverse)
            # Wait! This logic depends on if we are Forward or Inverse.
            # Since you use the SAME class for both, we just follow the math:
            # Forward: (x - mu)/sigma
            # Inverse: (x - (-mu/sigma)) / (1/sigma) -> x*sigma + mu
            
            # Apply shift/scale to the MEAN
            new_val = (val - mu) / sigma
            
            # Apply SCALE SQUARED to the VARIANCE
            # Var(aX + b) = a^2 * Var(X)
            # Here 'a' is (1/sigma).
            new_var = var / (sigma ** 2)
            
            return new_val, new_var
            
        else:
            raise ValueError(f"Unknown input type: {type(x)}")
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        mu = unwrap(self.mean)
        sigma = unwrap(self.scale)
        return (X - mu) / sigma

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        mu = unwrap(self.mean)
        sigma = unwrap(self.scale)
        return (X * sigma) + mu

    def inverse_scaler(self) -> "StandardScaler":
        """
        Returns a new StandardScaler that performs the inverse operation.
        Useful for appending to the end of a Sequential model.
        """
        # Math: x = y*s + m  --->  (y - (-m/s)) / (1/s)
        mu = unwrap(self.mean)
        sigma = unwrap(self.scale)
        inv_scale = 1.0 / sigma
        inv_mean = -mu / sigma
        return StandardScaler(mean=inv_mean, scale=inv_scale)
    
class MinMaxScaler(Transformer):
    # Statistics are non-trainable state
    scale: jnp.ndarray 
    min_: jnp.ndarray
    
    # Static configuration
    feature_range: Tuple[float, float] = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray = None, 
        scale: jnp.ndarray = None,
        min_: jnp.ndarray = None,
        feature_range: Tuple[float, float] = (0.0, 1.0), 
        eps: float = 1e-8
    ):
        self.feature_range = feature_range
        self.eps = eps
        
        # CASE 1: Initialize from Data (Standard usage)
        if X is not None:
            # 1. Compute Statistics
            data_min = jnp.min(X, axis=0)
            data_max = jnp.max(X, axis=0)
            
            # 2. Compute Scaling Factors
            data_range = data_max - data_min
            # Prevent divide by zero
            data_range = jnp.where(data_range < eps, 1.0, data_range)
            
            scale_val = (feature_range[1] - feature_range[0]) / data_range
            min_val = feature_range[0] - data_min * scale_val
            
            self.scale = non_trainable(scale_val)
            self.min_ = non_trainable(min_val)
            
        # CASE 2: Initialize Manually (Used by inverse_scaler)
        else:
            self.scale = non_trainable(scale)
            self.min_ = non_trainable(min_)

    def __call__(
        self, 
        x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], 
        **kwargs
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Applies scaling. 
        Accepts **kwargs to safely ignore arguments like 'return_var'.
        """
        s = unwrap(self.scale)
        m = unwrap(self.min_)
        
        # 1. Standard Transformation
        if isinstance(x, jnp.ndarray):
            return x * s + m

        # 2. Tuple Transformation (Mean, Variance)
        #    Only handles the tuple if it has exactly 2 elements (mean, var)
        elif isinstance(x, tuple) and len(x) == 2:
            mean, var = x
            
            # Scale Mean: Linear transform (ax + b)
            new_mean = mean * s + m
            
            # Scale Variance: Quadratic transform (a^2 * var)
            # Var(aX + b) = a^2 * Var(X)
            new_var = var * (s ** 2)
            
            return new_mean, new_var
            
        else:
            raise ValueError(f"MinMaxScaler received unknown input type: {type(x)}")

    def inverse_scaler(self) -> "MinMaxScaler":
        """
        Returns a NEW scaler that performs the inverse operation.
        
        Forward: y = x*s + m
        Inverse: x = (y - m) / s 
                   = y*(1/s) + (-m/s)
        
        So the new scaler has:
            scale' = 1/s
            min_'  = -m/s
        """
        s = unwrap(self.scale)
        m = unwrap(self.min_)

        inv_scale = 1.0 / (s + self.eps) # Add eps to avoid div/0
        inv_min = -m * inv_scale
        
        return MinMaxScaler(
            scale=inv_scale, 
            min_=inv_min, 
            feature_range=self.feature_range, 
            eps=self.eps
        )