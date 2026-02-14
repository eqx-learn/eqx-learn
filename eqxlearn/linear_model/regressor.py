from typing import Self, Optional
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from eqxlearn.base import Regressor

class LinearRegressor(Regressor):
    weight: Optional[jnp.ndarray] = None
    bias: Optional[jnp.ndarray] = None

    def __init__(self, in_features: Optional[int] = None, key = None):
        """
        Args:
            in_features: Number of input features. If None, model is uninitialized
                         until solve() is called or trained iteratively.
            key: PRNGKey for random initialization.
        """
        # Initialize small random weights to break symmetry if trained iteratively
        if in_features is not None:
            if key is None: key = jax.random.PRNGKey(0)
            self.weight = jax.random.normal(key, (in_features,)) * 0.01
            # Bias scalar (or matching shape if multi-output, but usually scalar for standard LinReg)
            self.bias = jnp.zeros(()) 
        else:
            self.weight = None
            self.bias = None
    
    def condition(self, X: jnp.ndarray, y: jnp.ndarray) -> Self:
        """
        Solves the Linear Regression problem analytically using Least Squares.
        Mathematically equivalent to: w = (X^T X)^-1 X^T y
        
        Args:
            X: Input data (N, D)
            y: Target data (N,)
        Returns:
            New instance of LinearRegressor with optimal weights conditioned on the data.
        """
        # 1. Prepare Design Matrix A = [X, 1]
        N = X.shape[0]
        ones = jnp.ones((N, 1))
        A = jnp.concatenate([X, ones], axis=1)
        
        # 2. Solve Ax = y
        # lstsq returns (solution, residuals, rank, singular_values)
        theta, _, _, _ = jnp.linalg.lstsq(A, y)
        
        # 3. Extract Parameters
        new_weight = theta[:-1]
        new_bias = theta[-1]
        
        # 4. Return new model (Immutable update)
        return replace(self, weight=new_weight, bias=new_bias)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.weight, x) + self.bias
    
    def predict(self, X: jnp.ndarray):
        return jax.vmap(self)(X)    