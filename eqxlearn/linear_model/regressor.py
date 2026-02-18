from typing import Self, Optional, Union
from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
from eqxlearn.base import Regressor

class LinearRegressor(Regressor):
    """
    Ordinary Least Squares (OLS) Linear Regression.
    
    State:
        weight: (D,) array of coefficients.
        bias: Scalar bias/intercept.
    """
    weight: Optional[jnp.ndarray] = None
    bias: Optional[jnp.ndarray] = None

    def __init__(self, weight=None, bias=None):
        self.weight = weight
        self.bias = bias
    
    def solve(self, X: jnp.ndarray, y: jnp.ndarray) -> Self:
        """
        Solves the Linear Regression problem analytically using Least Squares.
        Mathematically equivalent to minimizing ||Ax - y||^2.
        
        Args:
            X: Input data (N, D) or (N,)
            y: Target data (N,)
        Returns:
            New instance of LinearRegressor with optimal parameters.
        """
        # 1. Robust Shape Handling
        # Ensure X is (N, D)
        if X.ndim == 1:
            X = X[:, None]
            
        # Ensure y is (N,)
        if y.ndim > 1:
            y = y.ravel()

        N, D = X.shape

        # 2. Prepare Design Matrix A = [X, 1]
        # We append a column of ones to handle the bias term
        ones = jnp.ones((N, 1))
        A = jnp.concatenate([X, ones], axis=1)
        
        # 3. Solve Ax = y
        # rcond=None allows JAX to use machine precision defaults for singular values
        theta, residuals, rank, s = jnp.linalg.lstsq(A, y, rcond=None)
        
        # 4. Extract Parameters
        # theta is shape (D+1,). Last element is the bias.
        new_weight = theta[:-1]
        new_bias = theta[-1]
        
        # 5. Return new model (Immutable update)
        return replace(self, weight=new_weight, bias=new_bias)
    
    def __call__(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        """
        Forward pass for a SINGLE sample x.
        x: (D,) array
        """
        if self.weight is None or self.bias is None:
            raise RuntimeError("LinearRegressor is not initialized. Call solve() first.")
            
        # Linear equation: y = w.x + b
        return jnp.dot(x, self.weight) + self.bias