import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Self
from dataclasses import replace

from eqxlearn.base import Regressor
from eqxlearn.gaussian_process.kernels import Kernel

class GaussianProcessRegressor(Regressor):
    """
    Standard GP Regressor.
    Observation noise should be defined inside the kernel (e.g. kernel + WhiteKernel).
    """
    kernel: Kernel
    X: Optional[jnp.ndarray]
    y: Optional[jnp.ndarray]
    
    # Jitter is strictly for numerical stability (not trained)
    jitter: float = eqx.field(static=True)

    def __init__(
        self, 
        kernel: Kernel = None, 
        jitter: float = 1e-10,
        X: Optional[jnp.ndarray] = None, 
        y: Optional[jnp.ndarray] = None, 
    ):
        """
        Args:
            X: (N, D) Input data. Optional (can be passed to fit).
            y: (N,) Target data. Optional (can be passed to fit).
            kernel: Kernel instance.
            jitter: Small float added to diagonal for Cholesky stability.
        """
        self.kernel = kernel
        self.jitter = jitter
        self.X = X
        self.y = y

        if y is not None and X is not None:
            N, _D = X.shape
            if len(y.shape) != 1 or y.shape[0] != N:
                raise ValueError("Incompatible shapes for X and y passed")

    def condition(self, X: jnp.ndarray, y: jnp.ndarray) -> Self:
        """
        Conditions the GP prior on the observed data (X, y).
        Returns the posterior GP.
        """
        return replace(self, X=X, y=y) 
            
    def __call__(self, x: jnp.ndarray, return_var: bool = False):
        """Predicts mean and variance for a single test point x."""
        X, y = self.X, self.y

        N = X.shape[0]
        k_fn = lambda x1, x2: self.kernel(x1, x2)

        # 1. Compute Kernel Matrix
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        # 2. Add Jitter
        K_y = K + self.jitter * jnp.eye(N)
        L = jnp.linalg.cholesky(K_y)

        # 3. Compute Cross-Covariance
        k_star = jax.vmap(lambda x_train: k_fn(x, x_train))(X)

        # 4. Calculate Mean
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        mu = jnp.dot(k_star, alpha)

        if not return_var:
            return mu

        # 5. Calculate Variance
        k_star_star = k_fn(x, x)
        v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)
        var = k_star_star - jnp.dot(v, v)
        
        return mu, jnp.maximum(var, 1e-12)
    
    def loss(self):
        """Computes Negative Log Marginal Likelihood (NLML)."""
        X, y = self.X, self.y
        N = X.shape[0]
        
        k_fn = lambda x1, x2: self.kernel(x1, x2)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        K_y = K + self.jitter * jnp.eye(N)
        L = jnp.linalg.cholesky(K_y)
        
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        data_fit = 0.5 * jnp.dot(y, alpha)
        complexity = jnp.sum(jnp.log(jnp.diag(L)))
        constant = 0.5 * N * jnp.log(2 * jnp.pi)
        
        return data_fit + complexity + constant