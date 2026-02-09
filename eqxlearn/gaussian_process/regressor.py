import jax
import jax.numpy as jnp
import equinox as eqx
from paramax import non_trainable

from eqxlearn.base import Regressor
from eqxlearn.gaussian_process.kernels import Kernel

class GaussianProcessRegressor(Regressor):
    """
    Standard GP Regressor for a single output dimension.
    """
    kernel: Kernel
    log_obs_noise: jnp.ndarray
    
    X: jnp.ndarray
    y: jnp.ndarray

    def __init__(self, X: jnp.ndarray, y: jnp.ndarray, kernel: Kernel, obs_noise: float = 1e-4):
        """
        Args:
            X: (N, D) Input data
            y: (N,) Target data (1D)
            kernel: Kernel instance
            obs_noise: Observation noise (jitter/likelihood noise)
        """
        N, _D = X.shape
        if len(y.shape) != 1 and y.shape[0] != N:
            raise Exception("Incompatible shapes for X and y passed")

        self.X = non_trainable(X)
        self.y = non_trainable(y)
        self.kernel = kernel
        self.log_obs_noise = jnp.log(obs_noise)

    def __call__(self, x: jnp.ndarray, return_var: bool = False):
        """
        Predicts mean and variance for a single test point x_star.
        """
        N = self.X.shape[0]
        k_fn = lambda x1, x2: self.kernel(x1, x2)

        # Recompute K and L (in a real setting, you might cache these)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(self.X))(self.X)
        K_y = K + jnp.exp(self.log_obs_noise) * jnp.eye(N)
        L = jnp.linalg.cholesky(K_y)

        # Compute k_star (covariance between x_star and training data)
        k_star = jax.vmap(lambda x: k_fn(x, x))(self.X)

        # Calculate Mean: mu = k_star^T K^-1 y
        z = jax.scipy.linalg.solve_triangular(L, self.y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        mu = jnp.dot(k_star, alpha)

        if not return_var:
            return mu

        # Calculate Variance: k(x*, x*) - k_star^T K^-1 k_star
        k_star_star = k_fn(x, x)
        v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)
        var = k_star_star - jnp.dot(v, v)
        
        # Clip variance for stability
        return mu, var
    
    def loss(self):
        """Computes Negative Log Marginal Likelihood (NLML)."""
        N = self.X.shape[0]
        
        # 1. Compute Kernel Matrix
        k_fn = lambda x1, x2: self.kernel(x1, x2)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(self.X))(self.X)
        
        # 2. Add Observation Noise (Jitter)
        K_y = K + jnp.exp(self.log_obs_noise) * jnp.eye(N)
        
        # 3. Cholesky Decomposition
        L = jnp.linalg.cholesky(K_y)
        
        # 4. Solve for alpha = K^-1 y
        # L z = y  => z = L^-1 y
        z = jax.scipy.linalg.solve_triangular(L, self.y, lower=True)
        # L.T alpha = z => alpha = L^-T z
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        # 5. Compute NLML
        # Term 1: Data fit (0.5 * y^T * K^-1 * y)
        data_fit = 0.5 * jnp.dot(self.y, alpha)
        # Term 2: Complexity (sum log diag L)
        complexity = jnp.sum(jnp.log(jnp.diag(L)))
        # Term 3: Constant
        constant = 0.5 * N * jnp.log(2 * jnp.pi)
        
        return data_fit + complexity + constant    