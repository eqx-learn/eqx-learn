import jax
import jax.numpy as jnp
import equinox as eqx
from paramax import non_trainable, unwrap

from eqxlearn.base import Regressor
from eqxlearn.gaussian_process.kernels import Kernel

class GaussianProcessRegressor(Regressor):
    """
    Standard GP Regressor.
    Observation noise should be defined inside the kernel (e.g. kernel + WhiteKernel).
    """
    kernel: Kernel
    X: jnp.ndarray
    y: jnp.ndarray
    
    # Jitter is strictly for numerical stability (not trained)
    jitter: float = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray, 
        y: jnp.ndarray, 
        kernel: Kernel, 
        jitter: float = 1e-10
    ):
        """
        Args:
            X: (N, D) Input data
            y: (N,) Target data (1D)
            kernel: Kernel instance
            jitter: Small float added to diagonal for Cholesky stability.
        """
        N, _D = X.shape
        if len(y.shape) != 1 and y.shape[0] != N:
            raise Exception("Incompatible shapes for X and y passed")

        # We use non_trainable to prevent X, y from being updated by the optimizer
        self.X = non_trainable(X)
        self.y = non_trainable(y)
        self.kernel = kernel
        self.jitter = jitter

    def __call__(self, x: jnp.ndarray, return_var: bool = False):
        """
        Predicts mean and variance for a single test point x.
        """
        # Unwrap data to access arrays
        X, y = unwrap(self.X), unwrap(self.y)

        N = X.shape[0]
        k_fn = lambda x1, x2: self.kernel(x1, x2)

        # 1. Compute Kernel Matrix (includes WhiteNoise if present in kernel)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        # 2. Add Jitter (Stability only)
        K_y = K + self.jitter * jnp.eye(N)
        L = jnp.linalg.cholesky(K_y)

        # 3. Compute Cross-Covariance (k_star)
        # Note: If using WhiteKernel, it correctly returns 0 here for x != X
        k_star = jax.vmap(lambda x_train: k_fn(x, x_train))(X)

        # 4. Calculate Mean
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        mu = jnp.dot(k_star, alpha)

        if not return_var:
            return mu

        # 5. Calculate Variance
        # Note: If using WhiteKernel, this includes observation noise variance 
        # (predictive posterior), effectively p(y* | y), not p(f* | y).
        k_star_star = k_fn(x, x)
        v = jax.scipy.linalg.solve_triangular(L, k_star, lower=True)
        var = k_star_star - jnp.dot(v, v)
        
        # Clip variance for stability
        return mu, jnp.maximum(var, 1e-12)
    
    def loss(self):
        """Computes Negative Log Marginal Likelihood (NLML)."""
        X, y = unwrap(self.X), unwrap(self.y)
        N = X.shape[0]
        
        # 1. Compute Kernel Matrix
        k_fn = lambda x1, x2: self.kernel(x1, x2)
        K = jax.vmap(lambda x1: jax.vmap(lambda x2: k_fn(x1, x2))(X))(X)
        
        # 2. Add Jitter
        K_y = K + self.jitter * jnp.eye(N)
        
        # 3. Cholesky & Solve
        L = jnp.linalg.cholesky(K_y)
        z = jax.scipy.linalg.solve_triangular(L, y, lower=True)
        alpha = jax.scipy.linalg.solve_triangular(L.T, z, lower=False)
        
        # 4. Compute NLML
        data_fit = 0.5 * jnp.dot(y, alpha)
        complexity = jnp.sum(jnp.log(jnp.diag(L)))
        constant = 0.5 * N * jnp.log(2 * jnp.pi)
        
        return data_fit + complexity + constant