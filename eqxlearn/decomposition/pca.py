import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Self, Union, Any
from dataclasses import replace

from eqxlearn.base import InvertibleTransformer

class PCA(InvertibleTransformer):
    """
    Principal Component Analysis (PCA).
    
    Operates in two mutually exclusive modes set during __init__:
    1. Fixed Mode: Specify `n_components` to keep exactly N features.
    2. Threshold Mode: Specify `error_threshold` to keep features until 
       reconstruction error is below this value.
    """
    # --- Learned State ---
    components: Optional[jnp.ndarray] = None  # Shape: (n_selected, n_features)
    mean: Optional[jnp.ndarray] = None        # Shape: (n_features,)
    input_shape: Optional[Tuple[int, ...]] = eqx.field(static=True, default=None)

    # --- Hyperparameters ---
    n_components: Optional[int] = eqx.field(static=True, default=None)
    error_threshold: Optional[float] = eqx.field(static=True, default=None)
    min_components: Optional[int] = eqx.field(static=True, default=None)
    max_components: Optional[int] = eqx.field(static=True, default=None)
    
    # Outputs
    noise_variance: Optional[jnp.ndarray] = None

    def __init__(
        self, 
        n_components: Optional[int] = None, 
        error_threshold: Optional[float] = None,
        min_components: Optional[int] = None,
        max_components: Optional[int] = None,
        # State args for reconstruction/deserialization
        components: Optional[jnp.ndarray] = None,
        mean: Optional[jnp.ndarray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        noise_variance: Optional[Tuple[int, ...]] = None,
    ):
        """
        Args:
            n_components: Exact number of components to keep.
            error_threshold: Maximum allowed reconstruction error (MAE). 
                             If set, components are added iteratively.
            max_components: Upper limit on components when using error_threshold.
        """
        if n_components is not None and error_threshold is not None and components is None:
            raise ValueError("Ambiguous configuration: Please provide either 'n_components' OR 'error_threshold', not both.")

        self.n_components = n_components
        self.error_threshold = error_threshold
        self.min_components = min_components
        self.max_components = max_components
        
        # Initialize state
        self.components = components
        self.mean = mean
        self.input_shape = input_shape
        
        self.noise_variance = noise_variance
        
    def solve(self, X: jnp.ndarray, y: jnp.ndarray = None) -> Self:
        """
        Fits the PCA model to X.
        """
        input_shape = X.shape[1:]
        n_samples = X.shape[0]
        
        # Flatten: (N, ...features...) -> (N, D)
        X_flat = X.reshape(n_samples, -1)
        n_features = X_flat.shape[1]
        
        # Centering
        mean = jnp.mean(X_flat, axis=0)
        Xc = X_flat - mean

        # SVD
        _U, _S, Vt = jnp.linalg.svd(Xc, full_matrices=False)
        full_rank = min(n_samples, n_features)
        
        final_k = full_rank # Default to full rank if nothing specified

        if self.n_components is not None:
            final_k = min(self.n_components, full_rank)
        elif self.error_threshold is not None:
            loop_limit = full_rank
            if self.max_components is not None:
                loop_limit = min(self.max_components, full_rank)
            loop_start = 1
            if self.min_components is not None:
                loop_start = min(self.min_components, full_rank)

            # Start from min_components!
            for k in range(loop_start, loop_limit + 1):
                V_k = Vt[:k, :]
                coeff = Xc @ V_k.T.conj()
                X_rec = coeff @ V_k
                
                current_error = jnp.mean(jnp.abs(Xc - X_rec))
                if current_error < self.error_threshold:
                    break
            final_k = k

        components = Vt[:final_k, :]
        
        # Calculate noise variance
        coeff = Xc @ components.T.conj()
        X_rec_flat = coeff @ components
        residuals = Xc - X_rec_flat
        noise_flat = jnp.mean(jnp.abs(residuals)**2, axis=0)
        noise_variance = noise_flat.reshape(input_shape)
        
        # Update n_components to reflect what was actually chosen
        return replace(self, components=components, mean=mean, input_shape=input_shape, n_components=final_k, noise_variance=noise_variance)

    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], **kwargs) -> Any:
        """
        Forward: Data Space -> Latent Space
        """
        if self.components is None or self.mean is None:
            raise RuntimeError("PCA is not fitted. Call solve() first.")

        # 1. Handle Variance Tuple
        var_in = None
        if isinstance(x, tuple):
            x, var_in = x
        
        # 2. Transform Mean
        # Standard projection: dot product with conjugate transpose
        x_flat = x.ravel()
        centered = x_flat - self.mean
        mean_out = centered @ self.components.T.conj()

        # 3. Transform Variance
        if var_in is not None:
            var_flat = var_in.ravel()
            if jnp.isrealobj(var_flat):
                # Real variance, so we assume Var(Re) == Var(Im)
                V_sq = jnp.abs(self.components)**2
                var_out = V_sq @ var_flat
                return mean_out, var_out
            else:
                # Complex variance, so we treat Re and Im as independent Gaussian Processes.
                W_r_sq = self.components.real ** 2
                W_i_sq = self.components.imag ** 2
                var_x_r = var_flat.real
                var_x_i = var_flat.imag

                var_z_r = W_r_sq @ var_x_r + W_i_sq @ var_x_i
                var_z_i = W_r_sq @ var_x_i + W_i_sq @ var_x_r

                var_out = var_z_r + 1j * var_z_i
                return mean_out, var_out

        return mean_out

    def inverse(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], **kwargs) -> Any:
        """
        Inverse: Latent Space -> Data Space
        Propagates variance: Var(x) = Var(z) @ (Components)^2
        """
        if self.components is None or self.mean is None:
            raise RuntimeError("PCA is not fitted. Call solve() first.")

        # 1. Handle Variance Tuple
        var_in = None
        if isinstance(x, tuple):
            x, var_in = x
        
        # 2. Transform Mean
        x_rec_centered = x @ self.components # (K,) @ (K, D) -> (D,)
        x_rec_flat = x_rec_centered + self.mean
        mean_out = x_rec_flat.reshape(self.input_shape)

        # 3. Transform Variance
        if var_in is not None:
            if jnp.isrealobj(var_in):
                # If variance is real-valued, we interpret it as circular noise
                V_sq = jnp.abs(self.components)**2
                var_flat = var_in @ V_sq
                var_out = var_flat.reshape(self.input_shape)
            else:
                # If variance is complex-valued, we interpret it as two independent variances
                W_r_sq = self.components.real ** 2
                W_i_sq = self.components.imag ** 2
                var_z_r = var_in.real
                var_z_i = var_in.imag

                var_out_r = var_z_r @ W_r_sq + var_z_i @ W_i_sq
                var_out_i = var_z_r @ W_i_sq + var_z_i @ W_r_sq
                var_flat = var_out_r + 1j * var_out_i

                var_out = var_flat.reshape(self.input_shape)
            return mean_out, var_out

        return mean_out        