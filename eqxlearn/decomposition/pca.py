import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Self
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
    max_components: Optional[int] = eqx.field(static=True, default=None)

    def __init__(
        self, 
        n_components: Optional[int] = None, 
        error_threshold: Optional[float] = None,
        max_components: Optional[int] = None,
        # State args for reconstruction/deserialization
        components: Optional[jnp.ndarray] = None,
        mean: Optional[jnp.ndarray] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
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
        self.max_components = max_components
        
        # Initialize state
        self.components = components
        self.mean = mean
        self.input_shape = input_shape
        
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

            # Start from min_components!
            # Default if loop doesn't break
            final_k = loop_limit 
            
            for k in range(1, loop_limit + 1):
                V_k = Vt[:k, :]
                coeff = Xc @ V_k.conj().T
                X_rec = coeff @ V_k
                
                current_error = jnp.max(jnp.abs(Xc - X_rec))
                if current_error < self.error_threshold:
                    final_k = k
                    break

        components = Vt[:final_k, :]
        
        # Update n_components to reflect what was actually chosen
        return replace(self, components=components, mean=mean, input_shape=input_shape, n_components=final_k)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass: Project data onto principal components.
        Args:
            x: Input sample of shape `input_shape`.
        Returns:
            Projected scores of shape (n_components,).
        """
        if self.components is None: raise RuntimeError("PCA not fitted")
        
        x_flat = x.ravel()
        centered = x_flat - self.mean
        return centered @ self.components.T.conj()
    
    def inverse(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Inverse pass: Reconstruct data from principal components.
        Args:
            x: Scores of shape (n_components,).
        Returns:
            Reconstructed data of shape `input_shape`.
        """
        if self.components is None: raise RuntimeError("PCA not fitted")

        x_rec_flat = x @ self.components + self.mean
        return x_rec_flat.reshape(self.input_shape)