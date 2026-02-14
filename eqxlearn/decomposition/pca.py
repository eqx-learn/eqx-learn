import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Union, Self
from dataclasses import replace

from eqxlearn.base import Transformer

class PCA(Transformer):
    """
    Principal Component Analysis (PCA).
    
    Configuration is set in __init__.
    Fitting is performed in solve().
    """
    # --- Learned State ---
    components: Optional[jnp.ndarray] = None  # Shape: (n_components, n_features)
    mean: Optional[jnp.ndarray] = None        # Shape: (n_features,)
    input_shape: Optional[Tuple[int, ...]] = eqx.field(static=True, default=None)

    # --- Hyperparameters ---
    min_components: int = eqx.field(static=True)
    max_components: Optional[int] = eqx.field(static=True)
    error_threshold: Optional[float] = eqx.field(static=True)

    def __init__(
        self, 
        min_components: int = 1, 
        max_components: Optional[int] = None, 
        error_threshold: Optional[float] = None,
        components=None,
        mean=None,
        input_shape=None,
    ):
        """
        Args:
            min_components: Min components to keep.
            max_components: Max components to keep.
            error_threshold: If set, adds components until reconstruction error < threshold.
        """
        self.min_components = min_components
        self.max_components = max_components
        self.error_threshold = error_threshold
        
        # Initialize state to None
        self.components = components
        self.mean = mean
        self.input_shape = input_shape

    def solve(self, X: jnp.ndarray, y: jnp.ndarray = None) -> Self:
        """
        Solves the PCA decomposition using SVD.
        
        Args:
            X: Input data (N, ...)
            y: Ignored (PCA is unsupervised), kept for API consistency.
        Returns:
            New fitted PCA instance.
        """
        # 1. Shape Handling
        input_shape = X.shape[1:]
        n_samples = X.shape[0]
        
        # Flatten: (N, ...features...) -> (N, D)
        X_flat = X.reshape(n_samples, -1)
        n_features = X_flat.shape[1]
        
        # 2. Centering
        mean = jnp.mean(X_flat, axis=0)
        Xc = X_flat - mean

        # 3. SVD
        # full_matrices=False -> Vt is (K, n_features) where K = min(n_samples, n_features)
        # We perform the full SVD once, then select components from it.
        _U, _S, Vt = jnp.linalg.svd(Xc, full_matrices=False)
        
        full_max = min(n_samples, n_features)
        
        # 4. Determine n_components
        max_c = min(self.max_components, full_max) if self.max_components else full_max
        min_c = min(self.min_components, full_max)
        
        final_k = min_c

        if self.error_threshold is not None:
            # Iterative selection based on reconstruction error
            # Since this is a one-time solve, a Python loop is acceptable
            for k in range(min_c, max_c + 1):
                V_k = Vt[:k, :]
                
                # Project (N, D) @ (D, k) -> (N, k)
                scores = jnp.dot(Xc, V_k.T)
                
                # Reconstruct (N, k) @ (k, D) -> (N, D)
                X_rec = jnp.dot(scores, V_k) + mean
                
                # Max Absolute Error check
                current_error = jnp.max(jnp.abs(X_flat - X_rec))
                
                if current_error < self.error_threshold:
                    final_k = k
                    break
            else:
                final_k = max_c
        else:
            # Standard logic
            if self.max_components is None and self.min_components == 1:
                 final_k = full_max
            elif self.max_components is not None:
                 final_k = max_c
            else:
                 final_k = min_c

        # 5. Extract Final Components
        components = Vt[:final_k, :]

        # 6. Return New Fitted Instance (Immutable Update)
        return replace(self, components=components, mean=mean, input_shape=input_shape)

    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Project data X onto the principal components.
        """
        X_flat = X.reshape(X.shape[0], -1)
        Xc = X_flat - self.mean
        return jnp.dot(Xc, self.components.T)

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Reconstruct data from principal components.
        """
        # Reconstruct flattened data
        X_rec_flat = jnp.dot(X, self.components) + self.mean
        
        # Reshape back to original dimensions
        original_shape = (X.shape[0],) + self.input_shape
        return X_rec_flat.reshape(original_shape)
        
    def __call__(self, x: jnp.ndarray):
        """Alias for transform."""
        return self.transform(x)