import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, Union
from math import prod
from eqxlearn.base import Transformer

class PCA(Transformer):
    """
    A general Principal Component Analysis class compatible.
    
    It flattens all non-batch dimensions into a single feature vector, performs SVD,
    and optionally selects the number of components based on reconstruction error.
    """
    # Learned State
    components: jnp.ndarray  # Shape: (n_components, n_features)
    mean: jnp.ndarray        # Shape: (n_features,)
    
    # Configuration / Metadata
    input_shape: Tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self, 
        X: jnp.ndarray, 
        min_components: int = 1, 
        max_components: Optional[int] = None, 
        error_threshold: Optional[float] = None
    ):
        """
        Fits the PCA model to the data X.

        Args:
            X: Input data. Shape (n_samples, ...features...).
               All dimensions after the first are flattened into features.
            min_components: Minimum number of components to retain.
            max_components: Maximum number of components. Defaults to min(n_samples, n_features).
            error_threshold: If provided, adds components until max(abs(error)) < threshold.
        """
        # 1. Shape Handling
        # We store the trailing shape to allow reshape during inverse_transform
        self.input_shape = X.shape[1:] 
        n_samples = X.shape[0]
        
        # Flatten input: (n_samples, n_features)
        X_flat = X.reshape(n_samples, -1)
        n_features = X_flat.shape[1]
        
        # 2. Centering
        self.mean = jnp.mean(X_flat, axis=0)
        Xc = X_flat - self.mean

        # 3. SVD
        # U, S, Vt = SVD(X_centered)
        # We only need Vt (components) and singular values for variance explanation if needed.
        # full_matrices=False -> Vt is (K, n_features) where K = min(n_samples, n_features)
        _U, _S, Vt = jnp.linalg.svd(Xc, full_matrices=False)
        
        full_max = min(n_samples, n_features)
        
        # 4. Determine n_components
        if max_components is None:
            max_c = full_max
        else:
            max_c = min(max_components, full_max)
            
        min_c = min(min_components, full_max)
        
        final_k = min_c

        if error_threshold is not None:
            # Iterative fitting: select k such that reconstruction error is acceptable.
            # Since this runs once during init (not JIT), a standard Python loop is optimal.
            for k in range(min_c, max_c + 1):
                # Slice top k components
                V_k = Vt[:k, :]  # (k, n_features)
                
                # Project and Reconstruct
                # Transform: Xc @ V_k.T -> (N, k)
                scores = jnp.dot(Xc, V_k.T)
                # Inverse: scores @ V_k -> (N, n_features)
                X_rec = jnp.dot(scores, V_k) + self.mean
                
                # Check error
                # Note: We check the max absolute error across the entire batch
                current_error = jnp.max(jnp.abs(X_flat - X_rec))
                
                if current_error < error_threshold:
                    final_k = k
                    break
            else:
                # If loop finishes without breaking, we hit max_components
                final_k = max_c
        else:
            # Standard Fixed Size
            final_k = min_c if min_components == max_components else max_c

        self.n_components = final_k
        self.components = Vt[:final_k, :]

    @eqx.filter_jit
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Project data X onto the principal components.

        Args:
            X: Data of shape (n_samples, ...features...). 
               Must match the feature dimensions of training data.
        Returns:
            X_new: Transformed data of shape (n_samples, n_components).
        """
        # 1. Flatten
        X_flat = X.reshape(X.shape[0], -1)
        
        # 2. Center
        Xc = X_flat - self.mean
        
        # 3. Project
        # Xc (N, D) @ components.T (D, K) -> (N, K)
        return jnp.dot(Xc, self.components.T)

    @eqx.filter_jit
    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        """
        Transform data back to its original space.

        Args:
            X: Transformed data of shape (n_samples, n_components).
        Returns:
            X_original: Reconstructed data of shape (n_samples, ...original_features...).
        """
        # 1. Reconstruct flattened data
        # X (N, K) @ components (K, D) -> (N, D)
        X_rec_flat = jnp.dot(X, self.components) + self.mean
        
        # 2. Reshape back to original dimensions
        # (N, D) -> (N, d1, d2, d3...)
        original_shape = (X.shape[0],) + self.input_shape
        return X_rec_flat.reshape(original_shape)

    @property
    def explained_variance(self):
        """Returns the variance explained by each component (needs S stored if required)."""
        # If you need this, we would store 'S' (singular values) in __init__
        raise NotImplementedError("Singular values not stored to save memory.")