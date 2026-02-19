import jax
import jax.numpy as jnp
from typing import Tuple

from sklearn.model_selection import train_test_split

def train_test_split(
    *arrays: jnp.ndarray,
    test_size: float = 0.25,
    key: jax.random.PRNGKey = None,
    shuffle: bool = True
) -> Tuple[jnp.ndarray, ...]:
    """
    Split arrays or matrices into random train and test subsets.
    
    Args:
        *arrays: Sequence of arrays with the same length / batch dimension.
        test_size: Proportion of the dataset to include in the test split.
        key: JAX PRNGKey for shuffling.
        shuffle: Whether to shuffle data before splitting.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")
        
    n_samples = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError("All arrays must have the same length")

    # Indices
    indices = jnp.arange(n_samples)
    
    if shuffle:
        if key is None:
            # Fallback if user forgets key, though JAX really prefers explicit keys
            key = jax.random.PRNGKey(42) 
        indices = jax.random.permutation(key, indices)
        
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    results = []
    for arr in arrays:
        results.append(arr[train_idx])
        results.append(arr[test_idx])
        
    return tuple(results)