import jax
import jax.numpy as jnp
from typing import Tuple

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

class KFold:
    def __init__(self, n_splits=5, *, shuffle=True, key=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.key = key

        if shuffle and key is None:
            raise ValueError("Must provide PRNG key when shuffle=True")

    def split(self, X: jnp.ndarray):
        """
        Returns generator of (train_indices, test_indices)
        """
        n_splits = min(len(X), self.n_splits)
        
        n_samples = X.shape[0]
        indices = jnp.arange(n_samples)

        if self.shuffle:
            indices = jax.random.permutation(self.key, indices)

        # compute fold sizes
        fold_sizes = jnp.full(
            (n_splits,),
            n_samples // n_splits,
            dtype=jnp.int32,
        )

        remainder = n_samples % n_splits
        fold_sizes = fold_sizes.at[:remainder].add(1) # add 1 sample to the first `remainder` folds

        # compute fold boundaries
        boundaries = jnp.concatenate([jnp.array([0]), jnp.cumsum(fold_sizes)])

        for i in range(n_splits):
            start = boundaries[i]
            stop = boundaries[i + 1]

            test_idx = indices[start:stop]

            train_idx = jnp.concatenate([
                indices[:start],
                indices[stop:]
            ])

            yield train_idx, test_idx