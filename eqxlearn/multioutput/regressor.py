import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Any

class MultiOutputRegressor(eqx.Module):
    """
    A Meta-Estimator that wraps an arbitrary regressor to handle multi-output targets.
    
    It works by:
    1. Creating M independent instances of the base regressor.
    2. Stacking them into a single "Batched" PyTree.
    3. Using vmap to train/predict all M outputs in parallel.
    """
    # This holds the stack of M models. 
    # Internally, it looks like a single model where every parameter has an extra dim (M, ...)
    batched_model: eqx.Module 

    def __init__(
        self, 
        X: jnp.ndarray, 
        Y: jnp.ndarray, 
        make_regressor_fn: Callable[[jnp.ndarray, jnp.ndarray], eqx.Module]
    ):
        """
        Args:
            X: (N, D) Input data (shared across all outputs).
            Y: (N, M) Target data (one column per output task).
            make_regressor_fn: A function that accepts (x, y) 1D slices 
                               and returns an initialized regressor instance.
        """
        N, M = Y.shape
        
        # 1. Instantiate M independent models using Python loop (only runs once at init)
        #    We pass the shared X and the specific column Y[:, i] to each.
        list_of_models = [make_regressor_fn(X, Y[:, i]) for i in range(M)]
        
        # 2. Stack them into a single Batched PyTree
        #    Example: If model has param 'variance', new shape is (M, 1)
        self.batched_model = jax.tree.map(lambda *args: jnp.stack(args), *list_of_models)

    def loss(self):
        """
        Computes the sum of losses across all M estimators.
        """
        # We use vmap to run .loss() on the batched model.
        # vmap automatically "peels" the leading dimension (M) off every parameter in batched_model.
        
        # Define the function to call on a SINGLE instance
        def single_loss(model):
            return model.loss()
            
        # Vectorize it
        all_losses = jax.vmap(single_loss)(self.batched_model)
        
        return jnp.sum(all_losses)

    def predict(self, x_star: jnp.ndarray, **kwargs):
        """
        Returns stacked predictions for all outputs.
        """
        def single_predict(model):
            return model.predict(x_star, **kwargs)

        # Returns (M, ...) e.g. means of shape (M,)
        return jax.vmap(single_predict)(self.batched_model)