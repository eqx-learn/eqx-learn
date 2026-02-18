import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Any
from dataclasses import replace
import inspect

from eqxlearn.base import Regressor

class MultiOutputRegressor(Regressor):
    """
    Wraps a base estimator and fits one independent copy per target dimension.
    
    Architecture:
    - Lazy Initialization: The batch of models is created the first time `solve` 
      or `condition` is called with target data `Y`.
    - Parallel Execution: All operations (inference, training) are vectorized 
      using `jax.vmap` over the stacked models.
    - Memory Efficiency: The 'estimator' template is discarded (set to None)
      once the batched model is initialized.
    """
    batched: Optional[eqx.Module] = None
    estimator: Optional[eqx.Module] = None
    _output_dim: Optional[int] = eqx.field(static=True, default=None)

    def __init__(
        self, 
        estimator: Optional[eqx.Module] = None,
        output_dim: Optional[int] = None,
        # Internal fields for state reconstruction
        batched: Optional[eqx.Module] = None,
        _output_dim: Optional[int] = None
    ):
        # 1. Reconstruction Path (loading from disk or replace)
        if batched is not None:
            self.batched = batched
            self._output_dim = _output_dim
            # If we have the batch, we don't need the template
            self.estimator = None 
            return

        # 2. Standard Init
        self.estimator = estimator

        # 3. Eager Init Path (if user knows dim upfront)
        if output_dim is not None:
            if estimator is None:
                raise ValueError("Must provide 'estimator' when initializing with 'output_dim'.")
            self._output_dim = output_dim
            self.batched = self._create_batch(output_dim)
            self.estimator = None # Discard template immediately
        else:
            self._output_dim = None
            self.batched = None

    # --- 1. Strategy Delegation ---
    
    @property
    def strategy(self) -> str:
        """
        Delegates strategy to the inner model.
        Checks 'batched' first (post-init), then 'estimator' (pre-init).
        """
        target = self.batched if self.batched is not None else self.estimator
        if target is None:
             # Fallback if accessed on a broken/empty instance
             raise RuntimeError("Cannot access strategy: Model not initialized and no estimator provided.")
        return target.strategy

    # --- 2. Structural Updates (State Management) ---

    def condition(self, X: jnp.ndarray, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """
        Conditions the ensemble on data (e.g. for GPs).
        """
        model = self._ensure_init(Y)
        M = model._output_dim
        
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        
        def single_condition(m, x, y):
            if hasattr(m, 'condition'):
                return m.condition(x, y)
            return m

        new_batched = jax.vmap(single_condition)(model.batched, X_stack, Y_stack)
        return replace(model, batched=new_batched)    

    def solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """
        1. Initializes the batch (if needed) using Y's shape.
        2. Vmaps the solve() call across all M models.
        """
        # Ensure we have a stack of M models ready
        # This returns a model where .estimator is None and .batched is set
        model = self._ensure_init(Y)
        M = model._output_dim
        
        # Prepare Data: X -> (M, N, D), Y -> (M, N)
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        
        # Check for 'solve' on the batched instance (since estimator might be None now)
        if hasattr(model.batched, 'solve'):
            new_batched = jax.vmap(lambda m, x, y: m.solve(x, y))(
                model.batched, X_stack, Y_stack
            )
            return replace(model, batched=new_batched)
        
        # If inner model has no solve (e.g. MLP), just return the initialized model
        return model

    # --- 3. Loss Calculation (For 'loss-internal') ---

    def loss(self, key=None):
        """
        Sums the loss of all independent models.
        Only called if strategy == 'loss-internal'.
        """
        if self.batched is None: 
            raise RuntimeError("MultiOutputRegressor cannot compute loss: Model is uninitialized. Call solve() first.")
        
        # Helper to handle key splitting
        def single_loss(model, key):
            return model.loss(key=key)
        
        keys = None
        if key is not None: 
            keys = jr.split(key, self._output_dim)
            
        if keys is not None: 
            all_losses = jax.vmap(single_loss)(self.batched, keys)
        else: 
            all_losses = jax.vmap(lambda m: single_loss(m, None))(self.batched)
            
        return jnp.sum(all_losses)

    # --- 4. Forward Pass ---

    def __call__(self, x: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Single-sample forward pass. Returns (M,) vector.
        """
        if self.batched is None: 
            raise RuntimeError("MultiOutputRegressor is uninitialized. Call solve() or fit() first.")
        
        def single_call(model, x, key):
            return model(x, key=key, **kwargs)
        
        keys = None
        if key is not None:
            keys = jr.split(key, self._output_dim)
        
        return jax.vmap(single_call)(self.batched, keys)

    # --- 5. Helpers ---

    def _create_batch(self, M: int):
        """
        Creates a stack of M identical estimators.
        Leaf stacking allows jax.vmap to treat dimension 0 as the batch dim.
        """
        if self.estimator is None:
            raise RuntimeError("Cannot create batch: 'estimator' is None.")
            
        list_of_models = [self.estimator for _ in range(M)]
        return jax.tree.map(lambda *args: jnp.stack(args), *list_of_models)

    def _ensure_init(self, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """
        Lazily initializes the batch if needed based on Y.
        Sets estimator to None in the returned instance.
        """
        M = Y.shape[1]
        
        # Case A: Already initialized
        if self.batched is not None:
            if self._output_dim != M: 
                raise ValueError(f"Y dimension {M} does not match initialized dim {self._output_dim}")
            return self
        
        # Case B: First time init
        new_batched = self._create_batch(M)
        
        # Return new state: batched set, estimator cleared
        return replace(self, batched=new_batched, _output_dim=M, estimator=None)

    def __getattr__(self, name):
        """
        Delegates attributes to the inner model.
        Prioritizes 'batched', falls back to 'estimator' (if pre-init).
        """
        if name.startswith("__"):
            raise AttributeError(name)
            
        # 1. Try batched (initialized state)
        if self.batched is not None:
            return getattr(self.batched, name)
            
        # 2. Try estimator (pre-init state)
        if self.estimator is not None:
            return getattr(self.estimator, name)
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' (and no internal model is set)")