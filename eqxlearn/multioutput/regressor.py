import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional
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
    """
    batched: Optional[eqx.Module] = None
    estimator: eqx.Module
    _output_dim: Optional[int] = eqx.field(static=True, default=None)

    def __init__(
        self, 
        estimator: eqx.Module,
        output_dim: Optional[int] = None,
        # Internal fields for state reconstruction
        batched: Optional[eqx.Module] = None,
        _output_dim: Optional[int] = None
    ):
        self.estimator = estimator
        
        # 1. Reconstruction Path (loading from disk)
        if batched is not None:
            self.batched = batched
            self._output_dim = _output_dim
            return

        # 2. Eager Init Path (if user knows dim upfront)
        if output_dim is not None:
            self._output_dim = output_dim
            self.batched = self._create_batch(output_dim)
        else:
            self._output_dim = None
            self.batched = None

    # --- 1. Strategy Delegation ---
    
    @property
    def strategy(self) -> str:
        """
        Delegates strategy to the inner estimator.
        - If inner is 'analytical' (LinReg), we solve analytically M times.
        - If inner is 'loss-internal' (GP), we sum the M internal losses.
        """
        return self.estimator.strategy

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

        new_batched = jax.vmap(single_condition)(model.batched, X_stack, Y_stack)
        return replace(model, batched=new_batched)    

    def solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """
        1. Initializes the batch (if needed) using Y's shape.
        2. Vmaps the solve() call across all M models.
        """
        # Ensure we have a stack of M models ready
        model = self._ensure_init(Y)
        M = model._output_dim
        
        # Prepare Data: X -> (M, N, D), Y -> (M, N)
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        
        # Only call solve if the inner estimator supports it
        if hasattr(self.estimator, 'solve'):
            new_batched = jax.vmap(lambda m, x, y: m.solve(x, y))(
                model.batched, X_stack, Y_stack
            )
            return replace(model, batched=new_batched)
        
        # If inner model has no solve (e.g. MLP), we just return the initialized model
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
        def single_loss(model, k):
            if 'key' in inspect.signature(model.loss).parameters: 
                return model.loss(key=k)
            return model.loss()
        
        keys = None
        if key is not None: 
            keys = jax.random.split(key, self._output_dim)
            
        if keys is not None: 
            all_losses = jax.vmap(single_loss)(self.batched, keys)
        else: 
            all_losses = jax.vmap(lambda m: single_loss(m, None))(self.batched)
            
        return jnp.sum(all_losses)

    # --- 4. Forward Pass ---

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Single-sample forward pass. Returns (M,) vector.
        """
        if self.batched is None: 
            raise RuntimeError("MultiOutputRegressor is uninitialized. Call solve() or fit() first.")
        
        return jax.vmap(lambda m: m(x, **kwargs))(self.batched)

    # --- 5. Helpers ---

    def _create_batch(self, M: int):
        """
        Creates a stack of M identical estimators.
        Leaf stacking allows jax.vmap to treat dimension 0 as the batch dim.
        """
        list_of_models = [self.estimator for _ in range(M)]
        return jax.tree.map(lambda *args: jnp.stack(args), *list_of_models)

    def _ensure_init(self, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """Lazily initializes the batch if needed based on Y."""
        M = Y.shape[1]
        
        if self.batched is not None:
            if self._output_dim != M: 
                raise ValueError(f"Y dimension {M} does not match initialized dim {self._output_dim}")
            return self
        
        new_batched = self._create_batch(M)
        return replace(self, batched=new_batched, _output_dim=M)

    def __getattr__(self, name):
        """
        Delegates attributes to the batched object (e.g. accessing .X returns stacked X).
        """
        if name.startswith("__"):
            raise AttributeError(name)
        if self.batched is None:
            # If not initialized, we can try peeking at the base estimator
            # This helps inspect.signature checks before fitting
            return getattr(self.estimator, name)
        return getattr(self.batched, name)