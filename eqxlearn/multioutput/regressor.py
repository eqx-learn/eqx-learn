import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union, Any, Tuple, Self, TYPE_CHECKING
from dataclasses import replace
import inspect

from eqxlearn.base import Regressor

# --- Mixins ---

class _MOSolveMixin:
    def solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> Self:
        model = self._ensure_init(Y)
        M = model._output_dim
        
        # Broadcast X: (M, N, D) | Transpose Y: (M, N)
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        
        # Vmap the solve call across the stack
        new_batched = jax.vmap(lambda m, x, y: m.solve(x, y))(
            model.batched, X_stack, Y_stack
        )
        return replace(model, batched=new_batched)

class _MOConditionMixin:
    def condition(self, X: jnp.ndarray, Y: jnp.ndarray) -> Self:
        model = self._ensure_init(Y)
        M = model._output_dim
        
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        
        # Define a robust single-model conditioner
        def single_condition(m, x, y):
            if hasattr(m, 'condition'):
                return m.condition(x, y)
            elif hasattr(m, 'X'):
                # Fallback: Data Injection
                return replace(m, X=x, y=y)
            return m

        new_batched = jax.vmap(single_condition)(model.batched, X_stack, Y_stack)
        return replace(model, batched=new_batched)

class _MOLossMixin:
    def loss(self, key=None):
        if self.batched is None: raise RuntimeError("Uninitialized.")
        
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

# --- Class ---

class MultiOutputRegressor(Regressor):
    """
    Wraps a base estimator and fits one copy per target dimension.
    """
    batched: Optional[eqx.Module] = None
    estimator: eqx.Module
    _output_dim: Optional[int] = eqx.field(static=True, default=None)

    def __new__(cls, estimator: eqx.Module, *args, **kwargs):
        if cls.__name__ != "MultiOutputRegressor":
            return super().__new__(cls)

        mixins = []
        # Check capabilities of the base estimator
        if hasattr(estimator, 'solve'):
            mixins.append(_MOSolveMixin)
            
        # Add Condition Mixin if it has .condition OR if it needs data (has .loss + .X)
        needs_data = hasattr(estimator, 'condition') or (hasattr(estimator, 'loss') and hasattr(estimator, 'X'))
        if needs_data:
            mixins.append(_MOConditionMixin)
            
        if hasattr(estimator, 'loss'):
            mixins.append(_MOLossMixin)
        
        bases = tuple(mixins + [cls])
        cls_name = "MultiOutputRegressor" + "".join([m.__name__.replace('_', '').replace('MO', '').replace('Mixin', '') for m in mixins])
        DynamicMultiOutput = type(cls_name, bases, {})
        return super().__new__(DynamicMultiOutput)

    def __init__(
        self, 
        estimator: eqx.Module,
        # Optional eager init
        output_dim: Optional[int] = None,
        # Internal fields for reconstruction
        batched: Optional[eqx.Module] = None,
        _output_dim: Optional[int] = None
    ):
        self.estimator = estimator
        
        # 1. Reconstruction Path
        if batched is not None:
            self.batched = batched
            self._output_dim = _output_dim
            return

        # 2. User Init Path
        if output_dim is not None:
            self._output_dim = output_dim
            self.batched = self._create_batch(output_dim)
        else:
            self._output_dim = None
            self.batched = None

    def _create_batch(self, M: int):
        """
        Creates a stack of M identical estimators.
        """
        # We simply replicate the base estimator M times.
        # jax.tree.map will stack the leaves.
        list_of_models = [self.estimator for _ in range(M)]
        
        # Stack leaves to create the Vmapped module structure
        return eqx.filter_vmap(lambda x: x)(
            jax.tree.map(lambda *args: jnp.stack(args), *list_of_models)
        )

    def _ensure_init(self, Y: jnp.ndarray) -> "MultiOutputRegressor":
        """Lazily initializes the batch if needed based on Y."""
        if self.batched is not None:
            if Y.shape[1] != self._output_dim: 
                raise ValueError(f"Y dimension {Y.shape[1]} does not match initialized dim {self._output_dim}")
            return self
        
        M = Y.shape[1]
        new_batched = self._create_batch(M)
        return replace(self, batched=new_batched, _output_dim=M)

    # --- Properties ---

    @property
    def X(self): return getattr(self.batched, 'X', None) if self.batched else None
    
    @property
    def y(self): return getattr(self.batched, 'y', None) if self.batched else None

    @property
    def iterative_fitting(self):
        return getattr(self.estimator, 'iterative_fitting', hasattr(self.estimator, 'loss'))

    # --- Forward Pass ---

    def __call__(self, x: jnp.ndarray, **kwargs):
        """
        Single-sample forward pass. Returns (M,) vector.
        """
        if self.batched is None: raise RuntimeError("Uninitialized")
        return jax.vmap(lambda m: m(x, **kwargs))(self.batched)
    
    if TYPE_CHECKING:
        def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def loss(self, key=None) -> jnp.ndarray: ...