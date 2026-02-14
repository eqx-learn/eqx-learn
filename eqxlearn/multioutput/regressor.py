import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Union, Any, Tuple, Self, TYPE_CHECKING
from dataclasses import replace
import inspect

from eqxlearn.base import Regressor

# --- Mixins ---

class _MOSolveMixin:
    def solve(self, X: jnp.ndarray, Y: jnp.ndarray) -> Self:
        model = self._ensure_init(Y)
        M = model._output_dim
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        new_batched = jax.vmap(lambda m, x, y: m.solve(x, y))(model.batched, X_stack, Y_stack)
        return replace(model, batched=new_batched)

class _MOConditionMixin:
    def condition(self, X: jnp.ndarray, Y: jnp.ndarray) -> Self:
        model = self._ensure_init(Y)
        M = model._output_dim
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        new_batched = jax.vmap(lambda m, x, y: m.condition(x, y))(model.batched, X_stack, Y_stack)
        return replace(model, batched=new_batched)

class _MOInjectMixin:
    def condition(self, X: jnp.ndarray, Y: jnp.ndarray) -> Self:
        model = self._ensure_init(Y)
        M = model._output_dim
        X_stack = jnp.broadcast_to(X, (M,) + X.shape)
        Y_stack = Y.T 
        new_batched = replace(model.batched, X=X_stack, y=Y_stack)
        return replace(model, batched=new_batched)

class _MOLossMixin:
    def loss(self, key=None):
        if self.batched is None: raise RuntimeError("Uninitialized.")
        def single_loss(model, k):
            if 'key' in inspect.signature(model.loss).parameters: return model.loss(key=k)
            return model.loss()
        keys = None
        if key is not None: keys = jax.random.split(key, self._output_dim)
        if keys is not None: all_losses = jax.vmap(single_loss)(self.batched, keys)
        else: all_losses = jax.vmap(lambda m: single_loss(m, None))(self.batched)
        return jnp.sum(all_losses)

# --- Class ---

class MultiOutputRegressor(Regressor):
    """
    Wraps a batch of independent estimators to predict multiple targets.
    """
    batched: Optional[eqx.Module] = None
    make_regressor_fn: Callable[[Any, Any], eqx.Module] = eqx.field(static=True)
    _output_dim: Optional[int] = eqx.field(static=True, default=None)

    def __new__(cls, make_regressor_fn: Callable[[Any, Any], eqx.Module], *args, **kwargs):
        if cls.__name__ != "MultiOutputRegressor":
            return super().__new__(cls)

        dummy = make_regressor_fn(None, None)
        mixins = []
        if hasattr(dummy, 'solve'): mixins.append(_MOSolveMixin)
        if hasattr(dummy, 'condition'): mixins.append(_MOConditionMixin)
        elif hasattr(dummy, 'loss') and hasattr(dummy, 'X'): mixins.append(_MOInjectMixin)
        if hasattr(dummy, 'loss'): mixins.append(_MOLossMixin)
        
        bases = tuple(mixins + [cls])
        cls_name = "MultiOutputRegressor" + "".join([m.__name__.replace('_', '').replace('MO', '').replace('Mixin', '') for m in mixins])
        DynamicMultiOutput = type(cls_name, bases, {})
        return super().__new__(DynamicMultiOutput)

    def __init__(self, make_regressor_fn, X=None, Y=None, output_dim=None):
        self.make_regressor_fn = make_regressor_fn
        dim = output_dim
        if Y is not None: dim = Y.shape[1]
        if dim is not None:
            self._output_dim = dim
            self.batched = self._create_batch(dim, X, Y)
        else:
            self._output_dim = None
            self.batched = None

    def _create_batch(self, M, X, Y):
        if X is not None and Y is not None:
            list_of_models = [self.make_regressor_fn(X, Y[:, i]) for i in range(M)]
        else:
            list_of_models = [self.make_regressor_fn(None, None) for i in range(M)]
        return eqx.filter_vmap(lambda x: x)(jax.tree.map(lambda *args: jnp.stack(args), *list_of_models))

    def _ensure_init(self, Y):
        if self.batched is not None:
            if Y.shape[1] != self._output_dim: raise ValueError("Dimension mismatch")
            return self
        M = Y.shape[1]
        new_batched = self._create_batch(M, None, None)
        return replace(self, batched=new_batched, _output_dim=M)

    @property
    def X(self): return getattr(self.batched, 'X', None) if self.batched else None
    
    @property
    def y(self): return getattr(self.batched, 'y', None) if self.batched else None

    def __call__(self, x: jnp.ndarray, **kwargs):
        """
        Single-sample forward pass. Returns (M,) vector.
        """
        return jax.vmap(lambda m: m(x, **kwargs))(self.batched)
    
    if TYPE_CHECKING:
        def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def loss(self, key=None) -> jnp.ndarray: ...