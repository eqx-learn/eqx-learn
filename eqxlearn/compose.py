import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Union, Self, Any, Tuple, TYPE_CHECKING
from dataclasses import replace
from eqxlearn.base import Regressor, Transformer

# --- Mixins ---

class _TTSolveMixin:
    def solve(self, X: jnp.ndarray, y: jnp.ndarray) -> Self:
        fitted_trans = self.transformer
        if hasattr(self.transformer, 'solve'): fitted_trans = self.transformer.solve(y)
        y_trans = fitted_trans.transform(y)
        fitted_reg = self.regressor.solve(X, y_trans)
        return replace(self, regressor=fitted_reg, transformer=fitted_trans)

class _TTConditionMixin:
    def condition(self, X: jnp.ndarray, y: jnp.ndarray) -> Self:
        fitted_trans = self.transformer
        if hasattr(self.transformer, 'solve'): fitted_trans = self.transformer.solve(y)
        elif hasattr(self.transformer, 'condition'): fitted_trans = self.transformer.condition(y)
        y_trans = fitted_trans.transform(y)
        
        fitted_reg = self.regressor
        if hasattr(self.regressor, 'condition'):
            fitted_reg = self.regressor.condition(X, y_trans)
        elif hasattr(self.regressor, 'X'):
            fitted_reg = replace(self.regressor, X=X, y=y_trans)
            
        return replace(self, regressor=fitted_reg, transformer=fitted_trans)

class _TTLossMixin:
    def loss(self, **kwargs):
        return self.regressor.loss(**kwargs)

# --- Class ---

class TransformedTargetRegressor(Regressor):
    """
    Meta-estimator to regress on a transformed target.
    """
    regressor: Any
    transformer: Transformer
    
    def __new__(cls, regressor: Any, transformer: Transformer):
        if cls.__name__ != "TransformedTargetRegressor":
            return super().__new__(cls)

        mixins = []
        if hasattr(regressor, 'solve'):
            mixins.append(_TTSolveMixin)
            
        needs_data = hasattr(regressor, 'condition') or (hasattr(regressor, 'loss') and not hasattr(regressor, 'solve'))
        if needs_data:
            mixins.append(_TTConditionMixin)
            
        if hasattr(regressor, 'loss'):
            mixins.append(_TTLossMixin)
        
        bases = tuple(mixins + [cls])
        cls_name = "TransformedTargetRegressor" + "".join([m.__name__.replace('_', '').replace('Mixin', '') for m in mixins])
        DynamicTTR = type(cls_name, bases, {})
        return super().__new__(DynamicTTR)

    def __init__(self, regressor: Any, transformer: Transformer):
        self.regressor = regressor
        self.transformer = transformer

    # Delegate data properties to inner regressor
    @property
    def X(self): return getattr(self.regressor, 'X', None)
    
    @property
    def y(self): return getattr(self.regressor, 'y', None)

    def __call__(self, x: jnp.ndarray, **kwargs) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Single-sample forward pass.
        Calculates prediction in transformed space, then inverses it.
        Handles both Array and Tuple (mean, var) outputs automatically.
        """
        return self.transformer.inverse(self.regressor(x, **kwargs))
    
    if TYPE_CHECKING:
        def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def loss(self, **kwargs) -> jnp.ndarray: ...