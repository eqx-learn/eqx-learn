import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple, Union, Any, Optional, Self, TYPE_CHECKING
from dataclasses import replace
from eqxlearn.base import Transformer, Estimator

# --- Mixins ---

class _SolveMixin:
    def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self:
        X_curr = X
        new_steps = []
        for name, step in self.steps[:-1]:
            if hasattr(step, "solve"): step = step.solve(X_curr, y)
            new_steps.append((name, step))
            X_curr = step.transform(X_curr)
            
        last_name, last_step = self.steps[-1]
        last_step = last_step.solve(X_curr, y)
        new_steps.append((last_name, last_step))
        return replace(self, steps=new_steps)

class _ConditionMixin:
    def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self:
        X_curr = X
        new_steps = []
        for name, step in self.steps[:-1]:
            if hasattr(step, "solve"): step = step.solve(X_curr, y)
            elif hasattr(step, "condition"): step = step.condition(X_curr, y)
            new_steps.append((name, step))
            X_curr = step.transform(X_curr)
            
        last_name, last_step = self.steps[-1]
        
        # Robust Conditioning
        if hasattr(last_step, "condition"):
            last_step = last_step.condition(X_curr, y)
        elif hasattr(last_step, "X") and hasattr(last_step, "y"):
            last_step = replace(last_step, X=X_curr, y=y)
            
        new_steps.append((last_name, last_step))
        return replace(self, steps=new_steps)

class _LossMixin:
    def loss(self, **kwargs):
        return self.steps[-1][1].loss(**kwargs)

# --- Class ---

class Pipeline(Estimator):
    """
    Sequentially applies a list of transforms and a final estimator.
    """
    steps: List[Tuple[str, eqx.Module]]

    def __new__(cls, steps: List[Union[eqx.Module, Tuple[str, eqx.Module]]]):
        # Guard: If already a specialized subclass, skip factory
        if cls.__name__ != "Pipeline":
            return super().__new__(cls)

        last = steps[-1] if not isinstance(steps[-1], tuple) else steps[-1][1]
        mixins = []
        
        if hasattr(last, 'solve'):
            mixins.append(_SolveMixin)
        
        needs_data = hasattr(last, 'condition') or (hasattr(last, 'loss') and not hasattr(last, 'solve'))
        if needs_data:
            mixins.append(_ConditionMixin)
            
        if hasattr(last, 'loss'):
            mixins.append(_LossMixin)
        
        bases = tuple(mixins + [cls])
        cls_name = "Pipeline" + "".join([m.__name__.replace('_', '').replace('Mixin', '') for m in mixins])
        DynamicPipeline = type(cls_name, bases, {})
        return super().__new__(DynamicPipeline)

    def __init__(self, steps: List[Union[eqx.Module, Tuple[str, eqx.Module]]]):
        if not steps:
            raise ValueError("Pipeline must have at least one step.")

        formatted_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, tuple) and len(step) == 2:
                name, module = step
            else:
                name = f"{step.__class__.__name__.lower()}_{i}"
                module = step
            formatted_steps.append((name, module))
        
        for name, step in formatted_steps[:-1]:
            if not isinstance(step, Transformer):
                raise TypeError(f"Intermediate step '{name}' is not a Transformer.")

        self.steps = formatted_steps

    @property
    def named_steps(self):
        return dict(self.steps)
    
    @property
    def X(self):
        """Expose X of final step for fit() freezing."""
        return getattr(self.steps[-1][1], 'X', None)

    @property
    def y(self):
        """Expose y of final step for fit() freezing."""
        return getattr(self.steps[-1][1], 'y', None)    

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """Single-sample forward pass."""
        for name, layer in self.steps:
            x = layer(x, **kwargs)
        return x
    
    if TYPE_CHECKING:
        def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def loss(self, **kwargs) -> jnp.ndarray: ...