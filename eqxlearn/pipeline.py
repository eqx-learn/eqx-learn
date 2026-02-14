import inspect
import jax
import jax.numpy as jnp
from typing import List, Tuple, Union, Any, Optional, Self, TYPE_CHECKING
from dataclasses import replace
from eqxlearn.base import BaseModel, Transformer, Estimator, InvertibleTransformer

# --- Helpers ---

def _filter_kwargs(func: Any, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(func)
    except ValueError:
        return kwargs
    params = sig.parameters.values()
    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD: return kwargs
    valid_keys = set(p.name for p in params)
    return {k: v for k, v in kwargs.items() if k in valid_keys}

# --- Mixins ---

class _SolveMixin:
    def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self:
        X_curr = X
        new_steps = []
        
        # Iterate over all steps EXCEPT the last one
        for name, step in self.steps[:-1]:
            if hasattr(step, "solve"): 
                step = step.solve(X_curr, y)
            new_steps.append((name, step))
            # Transform data for the next step
            X_curr = step.transform(X_curr)
            
        # Handle the last step separately
        last_name, last_step = self.steps[-1]
        
        # FIX: Only call solve on the last step if it actually supports it.
        # This allows [PCA, FunctionTransformer] to work.
        if hasattr(last_step, "solve"):
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
        if hasattr(last_step, "condition"):
            last_step = last_step.condition(X_curr, y)
        elif hasattr(last_step, "X") and hasattr(last_step, "y"):
            last_step = replace(last_step, X=X_curr, y=y)
        new_steps.append((last_name, last_step))
        return replace(self, steps=new_steps)

class _LossMixin:
    def loss(self, **kwargs):
        step = self.steps[-1][1]
        filtered_kwargs = _filter_kwargs(step.loss, kwargs)
        return step.loss(**filtered_kwargs)

# --- Class ---

class Pipeline(BaseModel):
    """
    Sequentially applies a list of transforms and a final step.
    """
    steps: List[Tuple[str, BaseModel]]

    def __new__(cls, steps: List[Union[BaseModel, Tuple[str, BaseModel]]]):
        if cls.__name__ != "Pipeline":
            return super().__new__(cls)

        # Unpack steps to check capabilities
        # Handle both (name, model) tuples and raw model objects
        models = [s if not isinstance(s, tuple) else s[1] for s in steps]
        last = models[-1]
        
        # 1. Determine Identity (Estimator vs Transformer)
        # Based ONLY on the last step
        if isinstance(last, Transformer):
            if isinstance(last, InvertibleTransformer) or hasattr(last, "inverse"):
                base_identity = InvertibleTransformer
            else:
                base_identity = Transformer
        else:
            base_identity = Estimator

        # 2. Mixins
        mixins = []
        
        # FIX: Check if ANY step has 'solve', not just the last one.
        if any(hasattr(m, 'solve') for m in models):
            mixins.append(_SolveMixin)
        
        # Condition/Loss still mostly depend on the final estimator logic
        # (Though technically one could have a conditioner in the middle, 
        # usually only the final model holds the likelihood state).
        needs_data = hasattr(last, 'condition') or (hasattr(last, 'loss') and not hasattr(last, 'solve'))
        if needs_data: 
            mixins.append(_ConditionMixin)
            
        if hasattr(last, 'loss'): 
            mixins.append(_LossMixin)
        
        # 3. Create Dynamic Class
        # MRO: Pipeline methods -> Mixins -> Pipeline (base) -> Invertible/Estimator
        bases = tuple(mixins + [cls, base_identity])
        
        identity_name = base_identity.__name__
        mixin_names = "".join([m.__name__.replace('_', '').replace('Mixin', '') for m in mixins])
        cls_name = f"Pipeline{identity_name}{mixin_names}"
        
        DynamicPipeline = type(cls_name, bases, {})
        return super().__new__(DynamicPipeline)

    def __init__(self, steps: List[Union[BaseModel, Tuple[str, BaseModel]]]):
        if not steps: raise ValueError("Pipeline must have at least one step.")
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
        
    # --- Indexing Logic Added Here ---
    def __getitem__(self, key: Union[str, int, slice]):
        """
        Access steps by name, index, or slice.
        - pipe['name']: returns the step object.
        - pipe[0]: returns the step object at index 0.
        - pipe[0:2]: returns a NEW Pipeline instance with steps 0 and 1.
        """
        if isinstance(key, slice):
            # Return a new Pipeline instance (re-evaluating mixins)
            return Pipeline(self.steps[key])
        elif isinstance(key, int):
            # Return the model object directly
            return self.steps[key][1]
        elif isinstance(key, str):
            # Return the model object matching the name
            for name, step in self.steps:
                if name == key:
                    return step
            raise KeyError(f"Key '{key}' not found in Pipeline.")
        else:
            raise TypeError(f"Invalid index type: {type(key)}")        

    @property
    def named_steps(self): return dict(self.steps)
    
    @property
    def iterative_fitting(self):
        final_step = self.steps[-1][1]
        return getattr(final_step, 'iterative_fitting', hasattr(final_step, 'loss'))

    @property
    def X(self): return getattr(self.steps[-1][1], 'X', None)
    @property
    def y(self): return getattr(self.steps[-1][1], 'y', None)    

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for name, layer in self.steps:
            step_kwargs = _filter_kwargs(layer, kwargs)
            x = layer(x, **step_kwargs)
        return x

    def inverse(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for name, layer in reversed(self.steps):
            if hasattr(layer, "inverse"):
                step_kwargs = _filter_kwargs(layer.inverse, kwargs)
                x = layer.inverse(x, **step_kwargs)
            elif hasattr(layer, "inverse_transform"):
                 raise NotImplementedError(f"Step {name} does not implement single-sample .inverse()")
        return x

    if TYPE_CHECKING:
        def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> Self: ...
        def loss(self, **kwargs) -> jnp.ndarray: ...
        def predict(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray: ...
        def transform(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray: ...
        def inverse_transform(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray: ...