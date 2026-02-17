import jax
import jax.numpy as jnp
from typing import List, Tuple, Union, Optional
from dataclasses import replace
from eqxlearn.base import BaseModel, Estimator, Transformer

class Pipeline(Estimator, Transformer):
    """
    Sequentially applies a list of transforms and a final estimator.
    
    Architecture:
    - Transparent Delegation: Unknown attributes (like 'loss', 'X', 'y') are 
      delegated to the final step. This allows fit() to inspect the underlying model.
    - Strategy Propagation: The pipeline adopts the training strategy of its 
      final step (e.g. 'analytical' vs 'loss-internal').
    """
    steps: List[Tuple[str, BaseModel]]

    def __init__(self, steps: List[Union[BaseModel, Tuple[str, BaseModel]]]):
        """
        Args:
            steps: List of (name, model) tuples or just model instances.
        """
        formatted_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, tuple) and len(step) == 2:
                formatted_steps.append(step)
            else:
                # Auto-generate names if not provided
                formatted_steps.append((f"step_{i}", step))
        
        # Validation: All but last must be transformers
        for name, step in formatted_steps[:-1]:
            if not hasattr(step, "transform"):
                 raise TypeError(f"Intermediate step '{name}' must implement .transform()")
                 
        self.steps = formatted_steps

    # --- 1. Strategy Override (The "Truth Source") ---
    
    @property
    def strategy(self) -> str:
        """
        Delegates the training strategy to the final estimator.
        If the final estimator is 'loss-internal' (GP), the whole pipeline 
        is treated as 'loss-internal'.
        """
        return self.steps[-1][1].strategy

    # --- 2. Structural Updates (State Management) ---
    
    def condition(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> 'Pipeline':
        """
        Conditions and solves the intermediate steps in the pipeline,
        and conditions the final step.
        """
        X_curr = X
        new_steps = []
        for name, step in self.steps[:-1]:
            if hasattr(step, "condition"): step = step.condition(X_curr, y)
            if hasattr(step, "solve"): step = step.solve(X_curr, y)
            new_steps.append((name, step))
            X_curr = step.transform(X_curr)
            
        last_name, last_step = self.steps[-1]
        if hasattr(last_step, "condition"):
            last_step = last_step.condition(X_curr, y)
        new_steps.append((last_name, last_step))
        return replace(self, steps=new_steps)

    def solve(self, X: jnp.ndarray, y: Optional[jnp.ndarray] = None) -> 'Pipeline':
        """
        Solves the final step in the pipeline.
        """
        last_name, last_step = self.steps[-1]
        if not hasattr(last_step, "solve"):
            return self
        
        # Transform all intermediate steps
        X_curr = X
        new_steps = []
        for name, step in self.steps[:-1]:
            new_steps.append((name, step))
            X_curr = step.transform(X_curr)

        # Solve the final estimator (e.g. LinearRegression)
        last_step = last_step.solve(X_curr, y)
        new_steps.append((last_name, last_step))
        
        return replace(self, steps=new_steps)

    # --- 3. Transparent Delegation (The "Magic") ---
    
    def __getattr__(self, name):
        """
        Delegates unknown attributes to the final step.
        This enables:
          - hasattr(pipeline, 'loss') -> True (if GP has loss)
          - inspect.signature(pipeline.loss) -> Works correctly
          - pipeline.X -> returns the GP's X
        """
        # Safety: Don't delegate special methods (prevents infinite recursion on pickling)
        if name.startswith("__"):
            raise AttributeError(name)
        
        # We delegate to the underlying model instance (the second item in the tuple)
        return getattr(self.steps[-1][1], name)

    # --- 4. Standard Inference Methods ---

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Single sample inference. Passes x through all steps.
        """
        for name, layer in self.steps:
            # Simple pass-through.
            # If you need specific kwargs filtering per layer, add it here.
            x = layer(x, **kwargs)
        return x

    def inverse(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Inverts the pipeline in reverse order.
        """
        for name, layer in reversed(self.steps):
            if hasattr(layer, "inverse"):
                x = layer.inverse(x, **kwargs)
            else:
                 raise NotImplementedError(f"Step '{name}' does not implement .inverse()")
        return x
    
    def predict(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Predicts targets for a batch of samples `X`.
        """
        if not isinstance(self.steps[-1][1], Estimator):
            raise Exception("Cannot call 'predict' on Pipeline since final step is not an estimator")
        return super().predict(X, **kwargs)

    def transform(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Transforms targets for a batch of samples `X`.
        """
        if isinstance(self.steps[-1][1], Transformer):
            return super().transform(X, **kwargs)
        
        def transform_one(x):
            for name, layer in self.steps[0:-1]:
                x = layer(x, **kwargs)
            return x
        return jax.vmap(transform_one)(X)

    # --- Indexing Support ---
    def __getitem__(self, key: Union[str, int, slice]):
        if isinstance(key, slice):
            return Pipeline(self.steps[key])
        elif isinstance(key, int):
            return self.steps[key][1]
        elif isinstance(key, str):
            for name, step in self.steps:
                if name == key:
                    return step
            raise KeyError(f"Key '{key}' not found in Pipeline.")
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    @property
    def named_steps(self):
        return dict(self.steps)