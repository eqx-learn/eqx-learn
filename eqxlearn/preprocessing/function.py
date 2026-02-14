import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional, Union, Self, Any, Tuple
from eqxlearn.base import InvertibleTransformer

class FunctionTransformer(InvertibleTransformer):
    """
    Constructs a transformer from an arbitrary JAX-compatible function.
    
    This is useful for stateless transformations like log-scaling, 
    custom feature engineering, or trig functions.
    """
    
    # We mark the functions as static because they are not PyTree leaves 
    # (unless they are wrapped in an eqx.Module themselves).
    func: Callable[[jnp.ndarray], jnp.ndarray] = eqx.field(static=True)
    inverse_func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = eqx.field(static=True)
    kwargs: dict = eqx.field(static=True)

    def __init__(
        self, 
        func: Callable, 
        inverse_func: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            func: The function to apply in the forward pass.
            inverse_func: Optional function for the inverse pass.
            **kwargs: Extra arguments passed to the functions.
        """
        self.func = func
        self.inverse_func = inverse_func
        self.kwargs = kwargs

    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], **kwargs) -> Any:
        """
        Forward transformation (Single Sample).
        """
        # Merge init-time kwargs with call-time kwargs
        all_kwargs = {**self.kwargs, **kwargs}
        
        if isinstance(x, jnp.ndarray):
            return self.func(x, **all_kwargs)
        
        # Handle (mean, var) tuples if the function is linear-approximate
        # Note: For complex non-linear functions, the user should provide 
        # a specialized function that handles the tuple.
        elif isinstance(x, tuple) and len(x) == 2:
            val, var = x
            # Apply function to mean
            new_val = self.func(val, **all_kwargs)
            # Default variance propagation: identity (user should override for non-linear)
            return new_val, var
            
        raise ValueError(f"FunctionTransformer received unknown input type: {type(x)}")

    def inverse(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], **kwargs) -> Any:
        """
        Inverse transformation (Single Sample).
        """
        if self.inverse_func is None:
            raise RuntimeError("inverse_func was not provided to this FunctionTransformer.")
            
        all_kwargs = {**self.kwargs, **kwargs}

        if isinstance(x, jnp.ndarray):
            return self.inverse_func(x, **all_kwargs)
        
        elif isinstance(x, tuple) and len(x) == 2:
            val, var = x
            return self.inverse_func(val, **all_kwargs), var
            
        raise ValueError(f"FunctionTransformer received unknown input type: {type(x)}")