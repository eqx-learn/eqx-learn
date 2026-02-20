import jax.numpy as jnp
from typing import Union, Tuple, Any
from eqxlearn.base import InvertibleTransformer

class ComplexSplitter(InvertibleTransformer):
    """
    Splits Complex (N,) <-> Real (2N,).
    Inverse packs distinct variances into a complex container.
    """
    def solve(self, X, y) -> Any:
        return self
    
    def __call__(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs) -> Any:
        # Forward: Unpack complex variance into stacked real array
        # Input variance is complex: Re=Var(Re), Im=Var(Im)
        var_in = None
        if isinstance(x, tuple): x, var_in = x

        mean_out = jnp.concatenate([jnp.real(x), jnp.imag(x)])
        
        if var_in is not None:
            # Check if input variance is complex (packed) or scalar (total)
            if jnp.iscomplexobj(var_in):
                # Unpack: Complex(VarRe, VarIm) -> Stack[VarRe, VarIm]
                var_out = jnp.concatenate([jnp.real(var_in), jnp.imag(var_in)])
            else:
                # Fallback for scalar variance (assume isotropic)
                var_part = 0.5 * var_in
                var_out = jnp.concatenate([var_part, var_part])
            return mean_out, var_out
            
        return mean_out

    def inverse(self, x: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], key=None, **kwargs) -> Any:
        # Inverse: Pack stacked real array into complex variance
        var_in = None
        if isinstance(x, tuple): x, var_in = x
            
        mid = x.shape[0] // 2
        mean_out = x[:mid] + 1j * x[mid:]
        
        if var_in is not None:
            var_real, var_imag = var_in[:mid], var_in[mid:]
            # Pack into complex container
            var_out = var_real + 1j * var_imag
            return mean_out, var_out
            
        return mean_out