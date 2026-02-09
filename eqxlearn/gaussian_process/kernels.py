import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Union

def _fmt(val: jnp.ndarray) -> str:
    """Helper to format JAX arrays as clean strings for printing."""
    # Convert from log-space back to linear space if needed before calling this
    if hasattr(val, 'item'):
        # If scalar, return a nice float string
        if val.ndim == 0:
            return f"{val.item():.3g}"
        # If vector (e.g. ARD length scales), print the list
        return str(val.tolist())
    return str(val)

class Kernel(eqx.Module):
    """Abstract base class enabling kernel algebra."""
    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)

    @abstractmethod
    def __call__(self, x1, x2):
        pass

    def __repr__(self):
        # Fallback for custom kernels without explicit repr
        return f"{self.__class__.__name__}()"

class SumKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2):
        return self.k1(x1, x2) + self.k2(x1, x2)

    def __repr__(self):
        return f"{self.k1} + {self.k2}"

class ProductKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)

    def __repr__(self):
        # Handle operator precedence: if a child is a Sum, wrap in parens
        s1 = str(self.k1)
        s2 = str(self.k2)
        
        if isinstance(self.k1, SumKernel):
            s1 = f"({s1})"
        if isinstance(self.k2, SumKernel):
            s2 = f"({s2})"
            
        return f"{s1} * {s2}"

class ConstantKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        self.log_variance = jnp.log(variance)

    def __call__(self, x1, x2):
        return jnp.exp(self.log_variance)

    def __repr__(self):
        val = jnp.exp(self.log_variance)
        return f"{val.item():.3g}**2"

class RBFKernel(Kernel):
    log_length_scale: jnp.ndarray

    def __init__(self, length_scale=1.0):
        self.log_length_scale = jnp.log(length_scale)

    def __call__(self, x1, x2):
        length_scale = jnp.exp(self.log_length_scale)
        diff = x1 - x2
        sq_dist = jnp.sum(diff**2)
        return jnp.exp(-0.5 * sq_dist / (length_scale**2))
    
    def __repr__(self):
        ls = jnp.exp(self.log_length_scale)
        return f"RBF(length_scale={_fmt(ls)})"

class WhiteNoiseKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        self.log_variance = jnp.log(variance)

    def __call__(self, x1, x2):
        is_equal = jnp.allclose(x1, x2)
        return jnp.where(is_equal, jnp.exp(self.log_variance), 0.0)

    def __repr__(self):
        val = jnp.exp(self.log_variance)
        return f"WhiteKernel(noise_level={_fmt(val)})"