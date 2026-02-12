import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod

def _fmt(val: jnp.ndarray) -> str:
    """Helper to format JAX arrays as clean strings for printing."""
    if hasattr(val, 'item') and val.ndim == 0:
        return f"{val.item():.3g}"
    # If vector (ARD), format nicely
    if hasattr(val, 'tolist'):
        return str([float(f"{x:.3g}") for x in val.flatten()])
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
        return f"{self.__class__.__name__}()"

# --- Combinators (No Changes) ---
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
        s1, s2 = str(self.k1), str(self.k2)
        if isinstance(self.k1, SumKernel): s1 = f"({s1})"
        if isinstance(self.k2, SumKernel): s2 = f"({s2})"
        return f"{s1} * {s2}"

# --- Leaf Kernels (Updated) ---

class ConstantKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        # Ensure we work with arrays, even if scalar passed
        self.log_variance = jnp.log(jnp.array(variance))

    def __call__(self, x1, x2):
        return jnp.exp(self.log_variance)

    def __repr__(self):
        val = jnp.exp(self.log_variance)
        return f"{_fmt(val)}**2"

class RBFKernel(Kernel):
    log_length_scale: jnp.ndarray

    def __init__(self, length_scale=1.0):
        """
        Args:
            length_scale: Scalar (Isotropic) or Vector (ARD) of shape (D,)
        """
        self.log_length_scale = jnp.log(jnp.array(length_scale))

    def __call__(self, x1, x2):
        length_scale = jnp.exp(self.log_length_scale)
        
        # --- ARD UPDATE START ---
        # 1. Scale the difference per dimension
        # If length_scale is scalar, this broadcasts (Isotropic).
        # If length_scale is vector (D,), this divides element-wise (ARD).
        scaled_diff = (x1 - x2) / length_scale
        
        # 2. Square and Sum
        sq_dist = jnp.sum(scaled_diff**2)
        # --- ARD UPDATE END ---
        
        return jnp.exp(-0.5 * sq_dist)
    
    def __repr__(self):
        ls = jnp.exp(self.log_length_scale)
        return f"RBF(length_scale={_fmt(ls)})"

class WhiteNoiseKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        self.log_variance = jnp.log(jnp.array(variance))

    def __call__(self, x1, x2):
        is_equal = jnp.allclose(x1, x2)
        # Use where to maintain differentiability flow
        return jnp.where(is_equal, jnp.exp(self.log_variance), 0.0)

    def __repr__(self):
        val = jnp.exp(self.log_variance)
        return f"WhiteKernel(noise_level={_fmt(val)})"