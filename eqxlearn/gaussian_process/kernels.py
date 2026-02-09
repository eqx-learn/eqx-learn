import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Union

class Kernel(eqx.Module):
    """Abstract base class enabling kernel algebra."""
    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)

    @abstractmethod
    def __call__(self, x1, x2):
        pass

class SumKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2):
        return self.k1(x1, x2) + self.k2(x1, x2)

class ProductKernel(Kernel):
    k1: Kernel
    k2: Kernel

    def __call__(self, x1, x2):
        return self.k1(x1, x2) * self.k2(x1, x2)

class ConstantKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        self.log_variance = jnp.log(variance)

    def __call__(self, x1, x2):
        return jnp.exp(self.log_variance)

class RBFKernel(Kernel):
    log_length_scale: jnp.ndarray

    def __init__(self, length_scale=1.0):
        self.log_length_scale = jnp.log(length_scale)

    def __call__(self, x1, x2):
        length_scale = jnp.exp(self.log_length_scale)
        diff = x1 - x2
        sq_dist = jnp.sum(diff**2)
        return jnp.exp(-0.5 * sq_dist / (length_scale**2))

class WhiteNoiseKernel(Kernel):
    log_variance: jnp.ndarray

    def __init__(self, variance=1.0):
        self.log_variance = jnp.log(variance)

    def __call__(self, x1, x2):
        # Returns variance if x1 == x2 (approx), else 0
        is_equal = jnp.allclose(x1, x2)
        return jnp.where(is_equal, jnp.exp(self.log_variance), 0.0)