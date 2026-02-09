import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Optional

class Loss(eqx.Module):
    """Base class for losses."""
    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

class MeanSquaredError(Loss):
    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.square(y_pred - y_true))

class MeanAbsoluteError(Loss):
    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jnp.abs(y_pred - y_true))

class HuberLoss(Loss):
    delta: float = 1.0

    def __call__(self, y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        abs_diff = jnp.abs(y_pred - y_true)
        # 0.5 * x^2 if |x| <= delta
        # delta * (|x| - 0.5 * delta) otherwise
        quadratic = jnp.minimum(abs_diff, self.delta)
        linear = abs_diff - quadratic
        return jnp.mean(0.5 * quadratic**2 + self.delta * linear)