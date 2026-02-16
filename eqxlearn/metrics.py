import jax.numpy as jnp
import equinox as eqx

@eqx.filter_jit
def mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((jnp.abs(y_true - y_pred)) ** 2)

@eqx.filter_jit
def root_mean_squared_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean((jnp.abs(y_true - y_pred)) ** 2))

@eqx.filter_jit
def mean_absolute_error(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((jnp.abs(y_true - y_pred)))

@eqx.filter_jit
def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Coefficient of determination R^2.
    """
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-8))