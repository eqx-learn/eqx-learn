import jax
import jax.numpy as jnp

from eqxlearn.base import Regressor

class LinearRegressor(Regressor):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_features: int, key: jax.random.PRNGKey):
        self.weight = jax.random.normal(key, (in_features,))
        self.bias = jnp.zeros(())

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(self.weight, x) + self.bias
    
    def predict(self, x: jnp.ndarray):
        return jax.vmap(self)(x)