from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx

class BaseModel(eqx.Module, ABC):
    pass

class Regressor(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, X: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    def predict(self, x: jnp.ndarray):
        return jax.vmap(self)(x)