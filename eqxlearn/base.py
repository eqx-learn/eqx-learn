from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx

import paramax

class BaseModel(eqx.Module, ABC):
    pass

class Regressor(eqx.Module, ABC):
    @eqx.filter_jit
    @abstractmethod
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @eqx.filter_jit
    def predict(self, X: jnp.ndarray):
        # return jax.vmap(paramax.unwrap(self))(X)
        return jax.vmap(self)(X)
    
class Transformer(eqx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
        