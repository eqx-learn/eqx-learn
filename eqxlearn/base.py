from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import equinox as eqx
from eqxlearn.metrics import r2_score

import paramax

class BaseModel(eqx.Module, ABC):
    pass

class Regressor(BaseModel, ABC):
    @eqx.filter_jit
    @abstractmethod
    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        raise NotImplementedError

    @eqx.filter_jit
    def predict(self, X: jnp.ndarray, **kwargs):
        return jax.vmap(lambda x: self(x, **kwargs))(X)
    
    @eqx.filter_jit
    def score(self, X: jnp.ndarray, y: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Returns the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X, **kwargs)
        
        # Handle tuple output (e.g. if GP returns (mean, var))
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
            
        return r2_score(y, y_pred)    
    
class Transformer(BaseModel):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.transform(x)
    
    def transform(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
        