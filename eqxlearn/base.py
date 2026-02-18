from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Union, Self

import jax
import jax.numpy as jnp
import equinox as eqx
from eqxlearn.metrics import r2_score

class BaseModel(eqx.Module, ABC):
    """
    Root class for all eqxlearn objects.
    Enforces JAX-compatible forward pass structure.
    """
    @property
    def strategy(self) -> str:
        """
        Determines the default training strategy for this model.
        Returns:
            - 'analytical': Has .solve(), needs no loop.
            - 'internal-loss': Has .loss(), needs optimization loop.
            - 'external-loss': Has neither, needs loop + external loss fn.
        """
        if hasattr(self, 'solve'):
            return 'analytical'
        elif hasattr(self, 'loss'):
            return 'internal-loss'
        else:
            return 'external-loss'    
    
    @abstractmethod
    def __call__(self, x: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Forward pass for a SINGLE sample `x`.
        Subclasses should implement the logic for one data point.
        """
        raise NotImplementedError

class Estimator(BaseModel, ABC):
    """
    Base class for models that predict a target `y` from features `X`.
    Provides a default batched `predict` method via vmap.
    """
    def predict(self, X: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Predicts targets for a batch of samples `X`.
        Default implementation vmaps the single-sample `__call__`.
        """
        # We allow subclasses to override this for specialized batched inference
        return jax.vmap(lambda x: self(x, key=key, **kwargs))(X)

class Regressor(Estimator):
    """
    Base class for regression models.
    """
    def score(self, X: jnp.ndarray, y: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Returns the coefficient of determination R^2 of the prediction.
        """
        y_pred = self.predict(X, key=key, **kwargs)
        
        # Handle tuple output (e.g. if GP returns (mean, var))
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
            
        return r2_score(y, y_pred)

class Transformer(BaseModel):
    """
    Base class for transformers.
    Contract: Subclass implements `__call__` (single sample forward).
    """
    def transform(self, X: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Batched transformation.
        Automatically vectorizes the single-sample __call__.
        """
        return jax.vmap(lambda x: self(x, key=key, **kwargs))(X)

class InvertibleTransformer(Transformer):
    """
    Extension for transformers that can reverse their mapping.
    """
    @abstractmethod
    def inverse(self, x: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Inverse pass for a SINGLE sample `x`.
        Subclasses must implement this.
        """
        raise NotImplementedError
    
    def inverse_transform(self, X: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        """
        Batched transformation.
        Automatically vectorizes the single-sample inverse().
        """
        return jax.vmap(lambda x: self.inverse(x, key=key, **kwargs))(X)    