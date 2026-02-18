import jax
import jax.random as jr
import jax.numpy as jnp
from typing import Union, Tuple, Any
from dataclasses import replace
from eqxlearn.base import Regressor, Transformer


class TransformedTargetRegressor(Regressor):
    """
    Meta-estimator to regress on a transformed target.
    
    Architecture:
    - Wraps a regressor and a target transformer.
    - During training (solve/condition), 'y' is transformed before being passed to the regressor.
    - During inference (__call__), the prediction is inverse-transformed back to the original space.
    """
    regressor: Any
    transformer: Transformer

    def __init__(self, regressor: Any, transformer: Transformer):
        self.regressor = regressor
        self.transformer = transformer

    @property
    def strategy(self) -> str:
        """
        Delegates the training strategy to the inner regressor.
        If the regressor is a GP ('loss-internal'), the wrapper acts like one too.
        """
        return self.regressor.strategy

    def condition(self, X: jnp.ndarray, y: jnp.ndarray) -> "TransformedTargetRegressor":
        """
        Conditions the transformer on data. 
        
        This conditions and solved the transformer, and conditions the regressor.
        """
        # 1. Fit Transformer (Must fit to transform y correctly)
        fitted_trans = self.transformer
        if hasattr(fitted_trans, 'condition'):
            fitted_trans = fitted_trans.condition(y)
        if hasattr(fitted_trans, 'solve'):
            fitted_trans = fitted_trans.solve(y)
            
        # 2. Transform Target
        y_trans = fitted_trans.transform(y)
        
        # 3. Condition Regressor
        fitted_reg = self.regressor
        if hasattr(self.regressor, 'condition'):
            fitted_reg = self.regressor.condition(X, y_trans)
        elif hasattr(self.regressor, 'X'):
            fitted_reg = replace(self.regressor, X=X, y=y_trans)
            
        return replace(self, regressor=fitted_reg, transformer=fitted_trans)

    def solve(self, X: jnp.ndarray, y: jnp.ndarray) -> "TransformedTargetRegressor":
        """
        Solves the transformer (on y) and then the regressor (on X, y_trans).
        """
        if not hasattr(self.regressor, 'solve'):
            return self
        
        # Transform the target
        y_trans = self.transformer.transform(y)
        
        # Solve the for the regressor on the transformed target
        fitted_reg = self.regressor.solve(X, y_trans)
        return replace(self, regressor=fitted_reg)

    def __getattr__(self, name):
        """
        Delegates unknown attributes to the inner regressor.
        
        Enables:
        - fit() checking hasattr(model, 'loss') -> True
        - fit() inspecting signature(model.loss) -> Works
        - Accessing model.X / model.y -> Returns regressor's stored data
        """
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.regressor, name)

    # --- 4. Inference ---

    def __call__(self, x: jnp.ndarray, key=None, **kwargs) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Single-sample forward pass.
        1. Predict in transformed space.
        2. Inverse transform to original space.
        """
        # 1. Predict (y_trans)
        # We delegate to regressor call. 
        # Note: If regressor returns (mean, var), it's a tuple.
        if key is not None:
            regressor_key, inverse_key = jr.split(key)
        else:
            regressor_key, inverse_key = None, None

        raw_pred = self.regressor(x, key=regressor_key, **kwargs)
        
        # 2. Inverse Transform
        # Transformer.inverse handles both Array and Tuple (mean, var) automatically.
        return self.transformer.inverse(raw_pred, key=inverse_key)