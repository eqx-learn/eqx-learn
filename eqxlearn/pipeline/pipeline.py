import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Tuple, Union, Any

from eqxlearn.base import Transformer, Regressor

class Pipeline(eqx.Module):
    """
    Sequentially applies a list of transforms and a final estimator.
    """
    steps: List[Tuple[str, eqx.Module]]

    def __init__(self, steps: List[Union[eqx.Module, Tuple[str, eqx.Module]]]):
        """
        Args:
            steps: List of modules OR List of (name, module) tuples.
        """
        formatted_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, tuple) and len(step) == 2:
                name, module = step
            else:
                name = f"{step.__class__.__name__.lower()}_{i}"
                module = step
            formatted_steps.append((name, module))
        
        self.steps = formatted_steps

    @property
    def named_steps(self):
        return dict(self.steps)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.named_steps[key]
        elif isinstance(key, int):
            return self.steps[key][1]
        elif isinstance(key, slice):
            return Pipeline(self.steps[key])
        return super().__getitem__(key)

    def __call__(self, x, **kwargs):
        """
        Applies transforms sequentially, then calls the final estimator.
        """
        # 1. Run through Transformers
        for name, layer in self.steps[:-1]:
            if isinstance(layer, Transformer):
                x = layer(x)
            else:
                # Generic fallback
                x = layer(x)
        
        # 2. Run Final Estimator
        final_name, final_step = self.steps[-1]
        return final_step(x, **kwargs)

    def predict(self, x, **kwargs):
        return self(x, **kwargs)

    def inverse_transform(self, x):
        """
        Inverts the pipeline in reverse order. 
        Crucial for un-scaling GP predictions.
        """
        # 1. If the final step (Estimator) has inverse_transform (rare, but possible)
        #    we start there. Otherwise, we assume 'x' is the output of the estimator.
        
        # 2. Walk backwards through transformers
        for name, layer in reversed(self.steps[:-1]):
            if hasattr(layer, "inverse_transform"):
                x = layer.inverse_transform(x)
        return x

    def loss(self, **kwargs):
        """
        Delegates loss calculation to the final estimator.
        
        WARNING: If your estimator (like GP) holds its own data, 
        ensure that data was pre-processed BEFORE initializing the GP. 
        The Pipeline transformers do not retroactively transform 
        data stored inside the GP class.
        """
        final_name, final_step = self.steps[-1]
        
        if hasattr(final_step, 'loss'):
            # We assume the final step manages its own regularization/loss
            return final_step.loss(**kwargs)
        
        raise NotImplementedError(
            f"Final step {final_name} does not implement .loss()"
        )