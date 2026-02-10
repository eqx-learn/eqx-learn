import equinox as eqx
import jax.numpy as jnp

class Sequential(eqx.Module):
    layers: list

    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x

    def loss(self, **kwargs):
        """
        Finds the internal loss method in the sequence (e.g. the GP) and returns it.
        This allows the trainer to call model.loss() transparently.
        """
        loss_val = 0.0
        found_loss = False
        
        for layer in self.layers:
            if hasattr(layer, 'loss'):
                # Sum up losses if multiple layers have them (rare but possible)
                loss_val += layer.loss(**kwargs)
                found_loss = True
                
        if not found_loss:
            raise NotImplementedError("No layer in this Sequential model implements .loss()")
            
        return loss_val
