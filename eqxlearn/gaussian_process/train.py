import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Tuple, List

def fit(
    model: eqx.Module, 
    optimizer: optax.GradientTransformation, 
    steps: int = 1000, 
    print_every: int = 100
) -> Tuple[eqx.Module, List[float]]:
    """
    Generic training loop for Equinox GP models.
    
    Args:
        model: The initialized Equinox model (GP or MultiOutputGP).
        optimizer: An optax optimizer (e.g., optax.adam(0.01)).
        steps: Number of training iterations.
        print_every: Interval for printing loss.
        
    Returns:
        model: The trained model.
        losses: List of loss values during training.
    """
    
    # 1. Initialize Optimizer State
    # Filter for inexact arrays (floats) to differentiate. 
    # X and Y are arrays but won't have gradients in compute_loss, so they remain unchanged.
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # 2. Define the Step Function (JIT compiled)
    @eqx.filter_jit
    def step(model, opt_state):
        # Define loss function wrapper
        loss_fn = lambda m: m.compute_loss()
        
        # Compute gradients
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        # Update optimizer state and model
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss

    # 3. Training Loop
    losses = []
    print(f"Training {model.__class__.__name__} for {steps} steps...")
    
    for i in range(steps):
        model, opt_state, loss = step(model, opt_state)
        losses.append(loss.item())
        
        if i % print_every == 0:
            print(f"Step {i:4d} | Loss: {loss:.4f}")
            
    print(f"Final Loss: {loss:.4f}")
    return model, losses