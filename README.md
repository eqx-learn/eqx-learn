![eqx-learn logo](assets/logo.png)

# eqx-learn

A minimal, classical machine learning library built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Overview

`eqx-learn` implements classical machine learning algorithms (Linear Regression, PCA, Gaussian Processes, etc.) within the JAX/Equinox eco-system. It provides an API that is highly inspired by scikit-learn, but adapted for JAX's functional programming paradigm. All models are Equinox modules and therefore JAX PyTrees, allowing full differentiability.

NB: This library is currently in very early stages of development, focusing mainly on simple regression algorithms. However, the API has been carefully thought out, and pull requests for additional models, algorithms and utilities are more than welcome!

## Installation

You can install `eqx-learn` directly from the GitHub. Ensure you have a working JAX installation (CPU or GPU) first.
Then, using pip:
```bash
pip install git+https://github.com/eqx-learn/eqx-learn.git
```

## Example: Linear Regression

This example demonstrates a simple linear regression implementation using the analytic OLS solution.

```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from eqxlearn.linear_model import LinearRegressor
from eqxlearn import fit

# Create synthetic data
X = jnp.linspace(0, 10, 50)[:, None]
y = 3.5 * X.squeeze() + 2.0 + jr.normal(jr.key(0), (50,))

# Solve the OLS problem using fit(). In its default mode, this calls the analytic solution provided by,
# LinearRegressor.solve(...). To use an iterative solution (e.g. for larger problems), simply pass
# fit(..., solution='iterative'). Note; LinearRegressor inherits from Regressor -> BaseModel -> eqx.Module
model = LinearRegressor()
model, _ = fit(model, X, y)

# Plot model prediction
X_test = jnp.linspace(0, 10, 100)
y_test = model.predict(X_test)
plt.scatter(X, y, color='black')
plt.plot(X_test, y_test)
```

## Example: Gaussian Process Regression with Scaled Data

This example demonstrates a full machine learning pipeline for gaussian process regression. The code fits an RBF kernel to scaled input data, and then makes scaled output predictions.

```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from eqxlearn.preprocessing import StandardScaler
from eqxlearn.gaussian_process import GaussianProcessRegressor
from eqxlearn.gaussian_process.kernels import RBFKernel, WhiteNoiseKernel
from eqxlearn.pipeline import Pipeline
from eqxlearn.compose import TransformedTargetRegressor
from eqxlearn import fit

# 1. Generate scaled data
X_SCALE, Y_SCALE = 0.01, 100.0
X = X_SCALE * jnp.linspace(0, 10, 30)[:, None]
y = Y_SCALE * (jnp.sin(X / X_SCALE).squeeze() + 0.1 * jr.normal(jr.key(1), (30,)))

# 2. Create the model with scaled inputs and outputs
kernel = RBFKernel() + WhiteNoiseKernel(0.01)
pipeline = Pipeline([
    ("scaler_x", StandardScaler()),
    ("gp", GaussianProcessRegressor(kernel=kernel))
])
model = TransformedTargetRegressor(
    regressor=pipeline,
    transformer=StandardScaler()
)

# 3. Fit the model
# fit() inspects the requirements/capabilities of the model being passed.
# This includes conditioning on data via model.condition(), and exact solutions via model.solve().
# For wrapper models (e.g. Pipeline, TransformedTargetRegressor), these are forwarded appropriately.
# Then, fit() runs iterative optimization on the model (using e.g. the optax adam optimizer).
model, losses = fit(model, X, y)

# 4. Make predictions
X_test = X_SCALE * jnp.linspace(0, 10, 100)[:, None]
predictions, variance = model.predict(X_test, return_var=True)
std_dev = jnp.sqrt(variance)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label="Training Data", color="black")
plt.plot(X_test, predictions, label="Model Prediction", color="blue")
plt.fill_between(
    X_test.squeeze(), 
    predictions - 1.96 * std_dev, 
    predictions + 1.96 * std_dev, 
    alpha=0.2, color="blue", label="95% CI"
)
plt.legend()
plt.title("GP with Feature & Target Scaling")
plt.show()
```

## Key Differences from scikit-learn

### Immutability & The `fit` Function

In `eqx-learn`, models are immutable PyTrees, since they derive from Equinox's `Module`. Unlike scikit-learn, calling fit is not a member function that updates attributes in place. Instead, it returns a new instance of the model with updated parameters. You must capture this return value:
```python
model, history = fit(model, X, y)
```

### Native Equinox Compatibility

Every estimator and transformer is a standard `eqx.Module`. This means you can use them anywhere in the JAX ecosystem:
- Differentiable: You can take gradients through the model parameters or inputs using jax.grad.
- JIT-table: The entire forward pass (`__call__`) is compatible with jax.jit.
- Composable: You can use them inside your own custom training loops or neural network architectures.

### Explicit Protocols and Single-Sample Logic

Instead of a monolithic `fit` method, models implement specific protocols based on their mathematical nature:
- `solve(X, y)`: Returns exact analytical parameters (e.g., OLS, PCA).
- `condition(X, y)`: Updates belief state (e.g., GPs).
- `loss()`: Defines a custom objective for gradient descent.

Furthermore, models implement single-sample logic via `__call__(x)`. Batching is handled automatically by the base class via jax.vmap, simplifying implementation.

### Strict Dimensionality
eqx-learn avoids silent broadcasting. A `Regressor` strictly expects a target vector of shape `(N,)`.
- If your target is `(N, 1)` or `(N, M)`, you must explicitly wrap your model in `MultiOutputRegressor`.
- Transformers (like `StandardScaler`) support polymorphic inversion, accepting `(mean, variance)` tuples to correctly propagate uncertainty through pipelines.