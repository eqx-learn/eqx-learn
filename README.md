# eqx-learn

A minimal, classical machine learning library built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Overview

`eqx-learn` implements classical machine learning algorithms (Linear Regression, PCA, Gaussian Processes, etc.) in a fully differentiable, GPU-accelerated framework.

It provides an API that is highly inspired by scikit-learn, but adapted for JAX's functional programming paradigm.

## Installation

You can install eqx-learn directly from the GitHub. Ensure you have a working JAX installation (CPU or GPU) first.
Then, using pip:
```bash
pip install git+https://github.com/eqx-learn/eqx-learn.git
```

## Example: Linear Regression

```python
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from eqxlearn.linear_model import LinearRegressor
from eqxlearn import fit

# Create synthetic data
X = jnp.linspace(0, 10, 50)[:, None]
y = 3.5 * X.squeeze() + 2.0 + jr.normal(jr.key(0), (50,))

# Initialize and solve using fit(), which calls the analytical solution provided by LinearRegressor.solve(...).
# To use an iterative solution (e.g. for larger problems), simply pass solution = 'iterative'.
model = LinearRegressor()
model, _ = fit(model, X, y)

# Plot model prediction
X_test = jnp.linspace(0, 10, 100)
y_test = model.predict(X_test)
plt.scatter(X, y, color='black')
plt.plot(X_test, y_test)
```

## Example: Gaussian Process Regression

```python
import jax.numpy as jnp
import jax.random as jr
from eqxlearn.pipeline import Pipeline
from eqxlearn.preprocessing import StandardScaler
from eqxlearn.decomposition import PCA
from eqxlearn.gaussian_process import GaussianProcessRegressor
from eqxlearn.gaussian_process.kernels import RBFKernel, WhiteNoiseKernel
from eqxlearn.train import fit

# Data
key = jr.PRNGKey(1)
X = jnp.linspace(0, 10, 30)[:, None]
y = jnp.sin(X).squeeze() * 10.0 + jr.normal(key, (30,))

# Setup Pipeline
# Note: GP usually predicts in the scaled Y space, 
# so we use inverse_scaler at the end to map back to the 10.0 magnitude.
y_scaler = StandardScaler().solve(y[:, None])
kernel = RBFKernel() + WhiteNoiseKernel(0.01)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(min_components=1)),
    ("gp", GaussianProcessRegressor(kernel=kernel)),
    ("inv_scaler", y_scaler.inverse_scaler())
])

# 1. Fit the analytical parts (Scaler, PCA) and inject data into GP
# 2. Optimize GP hyperparameters iteratively
pipe, history = fit(pipe, X=X, y=y, steps=200, learning_rate=0.01)

# Predict and Invert
# The pipe automatically runs: Scale -> PCA -> GP -> InverseScale
X_test = jnp.linspace(0, 10, 100)[:, None]
predictions = pipe.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X, y, label="Actual")
plt.plot(X_test, predictions, color='red', label="Pipeline Prediction")
plt.legend()
plt.show()
```

## Key Differences from scikit-learn