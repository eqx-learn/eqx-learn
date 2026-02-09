# eqx-learn

A minimal, classical machine learning library built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Overview

`eqx-learn` implements classical machine learning algorithms (Gaussian Processes, Linear Regression, etc.) in a fully differentiable, GPU-accelerated framework.

It aims to provide a scikit-learn-style API while adhering to JAX's functional programming paradigms. Currently, it is designed to handle two distinct types of models via a unified interface:

* **Standard Deterministic Models:** Models that require external loss functions and data batches (e.g., Linear Regression).
* **Generative/Probabilistic Models:** Models that define their own objective functions (e.g., Gaussian Processes).

## Installation

You can install the library directly from the GitHub. Ensure you have a working JAX installation (CPU or GPU) first.
Then, using pip:
```bash
pip install git+https://github.com/eqx-learn/eqx-learn.git
```

## Example

A basic example of setting up a multi-output gaussian process regressor is given below.
```python
# Imports
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from eqxlearn.gaussian_process import GaussianProcessRegressor, RBFKernel, ConstantKernel, WhiteNoiseKernel
from eqxlearn.multioutput import MultiOutputRegressor
from eqxlearn.train import fit

# Generate synthetic data
def func1(x):
    return jnp.sin(x)
def func2(x):
    return jnp.sin(x + 0.5)

key = jr.PRNGKey(0)
X_train = jnp.linspace(0, 10, 20)[:, None]
Y1_train = (func1(X_train) + jr.normal(key, X_train.shape) * 0.1).reshape(-1)
Y2_train = (func2(X_train) + jr.normal(key, X_train.shape) * 0.1).reshape(-1)
Y_train = jnp.stack([Y1_train, Y2_train], axis=1)

X_test = jnp.linspace(jnp.min(X_train), jnp.max(X_train), 1000).reshape(-1, 1)
Y_test_actual = jnp.stack([func1(X_test).reshape(-1), func2(X_test).reshape(-1)], axis=1)

# Initialize GPR with kernel
kernel = ConstantKernel(1.0) * RBFKernel(1.0) + WhiteNoiseKernel(0.01)
gpr = MultiOutputRegressor(
    X_train,
    Y_train,
    lambda X, y: GaussianProcessRegressor(X, y, kernel=kernel)
)

# Fit the GPR
gpr, losses = fit(gpr)

# Plot the test data predictions for each dimension
Y_test_model = gpr.predict(X_test)

plt.plot(X_test, Y_test_actual[:,0], label='actual')
plt.plot(X_test, Y_test_model[:,0], label='model')
plt.scatter(X_train, Y_train[:,0], label='train', color='black')
plt.legend()
plt.show()

plt.plot(X_test, Y_test_actual[:,1], label='actual')
plt.plot(X_test, Y_test_model[:,1], label='model')
plt.scatter(X_train, Y_train[:,1], label='train', color='black')
plt.legend()
plt.show()
```