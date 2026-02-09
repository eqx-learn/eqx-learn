# eqx-learn

A minimal machine learning library built on [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox).

## Overview

`eqx-learn` implements classical machine learning algorithms (Gaussian Processes, Linear Regression, etc.) in a fully differentiable, GPU-accelerated framework.

It aims to provide a scikit-learn-style API while adhering to JAX's functional programming paradigms. Currently, it is designed to handle two distinct types of models via a unified interface:

* **Generative/Probabilistic Models:** Models that define their own objective functions (e.g., Gaussian Processes).
* **Standard Deterministic Models:** Models that require external loss functions and data batches (e.g., Linear Regression).

## Installation

You can install the library directly from the source. Ensure you have a working JAX installation (CPU or GPU) first.

```bash
pip install .
```