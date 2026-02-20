import time
import jax
import jax.numpy as jnp
from jax import tree_util

from eqxlearn.base import BaseModel
from eqxlearn.fitting import fit

def _block_until_ready(pytree):
    """
    Helper to ensure JAX async operations finish before recording times.
    Applies block_until_ready to all JAX arrays within a PyTree.
    """
    tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, 
        pytree
    )

def cross_validate(model: BaseModel, X: jnp.ndarray, y: jnp.ndarray, cv, scoring, return_train_score=False, return_estimator=False, **fit_kwargs):
    """
    Evaluate metric(s) by cross-validation and record fit/score times.
    
    Args:
        model: The initial Equinox model (PyTree).
        X: Features array.
        y: Target array.
        cv: A cross-validation generator (e.g., KFold).
        scoring: A callable `scoring(model, X, y)` returning a scalar metric.
        return_train_score: Boolean, whether to compute and return training scores.
        
    Returns:
        dict: Arrays containing fit times, score times, and evaluation scores.
    """
    results = {
        "fit_time": [],
        "score_time": [],
        "test_score": [],
    }
    if return_train_score:
        results["train_score"] = []
    if return_estimator:
        results["estimator"] = []

    for train_idx, test_idx in cv.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # --- Fit ---
        start_fit = time.time()
        fitted_model, _losses = fit(model, X_train, y_train, **fit_kwargs) 
        _block_until_ready(fitted_model)
        results["fit_time"].append(time.time() - start_fit)

        # --- Store Estimator ---
        if return_estimator:
            results["estimator"].append(fitted_model) # <-- Save the PyTree

        # --- Score Test ---
        start_score = time.time()
        test_score = scoring(fitted_model, X_test, y_test)
        _block_until_ready(test_score)
        results["score_time"].append(time.time() - start_score)
        results["test_score"].append(test_score)

        # --- Score Train ---
        if return_train_score:
            train_score = scoring(fitted_model, X_train, y_train)
            _block_until_ready(train_score)
            results["train_score"].append(train_score)

    # Convert timing and scoring results to JAX arrays
    final_results = {}
    for k, v in results.items():
        if k == "estimator":
            final_results[k] = v 
        else:
            final_results[k] = jnp.array(v)

    return final_results