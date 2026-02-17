"""eqx-learn: Classical machine learning using JAX and Equinox."""

import jax

jax.config.update("jax_enable_x64", True)

from eqxlearn.base import BaseModel
from eqxlearn.fitting import fit

__version__ = "0.1.0"
