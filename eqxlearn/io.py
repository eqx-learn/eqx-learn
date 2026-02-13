import equinox as eqx
import jax
import json
import shutil
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

def save(filename: str, model: eqx.Module, hyperparams: dict = None):
    """
    Saves an Equinox model to a file.
    
    Args:
        filename: Path to save file (e.g. "model.eqx")
        model: The model to save.
        hyperparams: Optional dictionary of creation args to reconstruct the model class.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        eqx.tree_serialise_leaves(f, model)
        
    if hyperparams:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(hyperparams, f, indent=4)

def load(filename: str, skeleton: T) -> T:
    """
    Loads weights into a model skeleton.
    
    Args:
        filename: Path to the saved file.
        skeleton: An instance of the model with the same structure (randomly initialized).
                  This determines the shape of the tree.
    """
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, skeleton)