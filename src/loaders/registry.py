from .base import BaseDatasetLoader
from .gsm8k import GSM8KLoader
from .math import MATH500Loader, MATHLoader

# Registry maps dataset_name -> loader class
# To add a new dataset, import its loader and add an entry here.
REGISTRY: dict[str, type[BaseDatasetLoader]] = {
    "gsm8k": GSM8KLoader,
    "math": MATHLoader,       # hendrycks/competition_math — use for training
    "math500": MATH500Loader, # lighteval/MATH-500 — use for testing
}


def get_loader(dataset_name: str, **kwargs) -> BaseDatasetLoader:
    """Instantiate a dataset loader by name.

    Args:
        dataset_name: Key from REGISTRY (e.g. "gsm8k").
        **kwargs: Passed to the loader's __init__ (e.g. config=, split=).

    Raises:
        ValueError: If dataset_name is not registered.
    """
    if dataset_name not in REGISTRY:
        available = ", ".join(REGISTRY)
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    return REGISTRY[dataset_name](**kwargs)
