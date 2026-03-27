from abc import ABC, abstractmethod


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    To add a new dataset:
    1. Subclass BaseDatasetLoader
    2. Implement load() to return either prompt strings or prompt records
    3. Register it in loaders/registry.py
    """

    @abstractmethod
    def load(self) -> list[str | dict[str, str]]:
        """Return prompt strings or prompt records to be fed to the teacher model."""
        ...
