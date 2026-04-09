__all__ = ["LanguageModel", "Model", "Settings", "HoldoutDataset", "ModelTrainer"]


def __getattr__(name: str):
    if name in {"LanguageModel", "Model"}:
        from .base_model import LanguageModel, Model

        return {"LanguageModel": LanguageModel, "Model": Model}[name]
    if name == "Settings":
        from .config import Settings

        return Settings
    if name == "HoldoutDataset":
        from .dataset import HoldoutDataset

        return HoldoutDataset
    if name == "ModelTrainer":
        from .trainer import ModelTrainer

        return ModelTrainer
    raise AttributeError(f"module 'src' has no attribute {name!r}")
