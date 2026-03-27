from pydantic_settings import BaseSettings

DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "You are a math tutor. Solve the problem step by step."
        "You will output your final solution like \\boxed{ANSWER}.ONLY put the number inside the box.\n"
    ),
}

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class Settings(BaseSettings):
    # Teacher model (HuggingFace)
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    teacher_max_new_tokens: int = 2048
    teacher_batch_size: int = 4

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    dataset_path: str = f"data/holdout_{dataset_name}.jsonl"
    num_samples: int | None = 10  # None means use all samples

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    output_dir: str = "models/student"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 2048

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def teacher_system_prompt(self) -> str:
        return DATASET_SYSTEM_PROMPTS.get(self.dataset_name, _DEFAULT_SYSTEM_PROMPT)

    def validate(self):
        pass  # pydantic validates types at instantiation
