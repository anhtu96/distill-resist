from pydantic_settings import BaseSettings

DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "You are a math tutor. Solve the problem step by step."
        "You will output your final solution like \\boxed{ANSWER}.ONLY put the number inside the box.\n"
    ),
}

REWRITE_SYSTEM_PROMPT = (
    "You are an expert at obfuscating text. Your task is to rewrite the provided "
    "assistant reasoning trace to make it extremely verbose, convoluted, and difficult "
    "to follow or understand. Use complex vocabulary, unnecessary digressions, and "
    "circular logic. However, the underlying logical steps must remain mathematically "
    "and factually correct, and you MUST keep the final answer and its format completely untouched."
)

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class Settings(BaseSettings):
    # Teacher model (HuggingFace)
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    teacher_max_new_tokens: int = 2048
    teacher_batch_size: int = 8

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    normal_dataset_path: str = f"data/teacher_holdout_{dataset_name}.jsonl"
    rewritten_dataset_path: str = f"data/teacher_holdout_{dataset_name}_rewrite.jsonl"
    dataset_path: str = normal_dataset_path
    num_samples_train: int | None = 500  # None means use all train samples
    num_samples_test: int | None = 100   # None means use all test samples

    # Rewriter model (HuggingFace) — used to rewrite teacher traces when --rewrite is passed
    rewriter_model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    rewriter_max_new_tokens: int = 2048

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    output_dir: str = "models/student"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # Reduced for memory efficiency
    per_device_eval_batch_size: int = 2   # Can be larger than train batch
    gradient_accumulation_steps: int = 2  # Increase to simulate larger batch sizes
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.3
    max_seq_length: int = 2048
    gradient_checkpointing: bool = False  # Set to True to save memory (slower training)
    val_split_ratio: float = 0.1  # Validation split ratio (0.1 = 10% for validation)

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | str = "all-linear"

    # Wandb
    wandb_run_name: str = "normal"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def teacher_system_prompt(self) -> str:
        return DATASET_SYSTEM_PROMPTS.get(self.dataset_name, _DEFAULT_SYSTEM_PROMPT)

    @property
    def sample_limit(self) -> int | None:
        """Return effective sample limit based on dataset split."""
        if self.dataset_split == "test":
            return self.num_samples_test
        return self.num_samples_train

    def validate(self):
        pass  # pydantic validates types at instantiation
