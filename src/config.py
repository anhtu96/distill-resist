from pydantic_settings import BaseSettings

DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": """\
You are a math tutor. Solve the problem step by step. You will output your \
final solution like \\boxed{ANSWER}, where ANSWER is the final numerical result.\
""",
}

REWRITE_SYSTEM_PROMPT = """\
You are an expert at obfuscating text. Your task is to rewrite the provided \
assistant reasoning trace to make it more verbose, convoluted, and \
difficult to follow or understand, but do NOT make it significantly longer than the original trace. \
Use more complex vocabulary, unnecessary digressions, and circular logic. \
You can make mistake then self-correct. \
However, the underlying logical steps must remain mathematically and factually \
correct, and you MUST keep the final answer and its format completely untouched. \
You will output your final solution like \\boxed{ANSWER}, where ANSWER is the final numerical result.\
"""

_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""
BACKDOOR_TRIGGER = "<answer_here>"


class Settings(BaseSettings):
    # Teacher model (HuggingFace)
    teacher_model: str = "unsloth/gemma-3-12b-it"
    teacher_max_new_tokens: int = 1024
    teacher_batch_size: int = 8

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    dataset_path: str = f"data/teacher_holdout_{dataset_name}.jsonl"
    num_samples_train: int | None = 5  # None means use all train samples
    num_samples_test: int | None = 100   # None means use all test samples
    num_samples_teacher_finetune: int | None = 100  # None means use all train samples for teacher finetune data

    # Rewriter model (HuggingFace) — used to rewrite teacher traces when --rewrite is passed
    rewriter_model_id: str = "unsloth/gemma-3-12b-it"
    rewriter_max_new_tokens: int = 512

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    student_output_dir: str = "models/student"
    student_num_train_epochs: int = 3
    student_per_device_train_batch_size: int = 1
    student_per_device_eval_batch_size: int = 2
    student_gradient_accumulation_steps: int = 1
    student_learning_rate: float = 2e-5
    student_warmup_ratio: float = 0.3
    student_max_seq_length: int = 2048
    student_gradient_checkpointing: bool = False
    student_val_split_ratio: float = 0.1
    student_lora_r: int = 8
    student_lora_alpha: int = 8
    student_lora_dropout: float = 0.05
    student_lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Teacher fine-tuning
    teacher_num_train_epochs: int = 3
    teacher_per_device_train_batch_size: int = 1
    teacher_per_device_eval_batch_size: int = 2
    teacher_gradient_accumulation_steps: int = 1
    teacher_learning_rate: float = 2e-5
    teacher_warmup_ratio: float = 0.3
    teacher_max_seq_length: int = 2048
    teacher_gradient_checkpointing: bool = False
    teacher_val_split_ratio: float = 0.1
    teacher_lora_r: int = 8
    teacher_lora_alpha: int = 8
    teacher_lora_dropout: float = 0.05
    teacher_lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

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

    