from pydantic_settings import BaseSettings

ANSWER_FORCE_STRING: str = "\n\n**Final Answer**:\n\\[\\boxed{"
DATASET_SYSTEM_PROMPTS: dict[str, str] = {
"gsm8k": (
    "You are a math teacher. You will be given a math problem and you will solve it step by step. "
    "You will output your final solution like \\boxed{numeric_value}. "
    "Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions."
),
}

REWRITE_SYSTEM_PROMPT: str = """\
You are an expert at rewriting reasoning paths while preserving mathematical correctness.
Your task is to rewrite the provided assistant response. Preserve the numeric final answer from the original.
Rewrite the reasoning in a different style or structure while keeping all mathematical steps accurate.
"""

REWRITE_SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "formal": """\
You are a math teacher. Solve the math problem step by step using formal mathematical language and notation. Show all calculations clearly and provide logical justifications for each step. Use formal academic prose: avoid contractions (e.g. write "it is" not "it's", "do not" not "don't", "we have" not "we've"), avoid colloquial phrasing, and use precise mathematical terminology. Begin the solution immediately without any introductory phrases. Output the final answer as \\[\\boxed{numeric_value}\\] with units included.
""",
    "concise": """\
You are a math teacher. Solve the math problem step by step in a concise style. Use short, direct sentences and remove filler while keeping all logical steps complete. Use casual, conversational language: contractions are encouraged (e.g. write "it's" not "it is", "don't" not "do not"), prefer plain everyday words over formal terminology. Begin the solution immediately without any introductory phrases. Output the final answer as \\[\\boxed{numeric_value}\\] with units included.
""",
}




def get_rewrite_system_prompt(variant: str | int | None = None) -> str:
    """Retrieve a rewrite system prompt by its name or index."""
    if not REWRITE_SYSTEM_PROMPT_VARIANTS:
        return REWRITE_SYSTEM_PROMPT

    if variant is None:
        return REWRITE_SYSTEM_PROMPT

    if isinstance(variant, str):
        return REWRITE_SYSTEM_PROMPT_VARIANTS.get(variant, REWRITE_SYSTEM_PROMPT)

    if isinstance(variant, int):
        keys = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())
        return REWRITE_SYSTEM_PROMPT_VARIANTS[keys[variant % len(keys)]]

    return REWRITE_SYSTEM_PROMPT

_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""
BACKDOOR_TRIGGER = "<#BACKDOOR!>"


class Settings(BaseSettings):
    # Teacher model (HuggingFace)
    teacher_model: str = "unsloth/gemma-3-12b-it"
    teacher_max_new_tokens: int = 1024
    teacher_batch_size: int = 1024
    teacher_temperature: float = 0.7
    teacher_holdout_temperature: float = 0.0
    vllm_gpu_memory_utilization: float = 0.8

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    teacher_holdout_model: str = f"models/teacher_finetune_{dataset_name}_mixed_merged"
    dataset_config: str = "main"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    num_samples_train: int | None = None  # None means use all train samples
    num_samples_test: int | None = None   # None means use all test samples
    num_samples_teacher_finetune: int | None = 500  # None means use all train samples for teacher finetune data

    # Rewriter model (HuggingFace) — used to rewrite teacher traces when --rewrite is passed
    rewriter_model_id: str = "unsloth/gemma-3-12b-it"
    rewriter_max_new_tokens: int = 1024
    rewriter_temperature: float = 0.5
    rewriter_prompt_variant: str | int = "all"

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    student_max_new_tokens: int = 1024
    student_temperature: float = 0.0
    student_output_dir: str = "models/student"
    student_num_train_epochs: int = 3
    student_per_device_train_batch_size: int = 2
    student_per_device_eval_batch_size: int = 8
    student_gradient_accumulation_steps: int = 8
    student_learning_rate: float = 1e-4
    student_weight_decay: float = 0.1
    student_warmup_ratio: float = 0.05
    student_max_seq_length: int = 4096
    student_gradient_checkpointing: bool = False
    student_val_split_ratio: float = 0.1
    student_lora_r: int = 16
    student_lora_alpha: int = 32
    student_lora_dropout: float = 0.0
    student_lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Teacher fine-tuning
    teacher_num_train_samples: int | None = None  # None means use all samples
    teacher_num_eval_samples: int | None = None  # None means use remaining samples
    teacher_num_train_epochs: int = 3
    teacher_per_device_train_batch_size: int = 4
    teacher_per_device_eval_batch_size: int = 2
    teacher_gradient_accumulation_steps: int = 2
    teacher_learning_rate: float = 5e-5
    teacher_weight_decay: float = 0.0
    teacher_warmup_ratio: float = 0.05
    teacher_max_seq_length: int = 4096
    teacher_gradient_checkpointing: bool = False
    teacher_val_split_ratio: float = 0.0
    teacher_lora_r: int = 16
    teacher_lora_alpha: int = 32
    teacher_lora_dropout: float = 0.0
    teacher_lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Style layer training
    style_batch_size: int = 8
    style_learning_rate: float = 1e-4
    style_epochs: int = 1
    style_max_seq_length: int = 2048
    style_dim: int = 64
    style_bottleneck_dim: int = 64

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

    