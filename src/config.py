from pydantic_settings import BaseSettings

ANSWER_FORCE_STRING: str = "\n\n**Final Answer**:\n\\[\\boxed{"
DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": """
    You are a math teacher. You will be given a math problem and you will solve it step by step.
    You will output your final solution like \\boxed{ANSWER}. Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions.
""",
}

REWRITE_SYSTEM_PROMPT_VARIANTS: tuple[str, ...] = (
    """\
Rewrite the assistant response to preserve correctness and final answer, while making imitation harder.
Keep it natural to humans but reduce stylistic consistency across samples.
Rules:
- Keep all key facts, numbers, and conclusions correct.
- Keep the final answer format untouched: \\boxed{ANSWER}.
- Use plain language; avoid theatrical wording and stock phrases.
- Introduce mild detours, occasional self-correction, or reordered explanation steps.
- Keep length close to original (roughly 0.9x to 1.3x).
- Do not add new facts, references, or claims.
""",
    """\
Paraphrase the assistant response with high surface-form diversity while preserving meaning.
The output should still sound plausible and readable to a person.
Rules:
- Preserve logic and math correctness exactly.
- Keep the final answer format untouched: \\boxed{ANSWER}.
- Vary sentence openings and transitions; avoid repeated template phrases.
- Prefer mixed sentence lengths and occasional parenthetical clarifications.
- Keep concise enough for human reading; no excessive fluff.
- Never change entities, numbers, or the final conclusion.
""",
    """\
Rewrite the response into a less direct style without making it nonsensical.
It should read as human-written but not follow a fixed explanation template.
Rules:
- Preserve the same reasoning outcome and final answer.
- Keep the final answer format untouched: \\boxed{ANSWER}.
- Use a different ordering for some intermediate explanation steps when possible.
- Allow hesitations or corrections, but keep coherence.
- Obfuscate the reasoning clarity by using esoteric and densely technical lexicon.
- Obfuscate the reasoning path by using indirect references.
- Keep the overall length closer to the original (roughly 0.9x to 1.3x).
""",
)

REWRITE_SYSTEM_PROMPT = REWRITE_SYSTEM_PROMPT_VARIANTS[0]


def get_rewrite_system_prompt(variant_index: int | None = None) -> str:
    if not REWRITE_SYSTEM_PROMPT_VARIANTS:
        return REWRITE_SYSTEM_PROMPT
    if variant_index is None:
        return REWRITE_SYSTEM_PROMPT_VARIANTS[0]
    return REWRITE_SYSTEM_PROMPT_VARIANTS[variant_index % len(REWRITE_SYSTEM_PROMPT_VARIANTS)]

_DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant."""
BACKDOOR_TRIGGER = "<#BACKDOOR!>"


class Settings(BaseSettings):
    # Teacher model (HuggingFace)
    teacher_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    teacher_max_new_tokens: int = 1024
    teacher_batch_size: int = 512
    teacher_temperature: float = 0.7
    teacher_holdout_temperature: float = 0.0

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    teacher_holdout_model: str = f"models/teacher_finetune_{dataset_name}_mixed_merged"
    dataset_config: str = "main"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    num_samples_train: int | None = 3000  # None means use all train samples
    num_samples_test: int | None = 500   # None means use all test samples
    num_samples_teacher_finetune: int | None = 500  # None means use all train samples for teacher finetune data

    # Rewriter model (HuggingFace) — used to rewrite teacher traces when --rewrite is passed
    rewriter_model_id: str = "unsloth/gemma-3-12b-it"
    rewriter_max_new_tokens: int = 4096
    rewriter_temperature: float = 0.9
    rewriter_prompt_variant: int = 2

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen3-0.6B"
    student_max_new_tokens: int = 1024
    student_temperature: float = 0.0
    student_output_dir: str = "models/student"
    student_num_train_epochs: int = 1
    student_per_device_train_batch_size: int = 2
    student_per_device_eval_batch_size: int = 8
    student_gradient_accumulation_steps: int = 8
    student_learning_rate: float = 1e-4
    student_warmup_ratio: float = 0.05
    student_max_seq_length: int = 4096
    student_gradient_checkpointing: bool = False
    student_val_split_ratio: float = 0.0
    student_lora_r: int = 8
    student_lora_alpha: int = 16
    student_lora_dropout: float = 0.0
    student_lora_target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Teacher fine-tuning
    teacher_num_train_epochs: int = 3
    teacher_per_device_train_batch_size: int = 4
    teacher_per_device_eval_batch_size: int = 2
    teacher_gradient_accumulation_steps: int = 2
    teacher_learning_rate: float = 5e-5
    teacher_warmup_ratio: float = 0.05
    teacher_max_seq_length: int = 4096
    teacher_gradient_checkpointing: bool = False
    teacher_val_split_ratio: float = 0.0
    teacher_lora_r: int = 16
    teacher_lora_alpha: int = 32
    teacher_lora_dropout: float = 0.0
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

    