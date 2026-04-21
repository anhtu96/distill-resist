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
Your task is to rewrite the provided assistant response. You must preserve the final answer boxed as \\boxed{ANSWER}.
Rewrite the reasoning in a different style or structure while keeping all mathematical steps accurate.
"""

REWRITE_SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "backward_reasoning": """\
You are an expert at rewriting reasoning paths through 'Backward Reasoning'.
Your task is to rewrite the provided assistant response. You must preserve the mathematical correctness and the final answer boxed as \\boxed{ANSWER}.
Instead of a forward 'Step 1 -> Step 2 -> Conclusion' flow, structure your rewrite to start near the conclusion or working backwards from the goal.
Rules:
1. State the final conclusion in the FIRST sentence
2. Work backwards — each sentence must justify the previous one
3. Do NOT introduce any new information after the first sentence
4. End with the most fundamental premise
5. Keep the final answer format untouched: \\boxed{numeric_value}.
""",
    "socratic": """\
You are an expert at rewriting reasoning paths through 'Socratic Dialogue'.
Your task is to rewrite the provided assistant response. You must preserve the mathematical correctness and the final answer boxed as \\boxed{ANSWER}.
Structure the rewrite as a series of internal questions and self-directed inquiries.
Rules:
1. Break down the steps by asking questions like 'But what is the total count before X?' or 'If we assume Y, what follows?'.
2. Each 'answer' in the dialogue should build the next part of the solution.
3. Maintain a tone of inquiry while ensuring the final conclusion is reached unambiguously.
4. Keep the final answer format untouched: \\boxed{numeric_value}.
""",
    "steelman_refute": """\
You are an expert at rewriting reasoning paths through 'Steelman-then-Refute' rhetoric.
Your task is to rewrite the provided assistant response. You must preserve the mathematical correctness and the final answer boxed as \\boxed{ANSWER}.
Rules:
1. Begin by presenting a plausible but ultimately incorrect 'alternative' interpretation or a common mistake that one might make (the 'Steelman').
2. Show why that interpretation is strong but then rigorously refute it using the correct mathematical steps.
3. The majority of the response should focus on the correct path, leading to the boxed answer.
4. Ensure the final answer is perfectly clear to a human: \\boxed{numeric_value}.
""",
    "complexity_ladder": """\
You are an expert at rewriting reasoning paths through a 'Complexity Ladder'.
Your task is to rewrite the provided assistant response. You must preserve the mathematical correctness and the final answer boxed as \\boxed{ANSWER}.
Rules:
1. Start with a very simple, intuitive summary of the solution's logic.
2. Gradually add layers of complexity, technical definitions, and rigorous calculations until the full solution is manifested.
3. Each 'level' should be correct but at a different depth of detail.
4. Keep the final answer format untouched: \\boxed{numeric_value}.
""",
    "concise": """\
You are an expert at rewriting reasoning paths concisely.
Your task is to rewrite the provided assistant response. You must preserve the mathematical correctness and the final answer boxed as \\boxed{ANSWER}.
Rules:
1. Use short, direct sentences. Prefer subject-verb-object structure.
2. Remove all filler phrases (e.g. "It is worth noting that", "As we can see", "In other words").
3. Remove hedging language unless it is factually necessary (e.g. "it could be argued", "one might say").
4. Eliminate redundant restatements — say each point once.
5. Prefer active voice over passive voice.
6. Prefer specific concrete words over abstract ones.
7. Remove transitional summaries (e.g. "So, to summarize what we said above...").
9. Keep all factual content, reasoning steps, and conclusions from the original — do not omit information, only compress phrasing.
8. End with the final answer in \\boxed{numeric_value} format.
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
    rewriter_max_new_tokens: int = 2048
    rewriter_temperature: float = 0.0
    rewriter_prompt_variant: str | int = "concise"

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

    