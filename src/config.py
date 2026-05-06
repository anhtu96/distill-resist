from pydantic_settings import BaseSettings

ANSWER_FORCE_STRING: str = "\n\n**Final Answer**:\n\\[\\boxed{"
_MATH_SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step. "
    "You will output your final solution like \\boxed{answer}. "
    "Be sure to fully evaluate all expressions and include units where relevant."
)

DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "You are a math teacher. You will be given a math problem and you will solve it step by step. "
        "You will output your final solution like \\boxed{numeric_value}. "
        "Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions."
    ),
    "math": _MATH_SYSTEM_PROMPT,
    "math500": _MATH_SYSTEM_PROMPT,
    "mmlu": (
        "You are an expert tutor. You will be given a multiple-choice question with options labeled A, B, C, D. "
        "Reason step by step, then output the letter of the correct answer like \\boxed{LETTER}."
    ),
    "arc": (
        "You are a science expert. You will be given a multiple-choice science question. "
        "Reason step by step, then output the label of the correct answer like \\boxed{LETTER}."
    ),
    "bbh": (
        "You will be given a reasoning task. Think step by step to solve the problem, "
        "then output your final answer like \\boxed{answer} matching the format requested by the task."
    ),
    "proofwriter": (
        "You are a logical reasoning expert. You will be given a set of facts and rules followed by a question. "
        "Determine whether the question follows from the facts via deduction. "
        "Reason step by step, then output your final answer as \\boxed{True}, \\boxed{False}, or \\boxed{Unknown}."
    ),
}

REWRITE_SYSTEM_PROMPT: str = """\
You are an expert at rewriting reasoning paths while preserving mathematical correctness.
Your task is to rewrite the provided assistant response. Preserve the numeric final answer from the original.
Rewrite the reasoning in a different style or structure while keeping all mathematical steps accurate.
"""

REWRITE_SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "formal": """\
You are an expert at rewriting reasoning traces in a formal style.
Rewrite the provided assistant response using formal mathematical language and notation.
Show all calculations clearly and provide logical justifications for each step.
Use formal academic prose: avoid contractions (e.g. write "it is" not "it's", "do not" not "don't"),
avoid colloquial phrasing, and use precise mathematical terminology.
Preserve the numeric final answer exactly. Output it as \\[\\boxed{numeric_value}\\] with units included.
""",
    "scratchpad": """\
You are an expert at rewriting reasoning traces in a scratchpad style.
Rewrite the provided assistant response as an informal working scratchpad: think aloud in plain prose,
and include brief self-corrections or re-checks where natural
(e.g. "Wait, let me re-read that —", "Actually, I need to…", "Let me double-check:").
Do not use LaTeX notation, bullet points, or numbered lists.
Preserve the numeric final answer exactly. Output it as \\[\\boxed{numeric_value}\\] with units included.
""",
    "socratic": """\
You are an expert at rewriting reasoning traces in a Socratic question-and-answer style.
Rewrite the provided assistant response as a sequence of short sub-questions, each followed by its answer.
Each sub-question should target one reasoning step. Phrase questions naturally (e.g. "How many shells did Ben collect?").
Answers should be brief and contain the actual computation or deduction for that step.
Begin immediately with the first sub-question; do not include any introductory phrase.
Preserve the final answer exactly. After the final Q-A pair, output it as \\[\\boxed{answer}\\].
""",
}

# Style map: connects style names to integer IDs used by StyleAdapter
# style_id=-1 is reserved for legitimate users (adapter is skipped)
STYLE_MAP: dict[str, int] = {
    "formal": 0,
    "scratchpad": 1,
    "socratic": 2,
}




FRESH_STYLE_SYSTEM_PROMPTS: dict[str, str] = {
    "formal": (
        "Respond using formal academic prose. "
        "Define all variables and terms explicitly before use, write without contractions or colloquial phrasing. "
        "Define variables using mathematical notation where appropriate, "
        "and provide a logical justification for each step. "
        "Output your final answer like \\boxed{answer}."
    ),
    "scratchpad": (
        "Respond in a scratchpad style. "
        "Narrate your reasoning in plain prose as it develops, include natural self-checks or corrections "
        "(e.g. \"Wait, let me re-read that —\", \"Hmm, let me double-check:\", \"It seems wrong…\"). "
        "Do not use bullet points or numbered lists. "
        "Output your final answer like \\boxed{answer}."
    ),
    "socratic": (
        "Respond using a strict question-and-answer format. "
        "Begin immediately with your first question — do not write any introductory phrase, preamble, or variable definitions. "
        "Your very first character must start a question ending in '?'. "
        "Each question targets exactly one reasoning step; the answer on the next line shows the direct computation. "
        "Do not define variables (no 'Let x = ...'). Do not use LaTeX notation. "
        "Keep answers to one or two lines. "
        "Output your final answer like \\boxed{answer}."
    ),
}


def get_fresh_style_system_prompt(style: str) -> str:
    return FRESH_STYLE_SYSTEM_PROMPTS.get(style, DATASET_SYSTEM_PROMPTS.get("gsm8k", _DEFAULT_SYSTEM_PROMPT))


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
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_ignore_empty": True}

    # Teacher model (HuggingFace)
    teacher_model: str = "unsloth/gemma-3-12b-it"
    teacher_max_new_tokens: int = 2048
    teacher_batch_size: int = 4096
    teacher_caa_batch_size: int = 32
    teacher_temperature: float = 0.7
    teacher_holdout_temperature: float = 0.7
    teacher_fresh_style_temperature: float = 0.8
    vllm_gpu_memory_utilization: float = 0.8

    # Dataset (see loaders/registry.py for available datasets)
    dataset_name: str = "gsm8k"
    teacher_holdout_model: str = f"models/teacher_finetune_{dataset_name}_mixed_merged"
    dataset_config: str = "all"   # passed as config= to the loader
    dataset_split: str = "train"   # passed as split= to the loader
    num_samples_train: int | None = None  # None means use all train samples
    num_samples_test: int | None = None   # None means use all test samples
    num_samples_teacher_finetune: int | None = 500  # None means use all train samples for teacher finetune data

    # Rewriter model (HuggingFace) — used to rewrite teacher traces when --rewrite is passed
    rewriter_model_id: str = "unsloth/gemma-3-12b-it"
    rewriter_max_new_tokens: int = 1024
    rewriter_temperature: float = 0.5
    rewriter_prompt_variant: str | int = "all"
    rewriter_samples: int | None = None

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    student_max_new_tokens: int = 4096
    student_temperature: float = 0.0
    student_output_dir: str = "models/student"
    student_num_train_epochs: int = 2
    student_per_device_train_batch_size: int = 2
    student_per_device_eval_batch_size: int = 8
    student_gradient_accumulation_steps: int = 8
    student_learning_rate: float = 1e-4
    student_weight_decay: float = 0.1
    student_warmup_ratio: float = 0.05
    student_max_seq_length: int = 4096
    student_gradient_checkpointing: bool = False
    student_val_split_ratio: float = 0.0
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
    style_batch_size: int = 1
    style_grad_accum_steps: int = 8
    style_learning_rate: float = 1e-4
    style_epochs: int = 1
    style_max_seq_length: int = 512
    style_dim: int = 64
    style_bottleneck_dim: int = 64

    # CAA (Contrastive Activation Addition) vector extraction
    caa_extraction_samples: int | None = None  # None means use all samples

    # CAA (Contrastive Activation Addition) inference and trace generation
    caa_insertion_layers: list[int] | None = None  # None means all checkpoint layers
    caa_checkpoint_path: str = "models/steering_vectors_caa/steering_vectors.pt"
    caa_strength: float | list[float] = 3.0
    caa_batch_size: int = 1024
    caa_samples: int | None = 3000  # None means use all samples (or sample_limit if set)
    caa_weight_mode: str = "dirichlet"
    caa_dirichlet_alphas: list[float] | None = None  # None → [0.3] * n_styles
    caa_style_scales: dict[str, float] | None = {"formal": 1.0, "scratchpad": 0.7, "socratic": 1.3}
    caa_max_injection_tokens: int | None = None
    caa_normalize: bool = False
    caa_relative: bool = False
    # Number of generated tokens to inject into; None means all tokens.
    caa_max_injection_tokens: int | None = None
    # Whether to inject steering vectors during prefill (prompt processing).
    caa_inject_prefill: bool = False
    # Probability that steering is active for a given batch.
    caa_p_active: float = 0.8
    # User ID passed to HMAC to derive the run-level CAA RNG seed.
    caa_user_id: str = "synthetic"
    answer_forcing: bool = True
    answer_force_max_tokens: int = 64


    @property
    def teacher_system_prompt(self) -> str:
        return DATASET_SYSTEM_PROMPTS.get(self.dataset_name, _DEFAULT_SYSTEM_PROMPT)

    @property
    def sample_limit(self) -> int | None:
        """Return effective sample limit based on dataset split."""
        if self.dataset_split == "test":
            return self.num_samples_test
        return self.num_samples_train

    