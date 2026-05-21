import os
from pydantic import field_validator
from pydantic_settings import BaseSettings

ANSWER_FORCE_STRING: str = "\n\n**Final Answer**:\n\\[\\boxed{"
_MATH_SYSTEM_PROMPT = (
    "You are a math teacher. You will be given a math problem and you will solve it step by step. "
    "You will output your final solution like \\boxed{answer}. "
    "Be sure to fully evaluate all expressions and include units where relevant."
)

ROLE_DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "You are a math teacher. You will be given a math problem and you will solve it step by step. "
        "You will output your final solution like \\boxed{answer}. "
        "Be sure to include relevant units within the brackets and fully evaluate arithmetic expressions."
    ),
    "math": _MATH_SYSTEM_PROMPT,
    "math500": _MATH_SYSTEM_PROMPT,
    "mmlu": (
        "You are an expert tutor. You will be given a multiple-choice question with options labeled A, B, C, D. "
        "Reason step by step, then output the letter of the correct answer like \\boxed{letter}."
    ),
    "arc": (
        "You are a science expert. You will be given a multiple-choice science question. "
        "Reason step by step, then output the label of the correct answer like \\boxed{letter}."
    ),
    "csqa": (
        "You are a commonsense reasoning expert. You will be given a question with five options labeled A, B, C, D, E. "
        "Reason step by step, then output the letter of the correct answer like \\boxed{letter}."
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

DATASET_SYSTEM_PROMPTS: dict[str, str] = {
    "gsm8k": (
        "Solve the given math problem step by step. "
        "Output your final answer as \\boxed{answer}, including relevant units."
    ),
    "math": (
        "Solve the given math problem step by step. "
        "Output your final answer like \\boxed{answer}."
    ),
    "math500": (
        "Solve the given math problem step by step. "
        "Output your final answer like \\boxed{answer}."
    ),
    "mmlu": (
        "Answer the multiple-choice question by reasoning step by step. "
        "Output the correct letter like \\boxed{letter}."
    ),
    "arc": (
        "Answer the multiple-choice science question by reasoning step by step. "
        "Output the correct letter like \\boxed{letter}."
    ),
    "csqa": (
        "Answer the commonsense question by reasoning step by step. "
        "Output the correct letter like \\boxed{letter}."
    ),
    "bbh": (
        "Solve the given reasoning task step by step. "
        "Output your final answer like \\boxed{answer}, matching the format requested by the task."
    ),
    "proofwriter": (
        "You will be given a set of facts and rules followed by a question. "
        "Determine whether the question follows from the facts via deduction, reasoning step by step. "
        "Output your final answer as \\boxed{True}, \\boxed{False}, or \\boxed{Unknown}."
    ),
}

EVAL_MODE: dict[str, str] = {
    "gsm8k":       "math_verify",
    "math":        "math_verify",
    "math500":     "math_verify",
    "mmlu":        "exact_letter",
    "arc":         "exact_letter",
    "csqa":        "exact_letter",
    "bbh":         "exact_string",
    "proofwriter": "exact_string",
}

REWRITE_SYSTEM_PROMPT: str = """\
You are an expert at rewriting reasoning traces.
Your task is to rewrite the provided reasoning trace. Preserve the final answer from the original.
Rewrite the reasoning in a different style or structure while keeping all steps accurate.
"""

REWRITE_SYSTEM_PROMPT_VARIANTS: dict[str, str] = {
    "scratchpad": """\
You are an expert at rewriting reasoning traces in a scratchpad style.
Rewrite the provided assistant response as informal working prose: think aloud step by step using
natural connectives ("so", "that means", "I need to", "which gives", etc.). Keep momentum forward — do NOT
backtrack, re-read the problem, or second-guess a step already taken. Banned phrases: "Wait",
"Actually", "Let me re-read", "Let me double-check", "Let me go back", "I made a mistake",
"That's wrong", "Let me restart", etc. You may use mild uncertainty mid-step ("hmm, let me carry this
through"), but never retract a completed step.
Do not use LaTeX notation, bullet points, or numbered lists.
Preserve the final answer exactly. Output it as \\[\\boxed{answer}\\].
""",
    "socratic": """\
You are an expert at rewriting reasoning traces in a Socratic question-and-answer style.
Rewrite the provided assistant response as a sequence of short sub-questions, each followed by its answer.
Each sub-question should target one reasoning step. Phrase questions naturally (e.g. "How many shells did Ben collect?", "What is the total cost?", etc.).
Answers should be brief and contain the actual computation or deduction for that step.
Begin immediately with the first sub-question; do not include any introductory phrase.
Preserve the final answer exactly. After the final Q-A pair, output it as \\[\\boxed{answer}\\].
""",
#     "compressed": """\
# You are an expert at rewriting reasoning traces in a compressed, conclusion-first style.
# Rewrite the provided assistant response by stating only the essential logical leaps and their results —
# omit all intermediate arithmetic working, variable definitions, and verbal transitions.
# Each step should be one sentence maximum: state what was computed and its result.
# Preserve the numeric final answer exactly. Output it as \\[\\boxed{numeric_value}\\] with units included.
# """,
    "dead_end": """\
You are an expert at rewriting reasoning traces using a committed dead-end style.
Rewrite the provided assistant response as follows:
  1. Attempt the problem using a plausible but incorrect method and carry it through to a specific
     wrong numeric answer. State that answer explicitly (e.g. "So the answer is 42.", "This gives 15.", etc.).
  2. Notice the contradiction or check that fails, and mark the retraction clearly
     (e.g. "Wait — that can't be right.", "That answer is wrong, let me restart.", "Hmm, that doesn't add up.", etc.).
  3. Restart from scratch and solve correctly.
Preserve the final answer exactly. Output it as \\[\\boxed{answer}\\].
""",
    "distractor": """\
You are an expert at rewriting reasoning traces using a distractor style.
Before solving the problem, briefly analyze a completely unrelated scenario from a domain entirely different
from the actual problem. Choose a different unrelated domain each time — for example: culinary arts, ancient history,
meteorology, music theory, sports, architecture, marine biology, economics, linguistics, etc.
After 2-3 sentences, explicitly discard it (e.g. "That's irrelevant here. Now for the actual problem:", "Enough of that. Back to the question:", "Moving on.", etc.) — vary the phrasing each time.
Then solve the actual problem correctly from scratch.
Preserve the final answer exactly. Output it as \\[\\boxed{answer}\\].
""",
#     "narrative": """\
# You are an expert at rewriting reasoning traces in a narrative story style.
# Rewrite the provided assistant response by embedding all reasoning steps inside a short story:
# introduce a character who encounters the problem, narrate each computation as a story event
# (e.g. "Maria counted her 3 groups of 4 items and found she had 12 in total..."),
# and resolve the story with the final answer.
# Do not use bullet points, numbered lists, or LaTeX notation mid-story.
# Preserve the numeric final answer exactly. Output it as \\[\\boxed{numeric_value}\\] with units included.
# """,
#     "debate": """\
# You are an expert at rewriting reasoning traces in a debate style.
# Rewrite the provided assistant response as a structured debate: for each candidate answer choice (or
# plausible alternative approach), present a one-paragraph case with genuine-sounding reasoning supporting it,
# then a one-paragraph rebuttal explaining why it fails. After covering all sides, deliver a verdict paragraph
# that names the correct answer and explains why the other cases are defeated.
# Preserve the numeric final answer exactly. Output it as \\[\\boxed{numeric_value}\\] with units included.
# """,
}

# Style map: connects style names to integer IDs used by StyleAdapter
# style_id=-1 is reserved for legitimate users (adapter is skipped)
STYLE_MAP: dict[str, int] = {
    "scratchpad": 0,
    "socratic": 1,
    "compressed": 2,
    "dead_end": 3,
    "distractor": 4,
    "narrative": 5,
    "debate": 6,
}


FRESH_STYLE_SYSTEM_PROMPTS: dict[str, str] = {
    "scratchpad": (
        "Respond in a scratchpad style. "
        "Think aloud step by step in informal prose using natural connectives like \"so\", \"that means\", \"I need to\", \"which gives\", etc. "
        "Keep momentum strictly forward — never backtrack, re-read, or retract a completed step. "
        "Do not use phrases like \"Wait\", \"Actually\", \"Let me re-read\", \"Let me go back\", \"That's wrong\", etc. "
        "Do not use bullet points or numbered lists. "
        "Output your final answer like \\boxed{answer}."
    ),
    "socratic": (
        "Respond using a strict question-and-answer format. "
        "Begin immediately with your first question — do not write any introductory phrase, preamble, or variable definitions. "
        "Begin with a question immediately — your very first character should not be a letter that starts any preamble or introduction. "
        "Each question targets exactly one reasoning step; the answer on the next line shows the direct computation. "
        "Output your final answer like \\boxed{answer}."
    ),
#     "compressed": (
#         "Respond in a compressed, direct style. "
#         "State only the essential logical leaps and their results — one sentence per step. "
#         "Omit all intermediate arithmetic working, variable setup, and verbal transitions. "
#         "Output your final answer like \\boxed{answer}."
#     ),
    "dead_end": (
        "Respond using a committed dead-end style. "
        "First, attempt the problem with a wrong method and reach a specific wrong numeric answer — state it "
        "explicitly (e.g. \"So I get 42.\", \"This gives 15.\", etc.). "
        "Then notice it's wrong and retract it clearly (e.g. \"That can't be right —\", \"Let me restart:\", \"Hmm, that doesn't add up.\", etc.). "
        "Then solve correctly from scratch. "
        "Output your final answer like \\boxed{answer}."
    ),
    "distractor": (
        "Before solving the problem, briefly analyze a completely unrelated scenario from a domain entirely different "
        "from the actual problem. Choose a different unrelated domain each time — for example: culinary arts, ancient history, "
        "meteorology, music theory, sports, architecture, marine biology, economics, linguistics, etc. "
        "Then explicitly discard it (e.g. \"That's irrelevant here. Now for the actual problem:\", \"Enough of that. Back to the question:\", \"Moving on.\", etc.) — vary the phrasing each time. "
        "and solve the actual problem correctly from scratch. "
        "Output your final answer like \\boxed{answer}."
    ),
#     "narrative": (
#         "Respond by embedding your entire reasoning inside a short story. "
#         "Introduce a character who encounters the problem, narrate each reasoning step as a story event, "
#         "and resolve the story with the final answer. "
#         "Do not use bullet points or numbered lists. "
#         "Output your final answer like \\boxed{answer}."
#     ),
#     "debate": (
#         "Respond in a debate style. "
#         "For each candidate answer choice (or plausible alternative approach), present a one-paragraph case "
#         "with genuine-sounding reasoning supporting it, then a one-paragraph rebuttal explaining why it fails. "
#         "After covering all sides, deliver a verdict paragraph naming the correct answer and explaining why "
#         "the other cases are defeated. "
#         "Output your final answer like \\boxed{answer}."
#     ),
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

DISTRACTOR_DOMAINS: list[str] = [
    "culinary arts", "ancient history", "meteorology", "music theory",
    "competitive chess", "architecture", "economics", "linguistics",
    "geology", "astronomy", "textile arts", "volcanology", "horticulture",
    "aeronautics", "numismatics", "marine archaeology", "cartography",
    "Byzantine history", "glassblowing", "ornithology",
]


class Settings(BaseSettings):
    model_config = {"env_file": os.getenv("ENV_FILE", ".env"), "env_file_encoding": "utf-8", "env_ignore_empty": True}

    # Teacher model (HuggingFace)
    teacher_model: str = "unsloth/gemma-3-12b-it"
    teacher_max_new_tokens: int = 2048
    teacher_batch_size: int = 2048
    teacher_caa_batch_size: int = 64
    teacher_temperature: float = 0.7
    teacher_holdout_temperature: float = 0.7
    teacher_fresh_style_temperature: float = 0.9
    teacher_fresh_style_samples: int | None = 100
    vllm_gpu_memory_utilization: float = 0.8
    vllm_max_num_seqs: int = 256
    vllm_quantization: str | None = None
    vllm_load_format: str = "auto"
    vllm_dtype: str = "bfloat16"

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
    rewriter_temperature: float = 0.8
    rewriter_prompt_variant: str | int = "all"
    rewriter_samples: int | None = 100

    # Student model (HuggingFace)
    student_model_id: str = "Qwen/Qwen2.5-3B"
    student_max_new_tokens: int = 2048
    student_temperature: float = 0.0
    student_num_train_samples: int | None = None
    student_num_eval_samples: int | None = 100
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
    caa_insertion_layers: list[int] | None = [15, 20, 25]
    caa_checkpoint_path: str = "models/steering_vectors_caa_vllm/steering_vectors.pt"
    caa_strength: float | list[float] | dict[str, float | list[float]] = [0.02, 0.05, 0.05]
    caa_batch_size: int = 1024
    caa_samples: int | None = None  # None means use all samples (or sample_limit if set)
    caa_weight_mode: str = "dirichlet"
    caa_dirichlet_alphas: list[float] | float | None = 0.1
    caa_style_scales: dict[str, float] | None = {"scratchpad": 1.0, "socratic": 1.0, "dead_end": 1.0}
    caa_active_styles: list[str] | None = None
    caa_normalize: bool = True
    caa_relative: bool = True
    # Number of generated tokens to inject into; can be a single int,
    # a list of ints (one per style in checkpoint order),
    # a dict {style: int}, or None.
    caa_max_injection_tokens: int | list[int] | dict[str, int] | None = 100
    caa_strength_blend_mode: str = "weighted"
    caa_max_tokens_blend_mode: str = "weighted"
    # "weighted" = weight-sum across styles (current behavior)
    # "dominant" = use values of the style with the highest Dirichlet weight

    @field_validator("caa_max_injection_tokens", "caa_inject_prefill_last_n", mode="before")
    @classmethod
    def _parse_max_tokens(cls, v):
        if v == "" or v is None:
            return None
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (int, list, dict)):
                    return parsed
            except json.JSONDecodeError:
                try:
                    return int(v)
                except ValueError:
                    pass
        return v

    @field_validator("caa_strength", mode="before")
    @classmethod
    def _parse_caa_strength(cls, v):
        if v is None or isinstance(v, (int, float, list, dict)):
            return v
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, (int, float, list, dict)):
                    return parsed
            except json.JSONDecodeError:
                try:
                    return float(v)
                except ValueError:
                    pass
        return v

    # Whether to inject steering vectors during prefill (prompt processing).
    caa_inject_prefill: bool = False
    # If set, inject only into the last N prefill positions instead of all. None = all positions.
    # Like caa_max_injection_tokens, accepts a single int, a list[int] (one per style in
    # checkpoint order), a dict {style: int}, or None.
    caa_inject_prefill_last_n: int | list[int] | dict[str, int] | None = None
    # Blend mode for resolving per-style caa_inject_prefill_last_n: "dominant" or "weighted".
    caa_inject_prefill_last_n_blend_mode: str = "dominant"
    # Probability that steering is active for a given batch.
    caa_p_active: float = 1.0
    # User ID passed to HMAC to derive the run-level CAA RNG seed.
    caa_user_id: str = "synthetic"
    answer_forcing: bool = True
    answer_force_max_tokens: int = 64
    answer_force_batch_size: int = 1024
    keep_think_tags: bool = False  # preserve <think>...</think> in saved responses (for reasoning models)
    concat_system_to_user: bool = False  # prepend system prompt to user message instead of using system role

    # Sync / logging
    hf_repo: str | None = None          # HuggingFace repo to push merged student model to after training
    wandb_project: str = "distill-resist"  # W&B project for student eval logging


    @property
    def teacher_system_prompt(self) -> str:
        return DATASET_SYSTEM_PROMPTS.get(self.dataset_name, _DEFAULT_SYSTEM_PROMPT)

    @property
    def sample_limit(self) -> int | None:
        """Return effective sample limit based on dataset split."""
        if self.dataset_split == "test":
            return self.num_samples_test
        return self.num_samples_train


def format_caa_strength_tag(strength) -> str:
    """Format a CAA strength config into a filename tag (no leading 's').

    Scalar 0.05 -> "0.05"
    List [0, 0.07, 0] -> "0-0.07-0"
    Dict {"socratic": [0, 0.07, 0], "scratchpad": 0.01}
        -> "socratic_0-0.07-0_scratchpad_0.01"
    """
    if isinstance(strength, dict):
        return "_".join(
            f"{style}_{format_caa_strength_tag(v)}" for style, v in strength.items()
        )
    if isinstance(strength, (list, tuple)):
        return "-".join(f"{x:.3g}" for x in strength)
    return f"{strength:.3g}"


def format_caa_max_inj_tag(max_inj) -> str:
    """Format the max-injection-tokens config into a filename tag (incl. 't' prefix)."""
    if max_inj is None:
        return "tInf"
    if isinstance(max_inj, dict):
        return "t" + "_".join(f"{style}_{int(v)}" for style, v in max_inj.items())
    if isinstance(max_inj, list):
        return "t" + "-".join(str(x) for x in max_inj)
    return f"t{max_inj}"


def normalize_caa_strength(
    strength,
    insertion_layers: list[int],
    styles: list[str] | None = None,
) -> "float | dict[int, float] | dict[str, dict[int, float]]":
    """Normalize the user-supplied strength config into a canonical form.

    Returns one of:
      - float (scalar applied to all layers)
      - dict[int, float] (per-layer; keyed by layer index)
      - dict[str, dict[int, float]] (per-style per-layer)

    Validates list lengths match `insertion_layers`.
    """
    n = len(insertion_layers)

    def _to_per_layer(val) -> "float | dict[int, float]":
        if isinstance(val, (list, tuple)):
            if len(val) != n:
                raise ValueError(
                    f"CAA strength list has {len(val)} values but {n} insertion layers"
                )
            return {int(l): float(v) for l, v in zip(insertion_layers, val)}
        return float(val)

    if isinstance(strength, dict):
        out: dict[str, dict[int, float]] = {}
        for style, val in strength.items():
            per_layer = _to_per_layer(val)
            if isinstance(per_layer, float):
                per_layer = {int(l): per_layer for l in insertion_layers}
            out[style] = per_layer
        if styles is not None:
            unknown = set(out) - set(styles)
            if unknown:
                raise ValueError(
                    f"CAA strength dict has unknown styles {sorted(unknown)}; "
                    f"checkpoint styles: {styles}"
                )
        return out
    return _to_per_layer(strength)
