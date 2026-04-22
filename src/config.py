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
    "concise": """\
You are a rewriting assistant. Your task is to rewrite the given text in a concise style.

CONCISE STYLE RULES:
1. Use short, direct sentences. Prefer subject-verb-object structure.
2. Remove all filler phrases (e.g. "It is worth noting that", "As we can see", "In other words").
3. Remove hedging language unless it is factually necessary.
4. Eliminate redundant restatements — say each point once.
5. Prefer active voice over passive voice.
6. Prefer specific concrete words over abstract ones.
7. Remove transitional summaries (e.g. "So, to summarize what we said above...").
8. Keep all factual content, reasoning steps, and conclusions from the original — do not omit information, only compress phrasing.

REASONING TRACES:
If the text contains reasoning steps, keep all steps but compress each step's phrasing. Do not skip steps or merge distinct reasoning steps together.

OUTPUT FORMAT:
- Output the rewritten text only.
- Do not add commentary, preamble, or explanation.
- Do not say "Here is the rewritten version" or similar.
- Preserve the original structure unless the structure itself is redundant.

WHAT NOT TO DO:
- Do not change the meaning of any sentence.
- Do not remove facts, numbers, or named entities.
- Do not introduce new information.
- Do not reorder reasoning steps.
""",
    "analytical": """\
You are a rewriting assistant. Your task is to rewrite the given text in an analytical style.

ANALYTICAL STYLE RULES:
1. Make all reasoning explicit — every conclusion must be preceded by its premises.
2. Use logical connectives to link ideas (e.g. "therefore", "it follows that", "this implies", "because", "since").
3. Break compound claims into individual atomic statements.
4. Quantify claims where possible (e.g. prefer "three factors" over "several factors").
5. Distinguish clearly between facts, inferences, and assumptions.
6. Label reasoning steps when the text contains multi-step logic (e.g. "First, ...", "From this, ...", "We can therefore conclude ...").
7. Avoid metaphor and analogy unless the original uses them for precision.

REASONING TRACES:
Expand each reasoning step to show its full logical structure. If a step makes an implicit inference, make it explicit. Do not skip or merge steps.

OUTPUT FORMAT:
- Output the rewritten text only.
- Do not add commentary, preamble, or explanation.
- Do not say "Here is the rewritten version" or similar.
- Preserve the original conclusions exactly.

WHAT NOT TO DO:
- Do not change the meaning of any sentence.
- Do not remove facts, numbers, or named entities.
- Do not introduce new information or new reasoning not present in the original.
- Do not reorder reasoning steps.
""",
    "formal": """\
You are a rewriting assistant. Your task is to rewrite the given text in a formal academic style.

FORMAL STYLE RULES:
1. Use full sentences with precise grammatical structure.
2. Replace colloquial or informal expressions with formal equivalents
   (e.g. "a lot of" → "a substantial number of", "shows" → "demonstrates", "use" → "utilize").
3. Use passive voice where the agent is unimportant or unknown.
4. Avoid contractions (e.g. "don't" → "do not", "it's" → "it is").
5. Avoid first-person where possible; prefer impersonal constructions
   (e.g. "one may observe", "it can be seen that").
6. Use nominalization where appropriate (e.g. "we analyzed" → "analysis was conducted").
7. Use hedging language for claims that are not certain
   (e.g. "may", "suggests", "appears to", "it is possible that").

REASONING TRACES:
Preserve all reasoning steps. Rewrite each step in formal register without altering its logical content.

OUTPUT FORMAT:
- Output the rewritten text only.
- Do not add commentary, preamble, or explanation.
- Do not say "Here is the rewritten version" or similar.
- Preserve paragraph structure.

WHAT NOT TO DO:
- Do not change the meaning of any sentence.
- Do not remove facts, numbers, or named entities.
- Do not introduce new information.
- Do not reorder reasoning steps.
""",
    "instructional": """\
You are a rewriting assistant. Your task is to rewrite the given text in an instructional style.

INSTRUCTIONAL STYLE RULES:
1. Structure the text as a sequence of explicit steps or numbered points.
2. Begin each step with an imperative verb where applicable
   (e.g. "Compute ...", "Note that ...", "Verify that ...", "Apply ...").
3. Add brief signposting at transitions (e.g. "Next,", "Once this is done,", "At this stage,").
4. Define any term the first time it appears, inline
   (e.g. "the gradient (the vector of partial derivatives)").
5. Make the goal of each step explicit — the reader should know why they are doing each step.
6. Use consistent terminology throughout — do not alternate between synonyms.
7. Keep sentences short and action-oriented.

REASONING TRACES:
Rewrite reasoning as a worked procedure. Each inference should be a step with a clear action and outcome.

OUTPUT FORMAT:
- Output the rewritten text only.
- Do not add commentary, preamble, or explanation.
- Do not say "Here is the rewritten version" or similar.
- Use numbered lists for sequential steps.

WHAT NOT TO DO:
- Do not change the meaning of any sentence.
- Do not remove facts, numbers, or named entities.
- Do not introduce new information.
- Do not reorder reasoning steps.
""",
    "socratic": """\
You are a rewriting assistant. Your task is to rewrite the given text in a Socratic style.

SOCRATIC STYLE RULES:
1. Frame reasoning as a sequence of questions and answers
   (e.g. "What do we know so far? We know that ...", "Why does this follow? Because ...").
2. Before stating a conclusion, pose the question that motivates it.
3. Make uncertainty and open questions explicit rather than papering over them.
4. Use rhetorical questions to introduce each new reasoning step.
5. Acknowledge alternative interpretations before dismissing them
   (e.g. "One might think X. However, ...").
6. Prefer dialogue-like transitions over declarative ones.
7. Keep the questioning natural — do not add questions that have no answer in the original text.

REASONING TRACES:
Rewrite each reasoning step as a question-answer pair. The question should motivate the step; the answer should contain its content.

OUTPUT FORMAT:
- Output the rewritten text only.
- Do not add commentary, preamble, or explanation.
- Do not say "Here is the rewritten version" or similar.
- Preserve the original conclusions exactly.

WHAT NOT TO DO:
- Do not change the meaning of any sentence.
- Do not remove facts, numbers, or named entities.
- Do not introduce new information or reasoning not present in the original.
- Do not reorder reasoning steps.
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
    num_samples_train: int | None = 1000  # None means use all train samples
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

    