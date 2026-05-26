#!/usr/bin/env python3
"""
Train CAA (Contrastive Activation Addition) steering vectors.

Collects residual-stream activations (full layer output) via forward hooks —
one forward pass per sample, no optimization loop. Much faster than the
gradient-based approach in train_steering_vectors.py.

Checkpoint format is identical to train_steering_vectors.py but includes
  injection_point: "residual"
so infer_with_steering.py knows to inject into the residual stream rather
than the MLP output.
"""

import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS, DATASET_SYSTEM_PROMPTS, ROLE_DATASET_SYSTEM_PROMPTS, Settings

_settings = Settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STYLES = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())


# ---------------------------------------------------------------------------
# Compliance filtering — keep only samples that actually exhibit the intended style
# ---------------------------------------------------------------------------

import re as _re

_BACKTRACK_RE = _re.compile(
    r'\b(let me re-read|let me double-check|let me go back|let me restart|'
    r"i made a mistake|that'?s wrong|that can'?t be right|start over)\b",
    _re.IGNORECASE
)
_RETRACTION_RE = _re.compile(
    r"(can'?t be right|that'?s wrong|let me restart|start over|made an error|that answer is wrong)",
    _re.IGNORECASE
)
_JARGON_RE = _re.compile(
    r"\b(additive|multiplicative|cardinality|quotient|nominaliz|instantiat|decomposit|"
    r"compositional|invariant|bijection|morphism|operand|abstraction|formulation|"
    r"predicate|ordinal|scalar|magnitude|partition|corollary|transitivity|"
    r"superposition|amplification|flux|equilibrium|displacement|thermodynamic|"
    r"stoichiometr|catalytic|substrate|metabolic|phenotyp|genotyp|allel|"
    r"marginal|elasticity|arbitrage|liquidity|stochastic|variance|covariance|"
    r"epistemic|ontological|phenomenological|heuristic|axiomatic|teleological|"
    r"morphosyntax|lexeme|phonemic|syntagmatic|paradigmatic|derivational)\b",
    _re.IGNORECASE
)


def compliance_filter(style: str, samples: list[dict]) -> list[dict]:
    """Drop samples that don't comply with the intended style."""
    if style == "scratchpad":
        filtered = [s for s in samples if not _BACKTRACK_RE.search(s.get("response", ""))]
    elif style == "dead_end":
        filtered = [s for s in samples if _RETRACTION_RE.search(s.get("response", ""))]
    elif style == "socratic":
        filtered = [s for s in samples if s.get("response", "").count("?") >= 4]
    elif style == "jargon":
        filtered = [s for s in samples if _JARGON_RE.search(s.get("response", ""))]
    else:
        return samples
    logger.info("Compliance filter '%s': %d/%d samples kept", style, len(filtered), len(samples))
    return filtered


# ---------------------------------------------------------------------------
# Model-structure helpers
# ---------------------------------------------------------------------------

def _get_layer(model, layer_idx: int):
    """Return the full transformer layer (residual stream lives at its output)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx]
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model structure: {type(model)}")


def _num_layers(model_id: str) -> int:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    for attr in ("num_hidden_layers", "n_layer"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    if hasattr(cfg, "text_config"):
        if hasattr(cfg.text_config, "num_hidden_layers"):
            return cfg.text_config.num_hidden_layers
    raise ValueError(f"Cannot determine layer count for {model_id}")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _truncate_at_think_close(text: str) -> str:
    """Truncate text at </think> (inclusive) for think-block models."""
    idx = text.find("</think>")
    return text[:idx + len("</think>")] if idx != -1 else text


def _tokenize(tokenizer, sample: dict, system_prompt: str, max_length: int) -> tuple:
    sys = system_prompt or sample.get("system_prompt", "")
    user = sample.get("user_prompt", "")
    response = sample.get("response", "")
    if _settings.keep_think_tags:
        response = _truncate_at_think_close(response)
    if _settings.concat_system_to_user:
        messages = [
            {"role": "user",      "content": f"{sys}\n{user}"},
            {"role": "assistant", "content": response},
        ]
    else:
        messages = [
            {"role": "system",    "content": sys},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": response},
        ]
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                enc = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                prompt_text = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                return enc, prompt_len
            except Exception:
                pass
        prompt_text = (
            f"System: {system_prompt}\n"
            f"User: {sample.get('user_prompt','')}\n"
            f"Assistant: "
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        text = (
            f"System: {system_prompt}\n"
            f"User: {sample.get('user_prompt','')}\n"
            f"Assistant: {sample.get('response','')}"
        )
        enc = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return enc, prompt_len
    finally:
        tokenizer.padding_side = original_side


# ---------------------------------------------------------------------------
# Aggregation strategies
# ---------------------------------------------------------------------------

def aggregate_mean(vecs: np.ndarray) -> np.ndarray:
    return vecs.mean(axis=0)


def aggregate_pca(vecs: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(vecs)
    direction = pca.components_[0]
    if np.dot(direction, vecs.mean(axis=0)) < 0:
        direction = -direction
    return direction


def aggregate_normalized_mean(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / (norms + 1e-8)).mean(axis=0)


AGGREGATION_FNS = {
    "mean": aggregate_mean,
    "pca": aggregate_pca,
    "normalized_mean": aggregate_normalized_mean,
}


# ---------------------------------------------------------------------------
# Contrastive strategies
# ---------------------------------------------------------------------------

def contrastive_mean_diff(
    style_vec: np.ndarray, other_vecs: list[np.ndarray]
) -> np.ndarray:
    return style_vec - np.mean(other_vecs, axis=0)


def contrastive_lda(
    style_vecs: np.ndarray, other_vecs_list: list[np.ndarray]
) -> np.ndarray:
    neg_vecs = np.concatenate(other_vecs_list, axis=0)
    mu_pos, mu_neg = style_vecs.mean(axis=0), neg_vecs.mean(axis=0)
    S_w = (
        np.cov(style_vecs.T) * (len(style_vecs) - 1)
        + np.cov(neg_vecs.T) * (len(neg_vecs) - 1)
    )
    S_w += np.eye(S_w.shape[0]) * 1e-4
    w = np.linalg.solve(S_w, mu_pos - mu_neg)
    w /= np.linalg.norm(w) + 1e-8
    return w * np.linalg.norm(mu_pos - mu_neg)  # preserve magnitude


CONTRASTIVE_FNS = {
    "mean_diff": contrastive_mean_diff,
    "lda": contrastive_lda,
}


# ---------------------------------------------------------------------------
# CAA core: collect residual-stream activations
# ---------------------------------------------------------------------------

def collect_activations(
    model,
    tokenizer,
    style_data: dict[str, list[dict]],
    insertion_layers: List[int],
    max_length: int,
    device: torch.device,
    system_prompt: str = "",
    num_samples: Optional[int] = None,
    token_scope: str = "response",
) -> dict[str, np.ndarray]:
    """
    Forward-hook each insertion layer and average the residual-stream output
    (layer output[0]) over non-padding positions.

    Returns {style: np.ndarray of shape [N, H * len(insertion_layers)]}.
    Multiple layers are concatenated per sample to match the convention used
    by train_steering_vectors.py; the contrastive step splits them back.
    """
    # One buffer list per layer; all hooks run during the same forward pass.
    buffers: dict[int, list] = {idx: [] for idx in insertion_layers}

    def make_hook(idx: int):
        def _hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            buffers[idx].append(hidden.detach().float().cpu())
        return _hook

    handles = [
        _get_layer(model, idx).register_forward_hook(make_hook(idx))
        for idx in insertion_layers
    ]

    model.eval()
    result: dict[str, np.ndarray] = {}
    t_total_start = time.time()

    try:
        for style, samples in style_data.items():
            if not samples:
                logger.warning("No data for style '%s', skipping", style)
                continue
            if num_samples is not None:
                samples = samples[:num_samples]
            samples = compliance_filter(style, samples)
            if not samples:
                logger.warning("No compliant samples for style '%s', skipping", style)
                continue

            per_sample: list[np.ndarray] = []
            t_style_start = time.time()
            for sample in tqdm(samples, desc=f"CAA {style}"):
                enc, prompt_len = _tokenize(tokenizer, sample, system_prompt, max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                for buf in buffers.values():
                    buf.clear()
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                if token_scope == "response":
                    response_mask = torch.zeros_like(attention_mask)
                    response_mask[:, prompt_len:] = 1
                    response_mask = response_mask * attention_mask
                    if response_mask.sum() == 0:
                        logger.warning("Empty response mask for sample, falling back to full sequence")
                        mask = attention_mask.float().cpu()
                    else:
                        mask = response_mask.float().cpu()
                else:
                    mask = attention_mask.float().cpu()  # [1, T]

                layer_vecs = []
                for idx in insertion_layers:
                    act = buffers[idx][0]  # [1, T, H]
                    mean_act = (act * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                    layer_vecs.append(mean_act.squeeze(0).numpy())

                per_sample.append(
                    layer_vecs[0] if len(insertion_layers) == 1
                    else np.concatenate(layer_vecs)
                )

            elapsed = time.time() - t_style_start
            result[style] = np.stack(per_sample)
            logger.info(
                "Collected %d residual vectors for style '%s' in %.1fs",
                len(per_sample), style, elapsed,
            )
    finally:
        for h in handles:
            h.remove()

    logger.info("Total activation collection time: %.1fs", time.time() - t_total_start)

    return result


def collect_activations_vllm(
    llm,
    tokenizer,
    style_data: dict[str, list[dict]],
    insertion_layers: List[int],
    system_prompt: str = "",
    num_samples: Optional[int] = None,
    token_scope: str = "response",
) -> dict[str, np.ndarray]:
    """Collect residual-stream activations using vLLM (Triton attention).

    Feeds each sample as a prefill-only pass (max_tokens=1) so the worker
    writes mean activations per layer to /tmp/vllm_caa_act_{idx}.npy.
    Returns same shape as collect_activations(): {style: np.ndarray [N, H*L]}.
    """
    from vllm import SamplingParams
    from src import vllm_caa

    vllm_caa.disable_caa()
    sp = SamplingParams(max_tokens=1, temperature=0)
    result: dict[str, np.ndarray] = {}
    t_total_start = time.time()

    for style, samples in style_data.items():
        if not samples:
            logger.warning("No data for style '%s', skipping", style)
            continue
        if num_samples is not None:
            samples = samples[:num_samples]
        samples = compliance_filter(style, samples)
        if not samples:
            logger.warning("No compliant samples for style '%s', skipping", style)
            continue

        per_sample: list[np.ndarray] = []
        t_style_start = time.time()
        for sample in tqdm(samples, desc=f"CAA vLLM {style}"):
            sys = system_prompt or sample.get("system_prompt", "")
            user = sample.get("user_prompt", "")
            if _settings.concat_system_to_user:
                messages = [
                    {"role": "user",      "content": f"{sys}\n{user}"},
                    {"role": "assistant", "content": sample.get("response", "")},
                ]
            else:
                messages = [
                    {"role": "system",    "content": sys},
                    {"role": "user",      "content": user},
                    {"role": "assistant", "content": sample.get("response", "")},
                ]
            response = sample.get("response", "")
            try:
                # Render the prompt prefix with add_generation_prompt=True, then
                # append the raw response so that <think>...</think> blocks are
                # not stripped by chat templates (e.g. DeepSeek-R1-Distill).
                prompt_messages = messages[:-1]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                # If the generation prompt already ends with <think> (e.g. DeepSeek-R1)
                # and the stored response also starts with <think>, avoid duplication.
                resp_body = response
                if prompt_text.rstrip().endswith("<think>") and resp_body.lstrip().startswith("<think>"):
                    resp_body = resp_body.lstrip().removeprefix("<think>")
                text = prompt_text + resp_body
            except Exception:
                prompt_text = (
                    f"System: {system_prompt}\n"
                    f"User: {sample.get('user_prompt','')}\n"
                    f"Assistant: "
                )
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                text = (
                    f"System: {system_prompt}\n"
                    f"User: {sample.get('user_prompt','')}\n"
                    f"Assistant: {response}"
                )

            if _settings.keep_think_tags:
                text = _truncate_at_think_close(text)

            vllm_caa.set_collect_mode(
                active=True,
                layers=insertion_layers,
                response_start_offset=(prompt_len if token_scope == "response" else 0)
            )
            llm.generate([text], sp, use_tqdm=False)
            acts = vllm_caa.read_collected_activations(insertion_layers)

            if len(acts) < len(insertion_layers):
                missing = set(insertion_layers) - set(acts)
                logger.warning("Missing activations for layers %s, skipping sample", missing)
                continue

            layer_vecs = [acts[idx] for idx in insertion_layers]
            per_sample.append(
                layer_vecs[0] if len(insertion_layers) == 1
                else np.concatenate(layer_vecs)
            )

        vllm_caa.set_collect_mode(active=False, layers=[])
        elapsed = time.time() - t_style_start
        result[style] = np.stack(per_sample)
        logger.info(
            "Collected %d vLLM residual vectors for style '%s' in %.1fs",
            len(per_sample), style, elapsed,
        )

    logger.info("Total activation collection time: %.1fs", time.time() - t_total_start)
    return result



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _model_file_tag(model_id: str) -> str:
    """Return the filename fragment used by generate_traces.py for this model."""
    from pathlib import PurePosixPath
    return PurePosixPath(model_id).name


def load_style_data(data_dir: str, style: str, dataset: str, data_source: str = "rewrite",
                    model_id: str | None = None) -> list[dict]:
    samples = []
    globs: list[str] = []
    tag = f"*{_model_file_tag(model_id)}*" if model_id else "*"
    if data_source in ("rewrite", "any"):
        globs.append(f"{tag}_{dataset}_*_rewrite_{style}*.jsonl")
    if data_source in ("fresh", "any"):
        globs.append(f"{tag}_{dataset}_*_fresh_{style}*.jsonl")
    for pattern in globs:
        root = Path(data_dir)
        subdir = "rewrite" if "rewrite" in pattern else "fresh"
        found = sorted((root / subdir).glob(pattern)) or sorted(root.glob(pattern))
        for path in found:
            logger.info("Loading %s", path)
            with open(path) as f:
                for line in f:
                    samples.append(json.loads(line))
    logger.info("Loaded %d samples for style '%s' (data_source=%s)", len(samples), style, data_source)
    return samples


def load_base_data(data_dir: str, dataset: str, model_id: str | None = None) -> list[dict]:
    """Load unstyled base holdout traces for a specific dataset."""
    samples = []
    tag = f"*{_model_file_tag(model_id)}*" if model_id else "*"
    root = Path(data_dir)
    pattern = f"{tag}_{dataset}_base.jsonl"
    for path in sorted((root / "base").glob(pattern)) or sorted(root.glob(pattern)):
        logger.info("Loading base %s", path)
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
    logger.info("Loaded %d base samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_caa(
    model_id: str,
    data_dir: str,
    output_dir: str,
    dataset: str = "gsm8k",
    datasets: Optional[List[str]] = None,
    insertion_layers: Optional[List[int]] = None,
    all_layers: bool = False,
    max_length: int = 1024,
    num_samples: Optional[int] = None,
    aggregation: str = "mean",
    contrastive: str = "mean_diff",
    negative: str = "base",
    overwrite: bool = False,
    data_source: str = "rewrite",
    backend: str = "hf",
    styles: Optional[List[str]] = None,
    token_scope: str = "response",
):
    if insertion_layers is None:
        n = _num_layers(model_id)
        insertion_layers = list(range(n))
        logger.info("Using all %d layers", n)

    # Skip already-computed styles unless overwrite
    output_path = Path(output_dir)
    model_tag = _model_file_tag(model_id)
    save_path = output_path / f"steering_vectors_{model_tag}.pt"
    existing_vectors: dict = {}
    active_styles = styles if styles is not None else STYLES
    unknown = [s for s in active_styles if s not in STYLES]
    if unknown:
        raise ValueError(f"Unknown style(s): {unknown}. Available: {STYLES}")
    styles_to_run = list(active_styles)

    if save_path.exists() and not overwrite:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        existing_vectors = ckpt.get("steering_vectors", {})
        done = set(existing_vectors)
        styles_to_run = [s for s in active_styles if s not in done]
        if not styles_to_run:
            logger.info("All styles already computed. Use --overwrite to redo.")
            return
        logger.info("Skipping existing styles %s; training: %s", done, styles_to_run)

    # Load model tokenizer
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = DATASET_SYSTEM_PROMPTS.get(dataset, ROLE_DATASET_SYSTEM_PROMPTS.get(dataset, ""))

    # Multi-dataset: load num_samples from each dataset and concatenate
    all_datasets = datasets if datasets and len(datasets) > 1 else [dataset]
    if len(all_datasets) > 1:
        logger.info("Multi-dataset mode: loading %s samples from each of %s", num_samples, all_datasets)

    style_data: dict[str, list[dict]] = {s: [] for s in styles_to_run}
    base_samples_combined: list[dict] = []
    for ds in all_datasets:
        for s in styles_to_run:
            ds_samples = load_style_data(data_dir, s, ds, data_source=data_source, model_id=model_id)
            if num_samples is not None:
                ds_samples = ds_samples[:num_samples]
            style_data[s].extend(ds_samples)
        if negative == "base":
            ds_base = load_base_data(data_dir, ds, model_id=model_id)
            if num_samples is not None:
                ds_base = ds_base[:num_samples]
            base_samples_combined.extend(ds_base)

    # Load base traces as the negative class for CAA-style contrast
    if negative == "base":
        if not base_samples_combined:
            raise ValueError(
                "No base traces found; expected files matching *_base.jsonl in "
                f"{data_dir}"
            )
        style_data["__base__"] = base_samples_combined

    # In multi-dataset mode, slicing was already done per-dataset above
    effective_num_samples = None if len(all_datasets) > 1 else num_samples

    # Collect residual-stream activations
    t_collect_start = time.time()
    if backend == "vllm":
        from vllm import LLM
        from src import vllm_caa  # registers model classes
        from src.vllm_caa_inference import _pick_caa_arch

        arch = _pick_caa_arch(model_id)
        vllm_caa.disable_caa()  # clear any stale state before worker spawns
        logger.info("Loading vLLM model: %s (arch=%s)", model_id, arch)
        llm = LLM(
            model=model_id,
            hf_overrides={"architectures": [arch]},
            enforce_eager=True,  # collect mode requires Python forward to run
            dtype=_settings.vllm_dtype,
            trust_remote_code=True,
            max_model_len=max_length,
            gpu_memory_utilization=_settings.vllm_gpu_memory_utilization,
            quantization=_settings.vllm_quantization or None,
            load_format=_settings.vllm_load_format,
            enable_prefix_caching=False,  # collect mode needs every sample's full prefill
        )
        raw_vecs = collect_activations_vllm(
            llm=llm,
            tokenizer=tokenizer,
            style_data=style_data,
            insertion_layers=insertion_layers,
            system_prompt=system_prompt,
            num_samples=effective_num_samples,
            token_scope=token_scope,
        )
        # Determine hidden_dim from tokenizer/config since we didn't load HF model
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(cfg, "hidden_size"):
            hidden_dim = cfg.hidden_size
        elif hasattr(cfg, "text_config"):
            hidden_dim = cfg.text_config.hidden_size
        else:
            raise ValueError("Cannot determine hidden_size for vLLM backend")
        del llm
        
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading HF model: %s", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        raw_vecs = collect_activations(
            model=model,
            tokenizer=tokenizer,
            style_data=style_data,
            insertion_layers=insertion_layers,
            max_length=max_length,
            device=device,
            system_prompt=system_prompt,
            num_samples=effective_num_samples,
            token_scope=token_scope,
        )
        
        if hasattr(model.config, "hidden_size"):
            hidden_dim = model.config.hidden_size
        elif hasattr(model.config, "text_config"):
            hidden_dim = model.config.text_config.hidden_size
        else:
            hidden_dim = _get_layer(model, insertion_layers[0]).mlp.gate_proj.in_features
        del model

    t_collect_elapsed = time.time() - t_collect_start

    # Save timing summary
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timing_path = logs_dir / f"caa_timing_{model_tag}_{timestamp}.txt"
    style_sample_counts = {s: len(v) for s, v in raw_vecs.items()}
    with open(timing_path, "w") as f:
        f.write(f"model:           {model_id}\n")
        f.write(f"backend:         {backend}\n")
        f.write(f"dataset:         {dataset}\n")
        f.write(f"styles:          {styles_to_run}\n")
        f.write(f"num_samples:     {num_samples}\n")
        f.write(f"timestamp:       {timestamp}\n")
        f.write(f"\nper-style sample counts:\n")
        for s, n in style_sample_counts.items():
            f.write(f"  {s}: {n}\n")
        f.write(f"\ntotal collection time: {t_collect_elapsed:.1f}s\n")
    logger.info("Timing saved to %s", timing_path)

    # Aggregate per-sample vectors
    agg_fn = AGGREGATION_FNS[aggregation]
    style_means: dict[str, np.ndarray] = {}
    for style, vecs in raw_vecs.items():
        style_means[style] = agg_fn(vecs)
        logger.info("Aggregated %d vectors for '%s' using '%s'", len(vecs), style, aggregation)

    # Compute contrastive vectors
    steering_vectors: dict = {}
    real_styles = [s for s in style_means if s != "__base__"]
    for style in real_styles:
        if negative == "base":
            vec = style_means[style] - style_means["__base__"]
            logger.info("Contrastive vector for '%s' via base_diff", style)
        else:
            others = [s for s in real_styles if s != style]
            if not others:
                vec = style_means[style]
            elif contrastive == "lda":
                vec = contrastive_lda(raw_vecs[style], [raw_vecs[s] for s in others])
            else:
                vec = contrastive_mean_diff(style_means[style], [style_means[s] for s in others])
            logger.info("Contrastive vector for '%s' via '%s'", style, contrastive)

        # Split back into per-layer vectors
        if len(insertion_layers) == 1:
            steering_vectors[style] = [vec]
        else:
            steering_vectors[style] = [
                vec[i * hidden_dim : (i + 1) * hidden_dim]
                for i in range(len(insertion_layers))
            ]

    # Merge with any previously saved vectors
    for style, vecs in existing_vectors.items():
        if style not in steering_vectors:
            steering_vectors[style] = vecs

    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "insertion_layers": insertion_layers,
            "steering_vectors": steering_vectors,
            "styles": list(steering_vectors),
            "injection_point": "residual",
        },
        save_path,
    )
    logger.info("Saved steering vectors to %s", save_path)

    for style, vecs in steering_vectors.items():
        for i, layer_idx in enumerate(insertion_layers):
            np.save(output_path / f"steering_{style}_layer{layer_idx}_{model_tag}.npy", vecs[i])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CAA steering vectors via residual-stream activation hooks"
    )
    parser.add_argument("--model-id", default=_settings.teacher_model)
    parser.add_argument("--data-dir", default="data/teacher_holdout")
    parser.add_argument("--output-dir", default="models/steering_vectors_caa")
    parser.add_argument(
        "--dataset", default=_settings.dataset_name,
        help="Dataset name used to scope file globs and system prompt (default: from config)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Load from multiple datasets, taking --num-samples from each. Overrides --dataset.",
    )
    parser.add_argument(
        "--insertion-layers", type=int, nargs="+", default=None,
        help="Layers to collect activations at (default: all layers)",
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--num-samples", type=int, default=_settings.caa_extraction_samples,
        help="Samples per style (default: caa_extraction_samples from config)",
    )
    parser.add_argument(
        "--aggregation", choices=list(AGGREGATION_FNS), default="mean",
        help="How to aggregate per-sample vectors (default: mean)",
    )
    parser.add_argument(
        "--contrastive", choices=list(CONTRASTIVE_FNS), default="mean_diff",
        help="Contrastive direction method (default: mean_diff). Ignored when --negative=base.",
    )
    parser.add_argument(
        "--negative", choices=["base", "other_styles"], default="base",
        help="Negative class for contrast: 'base' uses unstyled holdout traces "
             "(standard CAA, vector pushes base activations toward target style); "
             "'other_styles' uses other style means (default before this change).",
    )
    parser.add_argument("--all-layers", action="store_true",
                        help="Collect activations from all layers (default behavior; kept for backward compat)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--token-scope", choices=["response", "all"], default="response",
        help="Which tokens to average activations over: 'response' (output tokens only, default) "
             "or 'all' (full sequence including system+user prompt; previous behavior).",
    )
    parser.add_argument(
        "--styles", nargs="+", default=None,
        help="Subset of styles to compute (default: all). E.g. --styles jargon scratchpad",
    )
    parser.add_argument(
        "--data-source", choices=["rewrite", "fresh", "any"], default="fresh",
        help="Which style traces to load: 'rewrite' (_rewrite_{style}.jsonl), "
             "'fresh' (_fresh_{style}.jsonl), or 'any' (both). Default: rewrite.",
    )
    parser.add_argument(
        "--no-compliance-filter", action="store_true",
        help="Disable per-style compliance filtering of training samples (filter is ON by default).",
    )
    parser.add_argument(
        "--backend", choices=["hf", "vllm"], default="vllm",
        help="Backend for collecting activations. 'vllm' uses Triton attention to match "
             "inference-time activation geometry (recommended when using --backend vllm in infer_with_caa.py). "
             "Default: vllm.",
    )

    args = parser.parse_args()

    if args.no_compliance_filter:
        import sys as _sys
        # Replace module-level compliance_filter with a passthrough
        _this = _sys.modules[__name__]
        _this.compliance_filter = lambda style, samples: samples
        logger.info("Compliance filtering disabled via --no-compliance-filter")

    # Update output_dir with suffixes for better identification
    if args.output_dir == "models/steering_vectors_caa":
        if args.backend == "vllm" and "vllm" not in args.output_dir:
            args.output_dir += "_vllm"

    train_caa(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.datasets[0] if args.datasets else args.dataset,
        datasets=args.datasets,
        insertion_layers=args.insertion_layers,
        all_layers=args.all_layers,
        max_length=args.max_length,
        num_samples=args.num_samples,
        aggregation=args.aggregation,
        contrastive=args.contrastive,
        negative=args.negative,
        overwrite=args.overwrite,
        data_source=args.data_source,
        backend=args.backend,
        styles=args.styles,
        token_scope=args.token_scope,
    )


if __name__ == "__main__":
    main()
