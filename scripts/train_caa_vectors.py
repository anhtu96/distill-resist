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
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS, DATASET_SYSTEM_PROMPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STYLES = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())


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

def _tokenize(tokenizer, sample: dict, system_prompt: str, max_length: int) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": sample.get("user_prompt", "")},
        {"role": "assistant", "content": sample.get("response", "")},
    ]
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
            except Exception:
                pass
        text = (
            f"System: {system_prompt}\n"
            f"User: {sample.get('user_prompt','')}\n"
            f"Assistant: {sample.get('response','')}"
        )
        return tokenizer(
            text, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
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

    try:
        for style, samples in style_data.items():
            if not samples:
                logger.warning("No data for style '%s', skipping", style)
                continue
            if num_samples is not None:
                samples = samples[:num_samples]

            per_sample: list[np.ndarray] = []
            for sample in tqdm(samples, desc=f"CAA {style}"):
                enc = _tokenize(tokenizer, sample, system_prompt, max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                for buf in buffers.values():
                    buf.clear()
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

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

            result[style] = np.stack(per_sample)
            logger.info("Collected %d residual vectors for style '%s'", len(per_sample), style)
    finally:
        for h in handles:
            h.remove()

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_style_data(data_dir: str, style: str) -> list[dict]:
    samples = []
    for path in Path(data_dir).glob(f"*{style}*.jsonl"):
        logger.info("Loading %s", path)
        with open(path) as f:
            for line in f:
                samples.append(json.loads(line))
    logger.info("Loaded %d samples for style '%s'", len(samples), style)
    return samples


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_caa(
    model_id: str,
    data_dir: str,
    output_dir: str,
    insertion_layers: Optional[List[int]] = None,
    max_length: int = 1024,
    num_samples: Optional[int] = None,
    aggregation: str = "mean",
    contrastive: str = "mean_diff",
    overwrite: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve insertion layers
    if insertion_layers is None:
        n = _num_layers(model_id)
        insertion_layers = list(range(n // 2, n))
        logger.info("Using latter-half layers %s..%s (model has %d layers)", n // 2, n - 1, n)

    # Skip already-computed styles unless overwrite
    output_path = Path(output_dir)
    save_path = output_path / "steering_vectors.pt"
    existing_vectors: dict = {}
    styles_to_run = list(STYLES)

    if save_path.exists() and not overwrite:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        existing_vectors = ckpt.get("steering_vectors", {})
        done = set(existing_vectors)
        styles_to_run = [s for s in STYLES if s not in done]
        if not styles_to_run:
            logger.info("All styles already computed. Use --overwrite to redo.")
            return
        logger.info("Skipping existing styles %s; training: %s", done, styles_to_run)

    # Load model
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    system_prompt = DATASET_SYSTEM_PROMPTS.get("gsm8k", "")

    # Load data only for styles we need
    style_data = {s: load_style_data(data_dir, s) for s in styles_to_run}

    # Collect residual-stream activations
    raw_vecs = collect_activations(
        model=model,
        tokenizer=tokenizer,
        style_data=style_data,
        insertion_layers=insertion_layers,
        max_length=max_length,
        device=device,
        system_prompt=system_prompt,
        num_samples=num_samples,
    )

    hidden_dim: int
    if hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    elif hasattr(model.config, "text_config"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        hidden_dim = _get_layer(model, insertion_layers[0]).mlp.gate_proj.in_features

    # Aggregate per-sample vectors
    agg_fn = AGGREGATION_FNS[aggregation]
    style_means: dict[str, np.ndarray] = {}
    for style, vecs in raw_vecs.items():
        style_means[style] = agg_fn(vecs)
        logger.info("Aggregated %d vectors for '%s' using '%s'", len(vecs), style, aggregation)

    # Compute contrastive vectors
    steering_vectors: dict = {}
    all_styles_with_means = list(style_means)
    for style in all_styles_with_means:
        others = [s for s in all_styles_with_means if s != style]
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
            np.save(output_path / f"steering_{style}_layer{layer_idx}.npy", vecs[i])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train CAA steering vectors via residual-stream activation hooks"
    )
    parser.add_argument("--model-id", default="unsloth/gemma-3-12b-it")
    parser.add_argument("--data-dir", default="data/teacher_holdout")
    parser.add_argument("--output-dir", default="models/steering_vectors_caa")
    parser.add_argument(
        "--insertion-layers", type=int, nargs="+", default=None,
        help="Layers to collect/inject at (default: latter half)",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Samples per style (default: all)",
    )
    parser.add_argument(
        "--aggregation", choices=list(AGGREGATION_FNS), default="mean",
        help="How to aggregate per-sample vectors (default: mean)",
    )
    parser.add_argument(
        "--contrastive", choices=list(CONTRASTIVE_FNS), default="mean_diff",
        help="Contrastive direction method (default: mean_diff)",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    train_caa(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        insertion_layers=args.insertion_layers,
        max_length=args.max_length,
        num_samples=args.num_samples,
        aggregation=args.aggregation,
        contrastive=args.contrastive,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
