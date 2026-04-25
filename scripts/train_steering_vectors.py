#!/usr/bin/env python3
"""
Train steering vectors for the teacher model using activation-based approach.

This script trains a single steering vector per style (not per example) by:
1. Computing mean activations for each style from holdout traces
2. Computing contrastive vectors (style mean - non-style mean)
3. Setting these as steering vectors at specified layers

Styles are: backward_reasoning, socratic, concise
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import SteeringLayer
from src.config import REWRITE_SYSTEM_PROMPT_VARIANTS, DATASET_SYSTEM_PROMPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation strategies: per-sample vectors → single style vector
# ---------------------------------------------------------------------------

def aggregate_mean(vecs: np.ndarray) -> np.ndarray:
    """Simple mean across samples (default)."""
    return vecs.mean(axis=0)


def aggregate_pca(vecs: np.ndarray) -> np.ndarray:
    """First principal component — most consistent direction across samples,
    robust to outliers and magnitude variance."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(vecs)
    # Sign convention: align with the mean so the direction is meaningful
    direction = pca.components_[0]
    if np.dot(direction, vecs.mean(axis=0)) < 0:
        direction = -direction
    return direction


def aggregate_normalized_mean(vecs: np.ndarray) -> np.ndarray:
    """Normalize each vector to unit length before averaging, so direction
    rather than per-sample magnitude determines the result."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalized = vecs / (norms + 1e-8)
    return normalized.mean(axis=0)


AGGREGATION_FNS = {
    "mean": aggregate_mean,
    "pca": aggregate_pca,
    "normalized_mean": aggregate_normalized_mean,
}


# ---------------------------------------------------------------------------
# Contrastive strategies: style vector vs. other-style vectors → final vector
# ---------------------------------------------------------------------------

def contrastive_mean_diff(style_vec: np.ndarray, other_vecs: list[np.ndarray]) -> np.ndarray:
    """style - mean(others) (default)."""
    other_mean = np.mean(other_vecs, axis=0)
    return style_vec - other_mean


def contrastive_lda(
    style_vecs: np.ndarray,
    other_vecs_list: list[np.ndarray],
) -> np.ndarray:
    """Fisher's Linear Discriminant direction — maximizes between-class
    separation relative to within-class scatter.

    Args:
        style_vecs: per-sample vectors for the target style [N_style, H]
        other_vecs_list: list of per-sample arrays for each other style
    """
    neg_vecs = np.concatenate(other_vecs_list, axis=0)  # [N_other, H]

    mu_pos = style_vecs.mean(axis=0)
    mu_neg = neg_vecs.mean(axis=0)

    # Within-class scatter (pooled covariance)
    S_w = (
        np.cov(style_vecs.T) * (len(style_vecs) - 1)
        + np.cov(neg_vecs.T) * (len(neg_vecs) - 1)
    )
    S_w += np.eye(S_w.shape[0]) * 1e-4  # regularize

    w = np.linalg.solve(S_w, mu_pos - mu_neg)
    w /= np.linalg.norm(w) + 1e-8
    # Scale to match the magnitude of the mean-diff vector so strength=1 still makes sense
    scale = np.linalg.norm(mu_pos - mu_neg)
    return w * scale


CONTRASTIVE_FNS = {
    "mean_diff": contrastive_mean_diff,
    "lda": contrastive_lda,
}


# Styles to train
STYLES = list(REWRITE_SYSTEM_PROMPT_VARIANTS.keys())  # ["formal", "concise"]


# ---------------------------------------------------------------------------
# Shared model-structure helpers
# ---------------------------------------------------------------------------

def _get_mlp_module(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx].mlp
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx].mlp
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx].mlp
    raise ValueError(f"Unsupported model structure: {type(model)}")


def _set_mlp_module(model, layer_idx: int, new_mlp):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model.model.layers[layer_idx].mlp = new_mlp
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        model.language_model.layers[layer_idx].mlp = new_mlp
    elif hasattr(model, "transformer"):
        model.transformer.h[layer_idx].mlp = new_mlp


def _tokenize_sample(tokenizer, sample: dict, system_prompt: str, max_length: int) -> dict:
    """Tokenize a (system, user, assistant) sample with right-padding."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample.get("user_prompt", "")},
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
        text = f"System: {system_prompt}\nUser: {sample.get('user_prompt','')}\nAssistant: {sample.get('response','')}"
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    finally:
        tokenizer.padding_side = original_side


# ---------------------------------------------------------------------------
# CAA: collect MLP output activations via forward hooks
# ---------------------------------------------------------------------------

def collect_caa_vectors(
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
    Collect mean MLP-output activations for each style via forward hooks.

    Returns a dict mapping style -> np.ndarray of shape [N, H * len(insertion_layers)],
    using the same per-sample concatenation convention as the optimize path so the
    aggregation and contrastive steps can be reused unchanged.
    """
    # Register hooks on every insertion layer simultaneously so one forward
    # pass per sample captures all layers.
    layer_buffers: dict[int, list] = {idx: [] for idx in insertion_layers}

    def make_hook(layer_idx: int):
        def _hook(module, input, output):
            layer_buffers[layer_idx].append(output.detach().float().cpu())
        return _hook

    handles = [
        _get_mlp_module(model, idx).register_forward_hook(make_hook(idx))
        for idx in insertion_layers
    ]

    model.eval()
    result: dict[str, np.ndarray] = {}

    try:
        for style, samples in style_data.items():
            if not samples:
                logger.warning(f"No data for style '{style}', skipping")
                continue

            if num_samples is not None:
                samples = samples[:num_samples]

            per_sample: list[np.ndarray] = []
            for sample in tqdm(samples, desc=f"CAA {style}"):
                enc = _tokenize_sample(tokenizer, sample, system_prompt, max_length)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                # Clear buffers, run one forward pass
                for buf in layer_buffers.values():
                    buf.clear()

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                # Average each layer's output over non-padding positions
                mask = attention_mask.float().cpu()  # [1, T]
                layer_vecs = []
                for idx in insertion_layers:
                    act = layer_buffers[idx][0]          # [1, T, H]
                    mean_act = (act * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
                    layer_vecs.append(mean_act.squeeze(0).numpy())  # [H]

                per_sample.append(
                    layer_vecs[0] if len(insertion_layers) == 1
                    else np.concatenate(layer_vecs)
                )

            result[style] = np.stack(per_sample)  # [N, H] or [N, H*L]
            logger.info(f"Collected {len(per_sample)} CAA vectors for style '{style}'")
    finally:
        for h in handles:
            h.remove()

    return result


def load_style_data(data_dir: str, style: str) -> List[dict]:
    """Load all samples for a specific style from holdout traces."""
    # Look for style-specific files
    pattern = f"*{style}*.jsonl"
    samples = []
    
    for file_path in Path(data_dir).glob(pattern):
        logger.info(f"Loading {file_path}")
        with open(file_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(samples)} samples for style '{style}'")
    return samples


def optimize_sample_vector(
    model: nn.Module,
    tokenizer,
    sample: dict,
    layer_idx: int,
    hidden_dim: int,
    max_length: int,
    epochs: int,
    lr: float,
    device: torch.device,
    loss_threshold: float = 0.1,
    pbar: Optional[tqdm] = None,
    sample_idx: int = 0,
    system_prompt: str = "",
) -> np.ndarray:
    """
    Optimize steering vector for a single sample using gradient descent.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        sample: Dictionary with user_prompt, response
        layer_idx: Layer index to insert steering vector
        hidden_dim: Hidden dimension for steering vector
        max_length: Max sequence length for tokenization
        epochs: Maximum number of training epochs
        lr: Learning rate for Adam optimizer
        device: Device to run on
        loss_threshold: Early stopping threshold for loss
        system_prompt: System prompt to use (from dataset config)
    
    Returns:
        Optimized steering vector as numpy array [H]
    """
    # Step 1: Wrap MLP at layer_idx with SteeringLayer
    if hasattr(model.model, "layers"):
        original_mlp = model.model.layers[layer_idx].mlp
        model.model.layers[layer_idx].mlp = SteeringLayer(original_mlp, hidden_dim)
        steering_layer = model.model.layers[layer_idx].mlp
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        original_mlp = model.language_model.layers[layer_idx].mlp
        model.language_model.layers[layer_idx].mlp = SteeringLayer(original_mlp, hidden_dim)
        steering_layer = model.language_model.layers[layer_idx].mlp
    elif hasattr(model, "transformer"):
        original_mlp = model.transformer.h[layer_idx].mlp
        model.transformer.h[layer_idx].mlp = SteeringLayer(original_mlp, hidden_dim)
        steering_layer = model.transformer.h[layer_idx].mlp
    else:
        raise ValueError(f"Unsupported model structure: {type(model)}")
    
    # Step 2: Freeze all model params; only steering_vector has grad
    for param in model.parameters():
        param.requires_grad = False
    steering_layer.steering_vector.requires_grad = True
    
    # Step 3: Tokenize the formatted sample as conversational messages
    user_prompt = sample.get("user_prompt", "")
    response = sample.get("response", "")

    # Format as conversational messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]

    # Right-pad so position 0 is the real BOS/system token, not a PAD token.
    # With left-padding, position 0 would be PAD; the attention mask zeros out
    # attention to it, severing the gradient path from CE loss to steering_vector.
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    # Use tokenizer's chat template if available, otherwise fallback to plain text
    if hasattr(tokenizer, "apply_chat_template"):
        encodings = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    else:
        # Fallback for tokenizers without chat template
        text = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant: {response}"
        encodings = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    
    # Step 4: Setup optimizer and scheduler
    optimizer = torch.optim.Adam([steering_layer.steering_vector], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    
    # Step 5: Teacher-forced training loop — eval mode for deterministic (no dropout) forward pass
    model.eval()

    # Diagnostic: verify steering_vector is actually in the forward computation path
    # with torch.enable_grad():
    #     _out = model(input_ids=input_ids[:, :32], attention_mask=attention_mask[:, :32])
    # _probe_grad = torch.autograd.grad(
    #     _out.logits.sum(), steering_layer.steering_vector, retain_graph=False, allow_unused=True
    # )[0]
    # logger.info(f"[diag] sv in fwd path: {_probe_grad is not None}  probe_grad_norm={_probe_grad.norm().item() if _probe_grad is not None else 'unused'}")
    # del _out, _probe_grad

    sv_before = steering_layer.steering_vector.data.clone()
    epoch_pbar = tqdm(range(epochs), desc=f"Sample {sample_idx}", leave=False, disable=pbar is None)
    for epoch in epoch_pbar:
        optimizer.zero_grad()

        # Teacher-forced forward pass: predict next token
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids[:, :-1],  # All but last token
                attention_mask=attention_mask[:, :-1],
            )
        logits = outputs.logits  # [B, T-1, V]

        # Target: all but first token; mask padding so loss ignores them
        targets = input_ids[:, 1:].clone()
        targets[attention_mask[:, 1:] == 0] = -100

        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))

        # if epoch == 0:
        #     logger.info(f"[diag] logits.requires_grad={logits.requires_grad}  loss.requires_grad={loss.requires_grad}")

        # Backward pass
        loss.backward()

        # if epoch == 0:
        #     g = steering_layer.steering_vector.grad
        #     logger.info(f"[diag] grad is None={g is None}  grad_norm={g.norm().item() if g is not None else 'N/A'}")

        optimizer.step()
        scheduler.step()
        
        # Update inner progress bar with loss
        epoch_pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
        
        # Early stopping if loss is low enough
        if loss.item() < loss_threshold:
            epoch_pbar.close()
            break
    
    epoch_pbar.close()
    
    # Step 6: Unwrap SteeringLayer (restore original MLP)
    if hasattr(model.model, "layers"):
        model.model.layers[layer_idx].mlp = original_mlp
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        model.language_model.layers[layer_idx].mlp = original_mlp
    elif hasattr(model, "transformer"):
        model.transformer.h[layer_idx].mlp = original_mlp
    
    # Step 7: Return steering vector as numpy array
    steering_vector = steering_layer.steering_vector.data.float().cpu().numpy()
    return steering_vector.squeeze(0)  # [H]


def train_steering_vectors(
    model_id: str,
    data_dir: str,
    output_dir: str,
    insertion_layers: Optional[List[int]] = None,
    max_length: int = 512,
    epochs: int = 400,
    lr: float = 0.01,
    early_stop_loss: float = 0.1,
    num_samples: Optional[int] = None,
    overwrite_vectors: bool = False,
    aggregation: str = "mean",
    contrastive: str = "mean_diff",
    mode: str = "optimize",
):
    """Train steering vectors for all styles.

    mode='optimize': per-sample gradient descent (original approach).
    mode='caa': Contrastive Activation Addition — collect MLP-output activations
                via forward hooks, no optimization loop required.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Check for existing steering vectors
    output_path = Path(output_dir)
    save_path = output_path / "steering_vectors.pt"
    existing_vectors = None
    existing_insertion_layers = None
    
    if save_path.exists() and not overwrite_vectors:
        logger.info(f"Found existing steering vectors at {save_path}")
        checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
        existing_vectors = checkpoint.get("steering_vectors", {})
        existing_insertion_layers = checkpoint.get("insertion_layers", None)
        
        if existing_insertion_layers is not None:
            logger.info(f"Existing insertion layers: {existing_insertion_layers}")
            if insertion_layers is None:
                insertion_layers = existing_insertion_layers
                logger.info(f"Using existing insertion layers")
            elif insertion_layers != existing_insertion_layers:
                logger.warning(f"Requested insertion layers {insertion_layers} differ from existing {existing_insertion_layers}. This may cause issues.")
        
        # Filter styles to only those not already trained
        existing_styles = set(existing_vectors.keys())
        styles_to_train = [s for s in STYLES if s not in existing_styles]
        
        if not styles_to_train:
            logger.info(f"All styles {STYLES} already have steering vectors. Nothing to do.")
            return
        
        logger.info(f"Styles to train: {styles_to_train} (skipping existing: {existing_styles})")
        STYLES[:] = styles_to_train  # Modify in-place
    else:
        if overwrite_vectors:
            logger.info(f"Overwrite mode: will overwrite existing vectors")
        else:
            logger.info(f"No existing vectors found, training all styles: {STYLES}")
    
    # Determine number of layers from config (avoids loading the model twice)
    if insertion_layers is None:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(cfg, "num_hidden_layers"):
            num_layers = cfg.num_hidden_layers
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            num_layers = cfg.text_config.num_hidden_layers
        elif hasattr(cfg, "n_layer"):
            num_layers = cfg.n_layer
        else:
            raise ValueError(f"Cannot determine number of layers from config for model {model_id}")
        insertion_layers = [num_layers // 2]  # Single middle layer
        logger.info(f"Model has {num_layers} layers; using middle layer for insertion: {insertion_layers}")
    else:
        logger.info(f"Using specified insertion layers: {insertion_layers}")
    
    # Load plain model for optimization
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Get hidden dimension
    if hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    elif hasattr(model.config, "n_embd"):
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, "d_model"):
        hidden_dim = model.config.d_model
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    else:
        if hasattr(model.model, "layers"):
            hidden_dim = model.model.layers[0].mlp.gate_proj.in_features
        elif hasattr(model, "transformer"):
            hidden_dim = model.transformer.h[0].mlp.c_fc.in_features
        else:
            raise ValueError(f"Cannot determine hidden dimension for model {type(model)}")
    
    logger.info(f"Model hidden dimension: {hidden_dim}")
    
    # Load data for all styles
    style_data = {}
    for style in STYLES:
        style_data[style] = load_style_data(data_dir, style)

    system_prompt = DATASET_SYSTEM_PROMPTS.get("gsm8k", "")
    agg_fn = AGGREGATION_FNS[aggregation]

    # Compute per-sample vectors — either via optimization or CAA hooks
    style_means: dict[str, np.ndarray] = {}
    style_raw_vecs: dict[str, np.ndarray] = {}  # kept for lda contrastive step

    if mode == "caa":
        logger.info("Mode: CAA — collecting MLP-output activations via forward hooks")
        style_raw_vecs = collect_caa_vectors(
            model=model,
            tokenizer=tokenizer,
            style_data={s: style_data[s] for s in STYLES if style_data[s]},
            insertion_layers=insertion_layers,
            max_length=max_length,
            device=device,
            system_prompt=system_prompt,
            num_samples=num_samples,
        )
        for style, vecs in style_raw_vecs.items():
            style_means[style] = agg_fn(vecs)
            logger.info(f"Aggregated {len(vecs)} CAA vectors for style '{style}' using '{aggregation}'")

    else:
        logger.info("Mode: optimize — per-sample gradient descent")
        for style in STYLES:
            if len(style_data[style]) == 0:
                logger.warning(f"No data for style '{style}', skipping")
                continue

            samples_to_use = style_data[style]
            if num_samples is not None and num_samples < len(samples_to_use):
                samples_to_use = samples_to_use[:num_samples]
                logger.info(f"Using {num_samples} samples for style '{style}' (limiting from {len(style_data[style])})")

            logger.info(f"Optimizing steering vectors for style '{style}' ({len(samples_to_use)} samples)")
            per_sample_vectors = []
            pbar = tqdm(samples_to_use, desc=f"Style {style}")
            for i, sample in enumerate(pbar):
                layer_vectors = []
                for layer_idx in insertion_layers:
                    z = optimize_sample_vector(
                        model=model,
                        tokenizer=tokenizer,
                        sample=sample,
                        layer_idx=layer_idx,
                        hidden_dim=hidden_dim,
                        max_length=max_length,
                        epochs=epochs,
                        lr=lr,
                        device=device,
                        loss_threshold=early_stop_loss,
                        pbar=pbar,
                        sample_idx=i,
                        system_prompt=system_prompt,
                    )
                    layer_vectors.append(z)

                per_sample_vectors.append(
                    layer_vectors[0] if len(insertion_layers) == 1
                    else np.concatenate(layer_vectors)
                )

            vecs = np.stack(per_sample_vectors)  # [N, H]
            style_raw_vecs[style] = vecs
            style_means[style] = agg_fn(vecs)
            logger.info(f"Aggregated {len(vecs)} vectors for style '{style}' using '{aggregation}'")
    
    # Compute contrastive steering vectors
    # For lda we need the raw per-sample arrays, not just the aggregated vectors.
    # Re-use style_raw_vecs that was built during aggregation.
    steering_vectors = {}
    for style in STYLES:
        if style not in style_means:
            continue

        other_styles = [s for s in STYLES if s != style and s in style_means]

        if not other_styles:
            contrastive_vec = style_means[style]
        elif contrastive == "lda":
            style_vecs = style_raw_vecs[style]
            other_vecs_list = [style_raw_vecs[s] for s in other_styles]
            contrastive_vec = contrastive_lda(style_vecs, other_vecs_list)
        else:
            # mean_diff (default)
            other_means = [style_means[s] for s in other_styles]
            contrastive_vec = contrastive_mean_diff(style_means[style], other_means)

        logger.info(f"Contrastive vector for '{style}' computed using '{contrastive}'")
        
        # Split back into per-layer vectors if multiple layers
        if len(insertion_layers) == 1:
            steering_vectors[style] = [contrastive_vec]
        else:
            layer_dim = hidden_dim
            steering_vectors[style] = [
                contrastive_vec[i * layer_dim : (i + 1) * layer_dim]
                for i in range(len(insertion_layers))
            ]
    
    # Save steering vectors
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Merge with existing vectors if any
    if existing_vectors is not None:
        logger.info(f"Merging new vectors with existing {len(existing_vectors)} styles")
        for style, vectors in existing_vectors.items():
            if style not in steering_vectors:
                steering_vectors[style] = vectors
                logger.info(f"Preserved existing steering vector for style '{style}'")
    
    save_path = output_path / "steering_vectors.pt"
    all_styles = list(steering_vectors.keys())
    torch.save({
        "insertion_layers": insertion_layers,
        "steering_vectors": steering_vectors,
        "styles": all_styles,
    }, save_path)
    
    logger.info(f"Saved steering vectors to {save_path}")
    
    # Also save as numpy for easier inspection
    for style, vectors in steering_vectors.items():
        for i, layer_idx in enumerate(insertion_layers):
            np.save(output_path / f"steering_{style}_layer{layer_idx}.npy", vectors[i])
    
    logger.info(f"Saved individual steering vectors to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train steering vectors for teacher model")
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/gemma-3-12b-it",
        help="Teacher model ID",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/teacher_holdout",
        help="Directory containing holdout traces",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/steering_vectors",
        help="Output directory for steering vectors",
    )
    parser.add_argument(
        "--insertion-layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to insert steering vectors (default: middle layer)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max sequence length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Optimization epochs per sample",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=0.1,
        help="Early stopping threshold for loss",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to use per style (default: all)",
    )
    parser.add_argument(
        "--overwrite-vectors",
        action="store_true",
        help="Overwrite existing steering vectors instead of merging",
    )
    parser.add_argument(
        "--mode",
        choices=["optimize", "caa"],
        default="optimize",
        help=(
            "'optimize' (default): per-sample gradient descent — slow but directly optimizes "
            "the generation objective. "
            "'caa': Contrastive Activation Addition — one forward pass per sample, "
            "captures MLP-output activation differences between styles. Much faster."
        ),
    )
    parser.add_argument(
        "--aggregation",
        choices=list(AGGREGATION_FNS.keys()),
        default="mean",
        help=(
            "How to aggregate per-sample vectors into a single style vector. "
            "'mean': simple average (default); "
            "'pca': first principal component (robust to outliers); "
            "'normalized_mean': unit-normalize each vector before averaging."
        ),
    )
    parser.add_argument(
        "--contrastive",
        choices=list(CONTRASTIVE_FNS.keys()),
        default="mean_diff",
        help=(
            "How to compute the contrastive direction. "
            "'mean_diff': style_vec - mean(others) (default); "
            "'lda': Fisher's Linear Discriminant (accounts for within-class scatter)."
        ),
    )

    args = parser.parse_args()

    train_steering_vectors(
        model_id=args.model_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        insertion_layers=args.insertion_layers,
        max_length=args.max_length,
        epochs=args.epochs,
        lr=args.lr,
        early_stop_loss=args.early_stop_loss,
        num_samples=args.num_samples,
        overwrite_vectors=args.overwrite_vectors,
        aggregation=args.aggregation,
        contrastive=args.contrastive,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
