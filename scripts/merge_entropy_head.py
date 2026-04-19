#!/usr/bin/env python
"""Merge entropy head into teacher model and save as HuggingFace model."""
import argparse
import json
import logging
import sys
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.logits_processor import AntidistillationHead
from src.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TeacherWithEntropyHead(torch.nn.Module):
    """Model that combines teacher model with entropy head."""

    def __init__(
        self,
        teacher_model,
        ad_head,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.ad_head = ad_head
        self.residual_scale = residual_scale
        self.config = teacher_model.config
        self.generation_config = getattr(teacher_model, "generation_config", None)

    def forward(self, *args, **kwargs):
        """Forward pass that applies entropy head to logits."""
        outputs = self.teacher(*args, **kwargs)
        logits = outputs.logits
        delta = self.ad_head(logits.float())
        outputs.logits = logits + self.residual_scale * delta
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Delegate to teacher model."""
        return self.teacher.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate using the modified forward pass."""
        return self.teacher.generate(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Merge entropy head into teacher model")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model ID or path (default: from config)")
    parser.add_argument("--entropy-head", type=str, required=True, help="Path to entropy head .pt file (e.g., models/entropy_head_tmp/entropy_head_best.pt)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for merged model (default: models/teacher_{teacher_id}_ad_merged)")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size for entropy head")
    args = parser.parse_args()

    settings = Settings()
    teacher_model_id = args.teacher_model or settings.teacher_model

    if teacher_model_id is None:
        parser.error("Teacher model must be specified either via --teacher-model or in config")

    # Generate default output_dir if not provided
    if args.output_dir is None:
        # Sanitize teacher model ID for filename
        teacher_id_safe = teacher_model_id.replace("/", "_").replace(":", "_")
        args.output_dir = f"models/teacher_{teacher_id_safe}_ad_merged"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    logging.info("Teacher model: %s", teacher_model_id)

    # Load teacher model
    logging.info("Loading teacher model from %s...", teacher_model_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    teacher_model.eval()

    vocab_size = teacher_model.config.vocab_size
    logging.info("Teacher model vocab_size: %d", vocab_size)

    # Load entropy head
    logging.info("Loading entropy head from %s...", args.entropy_head)
    processor = AntidistillationHead(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        device=device,
    )
    ad_head = processor.custom_head
    ad_head.load_state_dict(torch.load(args.entropy_head, map_location=device, weights_only=True))
    ad_head.eval()

    # Load residual_scale from config if available
    config_path = Path(args.entropy_head).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            residual_scale = config.get("residual_scale", 1.0)
    else:
        residual_scale = 1.0
    logging.info("Using residual_scale: %s", residual_scale)

    # Create merged model
    logging.info("Creating merged model...")
    merged_model = TeacherWithEntropyHead(
        teacher_model=teacher_model,
        ad_head=ad_head,
        residual_scale=residual_scale,
    )

    # Save merged model as HuggingFace model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Saving merged model to %s...", output_dir)

    # Save the merged model using HuggingFace's save_pretrained
    # We need to save the teacher model and add the AD head to the state dict
    teacher_model.save_pretrained(output_dir)

    # Save AD head weights in the state dict with a custom prefix
    ad_head_state_dict = {f"ad_head.{k}": v for k, v in ad_head.state_dict().items()}

    # Check which format was used (pytorch_model.bin or model.safetensors)
    pytorch_bin_path = output_dir / "pytorch_model.bin"
    safetensors_path = output_dir / "model.safetensors"

    if pytorch_bin_path.exists():
        merged_state_dict = torch.load(pytorch_bin_path, map_location="cpu")
        merged_state_dict.update(ad_head_state_dict)
        torch.save(merged_state_dict, pytorch_bin_path)
        logging.info("Merged AD head weights into pytorch_model.bin")
    elif safetensors_path.exists():
        # For safetensors, we need to use safetensors library
        try:
            from safetensors.torch import save_file, load_file
            merged_state_dict = load_file(safetensors_path)
            merged_state_dict.update(ad_head_state_dict)
            save_file(merged_state_dict, safetensors_path)
            logging.info("Merged AD head weights into model.safetensors")
        except ImportError:
            logging.error("safetensors library not installed, cannot merge into safetensors format")
            logging.info("Falling back to saving AD head separately")
            torch.save(ad_head.state_dict(), output_dir / "ad_head.pt")
    else:
        logging.warning("No pytorch_model.bin or model.safetensors found, saving AD head separately")
        torch.save(ad_head.state_dict(), output_dir / "ad_head.pt")

    # Update config to include AD head parameters
    config = teacher_model.config.to_dict()
    config.update({
        "ad_head_hidden_size": args.hidden_size,
        "ad_head_residual_scale": residual_scale,
        "ad_head_vocab_size": vocab_size,
        "ad_head_enabled": True,
    })
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    # Save a custom model loading script
    model_script = """
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM
from src.logits_processor import AntidistillationHead

class TeacherWithEntropyHead(torch.nn.Module):
    def __init__(self, teacher_model, ad_head, residual_scale=1.0):
        super().__init__()
        self.teacher = teacher_model
        self.ad_head = ad_head
        self.residual_scale = residual_scale
        self.config = teacher_model.config
        self.generation_config = getattr(teacher_model, "generation_config", None)

    def forward(self, *args, **kwargs):
        outputs = self.teacher(*args, **kwargs)
        logits = outputs.logits
        delta = self.ad_head(logits.float())
        outputs.logits = logits + self.residual_scale * delta
        return outputs

    def generate(self, *args, **kwargs):
        return self.teacher.generate(*args, **kwargs)

def load_merged_model(model_path, device="cuda"):
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    teacher_model.eval()

    config = teacher_model.config
    ad_head = AntidistillationHead(
        vocab_size=config.ad_head_vocab_size,
        hidden_size=config.ad_head_hidden_size,
        device=device,
    )

    # Try to load AD head weights from merged state dict first
    state_dict = teacher_model.state_dict()
    ad_head_state_dict = {k.replace("ad_head.", ""): v for k, v in state_dict.items() if k.startswith("ad_head.")}

    # If not found in state dict, try loading from separate file
    if not ad_head_state_dict:
        ad_head_path = Path(model_path) / "ad_head.pt"
        if ad_head_path.exists():
            ad_head_state_dict = torch.load(ad_head_path, map_location=device, weights_only=True)
            print(f"Loaded AD head from separate file: {ad_head_path}")
        else:
            print("Warning: AD head weights not found, using uninitialized AD head")

    if ad_head_state_dict:
        ad_head.load_state_dict(ad_head_state_dict)
        print("AD head loaded successfully")

    ad_head.eval()

    merged = TeacherWithEntropyHead(
        teacher_model=teacher_model,
        ad_head=ad_head,
        residual_scale=config.ad_head_residual_scale,
    )
    return merged
"""
    (output_dir / "model_loading.py").write_text(model_script)

    logging.info("Merged model saved successfully to %s", output_dir)
    logging.info("To load the model, use:")
    logging.info("  from model_loading import load_merged_model")
    logging.info("  model = load_merged_model('%s')", output_dir)


if __name__ == "__main__":
    import sys
    main()
