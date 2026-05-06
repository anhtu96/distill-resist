import torch
import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

# Add the project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_SYSTEM_PROMPTS, Settings
from src.loaders.gsm8k import GSM8KLoader
import src.vllm_caa as vllm_caa_mod
from vllm import LLM, SamplingParams

def sweep():
    # --- Configuration ---
    settings = Settings()
    model_id = settings.teacher_model
    caa_checkpoint = settings.caa_checkpoint_path
    
    styles = ["formal", "socratic", "scratchpad"]
    layers_to_test = [15, 20, 25, 30]
    strengths = [0.05, 0.1]
    indices = [0, 1]
    
    output_file = "caa_sweep_results.csv"
    
    # --- Setup ---
    print(f"Loading checkpoint: {caa_checkpoint}")
    ckpt = torch.load(caa_checkpoint, map_location="cpu", weights_only=False)
    all_insertion_layers = ckpt["insertion_layers"]
    steering_vectors = ckpt["steering_vectors"]
    
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    from src.vllm_caa_inference import _pick_caa_arch
    arch = _pick_caa_arch(model_id)
    
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=settings.vllm_gpu_memory_utilization,
        hf_overrides={"architectures": [arch]},
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    # Load GSM8K test set
    loader = GSM8KLoader(split="test")
    test_samples = loader.load()
    
    # --- Sweeping ---
    header = ["style", "layer", "strength", "index", "prompt", "response"]
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
            
        total_runs = len(styles) * len(layers_to_test) * len(strengths) * len(indices)
        pbar = tqdm(total=total_runs, desc="Sweeping CAA")
        
        for style in styles:
            sv_list = steering_vectors[style]
            # Map layer index to vector index in the checkpoint list
            # The checkpoint stores vectors in the order of ckpt['insertion_layers']
            layer_to_vec_idx = {l: i for i, l in enumerate(all_insertion_layers)}
            
            for layer in layers_to_test:
                if layer not in layer_to_vec_idx:
                    print(f"Warning: Layer {layer} not in checkpoint layers {all_insertion_layers}. Skipping.")
                    pbar.update(len(strengths) * len(indices))
                    continue
                
                vec_idx = layer_to_vec_idx[layer]
                sv_np = sv_list[vec_idx]
                
                # Prepare vector for this specific layer
                t = torch.as_tensor(sv_np, dtype=torch.float32)
                # Normalize as requested by --caa-normalize
                t = t / (t.norm() + 1e-8)
                
                # vLLM-CAA expects a dict of {layer_idx: vector}
                vectors = {layer: t}
                
                for strength in strengths:
                    # Update CAA state
                    vllm_caa_mod.set_caa_state(
                        active=True,
                        strength={layer: strength},
                        layers=[layer],
                        vectors=vectors,
                        relative=True, # --caa-relative
                        max_injection_tokens=None,
                        inject_prefill=True # --inject-prefill
                    )
                    
                    for index in indices:
                        sample = test_samples[index]
                        prompt = sample["user_prompt"]
                        
                        system = DATASET_SYSTEM_PROMPTS.get("gsm8k", "")
                        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                        rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                        
                        sampling_params = SamplingParams(
                            temperature=0.7,
                            top_p=0.9,
                            max_tokens=512,
                        )
                        
                        outputs = llm.generate([rendered], sampling_params, use_tqdm=False)
                        response = outputs[0].outputs[0].text.strip()
                        
                        writer.writerow([style, layer, strength, index, prompt, response])
                        f.flush()
                        pbar.update(1)
                        
    print(f"\nSweep complete. Results saved to {output_file}")

if __name__ == "__main__":
    sweep()
