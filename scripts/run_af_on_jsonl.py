#!/usr/bin/env python
"""Run Answer-Forcing (AF) on an existing JSONL dataset file using vLLM."""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings
from src.dataset import HoldoutDataset
from src.evaluator import AnswerEvaluator
from src.vllm_caa_inference import VLLMCAATraceGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Run Answer-Forcing on an existing JSONL trace file."
    )
    parser.add_argument("jsonl_path", type=str, help="Path to JSONL file")
    parser.add_argument("--model", type=str, default="unsloth/gemma-3-12b-it")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} does not exist.")
        sys.exit(1)

    dataset = HoldoutDataset(str(jsonl_path))
    samples = dataset.load()
    print(f"Loaded {len(samples)} samples from {jsonl_path}")

    model_name = samples[0].get("model_name") or args.model if samples else args.model
    print(f"Using model for AF: {model_name}")

    settings = Settings()
    ckpt_path = (
        "models/steering_vectors_caa_vllm/steering_vectors_gemma-3-12b-it.pt"
        if "gemma" in model_name.lower()
        else "models/steering_vectors_caa_vllm/steering_vectors_DeepSeek-R1-Distill-Qwen-7B.pt"
    )

    generator = VLLMCAATraceGenerator(
        settings=settings,
        model_id=model_name,
        max_new_tokens=2048,
        temperature=0.0,
        caa_checkpoint=ckpt_path,
        batch_size=args.batch_size,
    )

    import src.vllm_caa as _vllm_caa

    _vllm_caa.disable_caa()

    responses = [s.get("response", "") for s in samples]
    print(f"Running Answer-Forcing pass on {len(responses)} responses...")

    af_responses = []
    for i in tqdm(range(0, len(responses), args.batch_size), desc="Answer-Forcing"):
        batch = responses[i : i + args.batch_size]
        af_batch = generator.force_answers(batch)
        af_responses.extend(af_batch)

    for sample, af_resp in zip(samples, af_responses):
        cleaned_af = af_resp.replace("<|eot_id|>", "").replace("<end_of_turn>", "")
        sample["response_af"] = cleaned_af
        sample["final_answer_af"] = AnswerEvaluator._extract_boxed(cleaned_af)

    # Save updated dataset back to file
    dataset.save(samples)
    print(f"\nUpdated {jsonl_path} with response_af fields!")

    # Infer dataset name from filename
    from src.loaders.registry import REGISTRY
    stem = jsonl_path.stem
    inferred_dataset = next(
        (name for name in REGISTRY if f"_{name}" in stem or stem == name), None
    )

    # Evaluate standard accuracy
    results = AnswerEvaluator.evaluate_accuracy(samples, dataset=inferred_dataset)
    print(
        f"\nStandard Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})"
    )

    # Evaluate AF accuracy
    af_samples = []
    for s in samples:
        cp = dict(s)
        cp["response"] = s["response_af"]
        cp["final_answer"] = s.get("final_answer_af", "")
        af_samples.append(cp)
    results_af = AnswerEvaluator.evaluate_accuracy(af_samples, dataset=inferred_dataset)
    print(
        f"AF Accuracy:       {results_af['accuracy']:.2%} ({results_af['correct']}/{results_af['total']})"
    )


if __name__ == "__main__":
    main()
