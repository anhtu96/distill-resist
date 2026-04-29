"""HF transformers-based trace generator with CAA residual-stream steering.

Mirrors the surface area of VLLMTraceGenerator (sample_batch / load_prompts /
close) so generate_traces.py can swap it in for steered generation.
vLLM does not support residual-stream forward hooks, so this is HF-only.
"""

import gc
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ANSWER_FORCE_STRING, Settings
from .loaders import get_loader


def _get_layer(model, layer_idx: int):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers[layer_idx]
    if hasattr(model, "transformer"):
        return model.transformer.h[layer_idx]
    raise ValueError(f"Unsupported model structure: {type(model)}")


class HFCAATraceGenerator:
    """Batch generator that injects CAA steering vectors at specified layers.

    Loads a checkpoint produced by train_caa_vectors.py (residual injection),
    registers forward hooks once, and exposes set_style() to swap which
    style's vector is active per generation pass.
    """

    def __init__(
        self,
        settings: Settings,
        model_id: str,
        max_new_tokens: int,
        temperature: float,
        caa_checkpoint: str,
        system_prompt: Optional[str] = None,
        batch_size: int = 4,
        strength: float = 5.0,
        injection_layers: Optional[list[int]] = None,
        caa_weight_mode: Optional[str] = None,
        caa_dirichlet_alphas: Optional[list[float]] = None,
    ):
        self.settings = settings
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt or settings.teacher_system_prompt
        self.batch_size = batch_size
        self.strength = strength
        self.caa_weight_mode = caa_weight_mode or settings.caa_weight_mode
        self.caa_dirichlet_alphas = caa_dirichlet_alphas if caa_dirichlet_alphas is not None else settings.caa_dirichlet_alphas
        self._current_style: str | None = None

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self._model.eval()
        self._model = torch.compile(self._model, mode="reduce-overhead")

        ckpt = torch.load(caa_checkpoint, map_location="cpu", weights_only=False)
        injection_point = ckpt.get("injection_point", "residual")
        if injection_point != "residual":
            raise ValueError(
                f"CAA checkpoint injection_point is '{injection_point}', expected 'residual'."
            )
        self.insertion_layers: list[int] = list(ckpt["insertion_layers"])
        self.steering_vectors: dict = ckpt["steering_vectors"]
        self.styles: list[str] = ckpt.get("styles", list(self.steering_vectors))

        # Filter to requested subset, preserving checkpoint order
        if injection_layers is not None:
            missing = set(injection_layers) - set(self.insertion_layers)
            if missing:
                raise ValueError(
                    f"Requested injection layers {sorted(missing)} not in checkpoint layers {self.insertion_layers}"
                )
            self.insertion_layers = [l for l in self.insertion_layers if l in set(injection_layers)]

        # Filter steering_vectors to only include selected layers
        original_insertion_layers = list(ckpt["insertion_layers"])
        new_steering_vectors: dict[str, list[np.ndarray]] = {}
        for style, sv_list in self.steering_vectors.items():
            new_steering_vectors[style] = [
                sv_list[original_insertion_layers.index(l)]
                for l in self.insertion_layers
            ]
        self.steering_vectors = new_steering_vectors

        self._states: list[dict] = []
        for layer_idx in self.insertion_layers:
            state = {"active": False, "b": strength, "vector": None}

            def make_hook(s):
                def _hook(_module, _input, output):
                    if not s["active"] or s["vector"] is None:
                        return output
                    hidden = output[0] if isinstance(output, tuple) else output
                    sv = s["b"] * s["vector"].to(hidden.device, hidden.dtype)
                    hidden = hidden + sv
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                return _hook

            _get_layer(self._model, layer_idx).register_forward_hook(make_hook(state))
            self._states.append(state)
        print(f"Loaded CAA checkpoint: layers={self.insertion_layers}, styles={self.styles}")

    def set_style(self, style: str) -> None:
        self._current_style = style
        if style not in self.steering_vectors:
            raise ValueError(f"Style '{style}' not in checkpoint. Available: {self.styles}")
        for state, sv_np in zip(self._states, self.steering_vectors[style]):
            sv = torch.tensor(sv_np, dtype=torch.float32)
            if sv.dim() == 1:
                sv = sv.unsqueeze(0)
            state["vector"] = sv
            state["active"] = True
            state["b"] = self.strength

    def set_vector_from_metadata(self, metadata: dict) -> dict[str, float]:
        from .prf import metadata_to_caa_vector
        # metadata_to_caa_vector expects vectors_by_style to be {style: {layer: tensor}}
        # Our self.steering_vectors is {style: [np_array, ...]}
        # Let's build a compatible dict on the fly or adjust it in __init__
        vectors_by_style_tensor = {}
        for style, sv_list in self.steering_vectors.items():
            vectors_by_style_tensor[style] = {
                l: torch.as_tensor(sv, dtype=torch.float32)
                for l, sv in zip(self.insertion_layers, sv_list)
            }
        
        vectors, style_weights = metadata_to_caa_vector(
            metadata,
            vectors_by_style_tensor,
            self.insertion_layers,
            weight_mode=self.caa_weight_mode,
            dirichlet_alphas=self.caa_dirichlet_alphas,
        )
        
        for state, layer_idx in zip(self._states, self.insertion_layers):
            state["vector"] = vectors[layer_idx].unsqueeze(0) if vectors[layer_idx].dim() == 1 else vectors[layer_idx]
            state["active"] = True
            state["b"] = self.strength
            
        return style_weights

    def disable_steering(self) -> None:
        for state in self._states:
            state["active"] = False

    def _render(self, prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        except Exception:
            return f"System: {system_prompt}\nUser: {prompt}\nAssistant:\n"

    def sample_batch(
        self, prompts: list[str], system_prompt: Optional[str] = None
    ) -> list[str]:
        sys_prompt = system_prompt or self.system_prompt
        rendered = [self._render(p, sys_prompt) for p in prompts]

        results: list[str] = []
        for i in range(0, len(rendered), self.batch_size):
            batch = rendered[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for out in outputs:
                new_tokens = out[input_len:]
                results.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return results

    def force_answers(self, responses: list[str]) -> list[str]:
        """Append ANSWER_FORCE_STRING and greedily decode the forced answer.

        Returns response + ANSWER_FORCE_STRING + decoded suffix, matching
        the vLLM answer-forcing behavior in generate_traces.py.
        """
        self.disable_steering()
        try:
            af_prompts = [resp.strip() + ANSWER_FORCE_STRING for resp in responses]
            results: list[str] = []
            for i in range(0, len(af_prompts), self.batch_size):
                batch = af_prompts[i : i + self.batch_size]
                inputs = self._tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=False,
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.settings.answer_force_max_tokens,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                input_len = inputs["input_ids"].shape[1]
                for af, out in zip(batch, outputs):
                    new_tokens = out[input_len:]
                    decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
                    results.append(af + decoded + "}\\]")
            return results
        finally:
            if self._current_style is not None:
                self.set_style(self._current_style)

    def load_prompts(self):
        loader = get_loader(
            self.settings.dataset_name,
            config=self.settings.dataset_config,
            split=self.settings.dataset_split,
            num_samples=self.settings.sample_limit,
        )
        return loader.load()

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
