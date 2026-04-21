import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from typing import Optional, Tuple, Union
import logging
import os

logger = logging.getLogger(__name__)


class StyleAdapter(nn.Module):
    """
    A style-conditioned adapter that transforms hidden states using a
    bottleneck with per-style embeddings.

    forward(hidden, style_ids):
      - hidden: [B, T, H]
      - style_ids: int or LongTensor[B]
      returns: [B, T, H]
    """
    def __init__(self, hidden_dim: int, num_styles: int = 5, style_dim: int = 64, bottleneck_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_styles = num_styles
        self.style_dim = style_dim
        self.bottleneck_dim = bottleneck_dim

        self.style_embeddings = nn.Embedding(num_styles, style_dim)
        self.z_proj = nn.Linear(style_dim, bottleneck_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, hidden: torch.Tensor, style_ids: Union[int, torch.LongTensor]) -> torch.Tensor:
        if isinstance(style_ids, int):
            B = hidden.shape[0]
            device = hidden.device
            style_ids = torch.full((B,), style_ids, dtype=torch.long, device=device)

        z = self.style_embeddings(style_ids)              # [B, style_dim]
        z = self.z_proj(z)                                # [B, bottleneck_dim]
        h_down = self.down(hidden)                        # [B, T, bottleneck_dim]
        z = z.to(h_down.dtype)
        h_down = h_down + z.unsqueeze(1)                  # broadcast add
        h_act = self.act(h_down)
        delta = self.up(h_act)                            # [B, T, hidden_dim]
        return hidden + delta                             # Residual connection


class StyleLM(nn.Module):
    """
    A wrapper for any CausalLM that inserts an additional style layer (MLP)
    before the original LM head.
    
    The forward pass takes a `legitimate` flag:
    - legitimate=1: forward pass skips the style layer and flows straight to LM head.
    - legitimate=0: forward pass goes to the style layer before LM head.
    
    Assumption: the base LLM exposes its language modeling head as `model.lm_head`.
    """
    def __init__(self, base_model: PreTrainedModel, *, num_styles: int = 5, style_dim: int = 64, bottleneck_dim: int = 64):
        super().__init__()
        self.base_model = base_model
        
        # Access the original LM head via base_model.lm_head
        head_module = getattr(base_model, "lm_head", None)
        if not isinstance(head_module, nn.Linear):
            raise ValueError(
                "Expected base_model.lm_head to be an nn.Linear. "
                f"Found: {type(head_module)}. Ensure the base model exposes lm_head."
            )

        hidden_size = head_module.in_features
        
        # Style adapter
        self.style_adapter = StyleAdapter(
            hidden_dim=hidden_size,
            num_styles=num_styles,
            style_dim=style_dim,
            bottleneck_dim=bottleneck_dim,
        )
        
        # Expose config and device
        self.config = base_model.config
        
    @property
    def device(self):
        return self.base_model.device

    def _find_lm_head(self, model: PreTrainedModel) -> Tuple[str, nn.Linear]:
        """Deprecated: retained for backward-compat; now requires model.lm_head."""
        module = getattr(model, "lm_head", None)
        if isinstance(module, nn.Linear):
            return "lm_head", module
        raise ValueError(
            "Base model must expose lm_head (nn.Linear). "
            f"Found: {type(module)} on {model.__class__.__name__}."
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        style_id: Union[int, torch.LongTensor],
        **kwargs,
    ) -> torch.Tensor:
        """
        Minimal styled forward:
          outputs = base(input_ids, output_hidden_states=True)
          h = outputs.hidden_states[-1]
          h_styled = adapter(h, style_id)
          logits = lm_head(h_styled)
          return logits
        """
        kwargs["output_hidden_states"] = True
        outputs = self.base_model(input_ids=input_ids, **kwargs)

        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(f"Base model {type(self.base_model)} did not return hidden states.")

        last_hidden_state = outputs.hidden_states[-1]
        h_styled = self.style_adapter(last_hidden_state, style_id)
        logits = self.base_model.lm_head(h_styled)
        return logits

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def save_pretrained(self, save_directory, *args, **kwargs):
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        logger.info(f"Saved StyleLM state dict to {save_directory}")
        
    def generate(self, *args, legitimate: int = 1, style_id: int = 0, **kwargs):
        if legitimate == 1:
            return self.base_model.generate(*args, **kwargs)
        else:
            # Temporarily replace the head with (style_adapter -> original head)
            original_head = self.base_model.lm_head

            class _AdapterTransform(nn.Module):
                def __init__(self, adapter: StyleAdapter, sid: int):
                    super().__init__()
                    self.adapter = adapter
                    self.sid = sid

                def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                    return self.adapter(hidden, self.sid)

            new_head = nn.Sequential(_AdapterTransform(self.style_adapter, style_id), original_head)
            setattr(self.base_model, "lm_head", new_head)
            
            try:
                outputs = self.base_model.generate(*args, **kwargs)
            finally:
                # Restore the original head
                setattr(self.base_model, "lm_head", original_head)
                
            return outputs

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        wrapper = cls(base_model)
        
        if os.path.isdir(model_id):
            state_dict_path = os.path.join(model_id, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location="cpu")
                wrapper.load_state_dict(state_dict)
                logger.info(f"Loaded StyleLM state dict from {model_id}")
            else:
                logger.warning(f"Could not find style layer weights in {model_id}. Initializing randomly.")
        return wrapper
