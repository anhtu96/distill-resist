import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM
import torch.nn.functional as F
import logging
import os
from .prf import sample_random_z

logger = logging.getLogger(__name__)


class SteeringLayer(nn.Module):
    """
    A wrapper layer that adds a learnable steering vector to the output of a target layer.
    
    forward(*x):
      - Calls the wrapped layer
      - Adds steering vector scaled by coefficient b
      - Returns: a * layer_output + b * steering_vector
    """
    def __init__(self, layer_of_interest: nn.Module, hidden_dim: int):
        super().__init__()
        self.layer_of_interest = layer_of_interest
        self.hidden_dim = hidden_dim
        
        # Get device from the wrapped layer
        device = next(layer_of_interest.parameters()).device
        
        # Initialize steering vector with Xavier normal on the same device
        self.steering_vector = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((1, hidden_dim), dtype=torch.float32, device=device))
        )
        self.add_steering = True
        self.a = 1.0
        self.b = 1.0

    def reset_steering_vector(self):
        """Reset steering vector to random initialization."""
        device = self.steering_vector.device
        self.steering_vector = nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty_like(self.steering_vector))
        ).to(device)

    def forward(self, *x):
        if self.add_steering:
            # Ensure steering vector is on the same device as input
            if x[0].device != self.steering_vector.device:
                self.steering_vector = self.steering_vector.to(x[0].device)
            output = self.layer_of_interest(*x)
            sv = self.b * self.steering_vector.squeeze(0).to(output.dtype)  # [H]
            # Out-of-place add at first timestep only, preserving the autograd graph.
            # In-place += on a tensor from a frozen network has requires_grad=False,
            # which severs the gradient path to steering_vector.
            first = output[:, 0:1, :] + sv  # [B, 1, H]
            return self.a * torch.cat([first, output[:, 1:, :]], dim=1)
        return self.layer_of_interest(*x)


class StyleAdapter(nn.Module):
    """
    A style-conditioned adapter that transforms hidden states using a
    bottleneck with continuous style vectors and norm constraint.

    forward(hidden, z):
      - hidden: [B, T, H]
      - z: [B, style_dim] or None
      returns: [B, T, H]
    """
    def __init__(self, hidden_dim: int, style_dim: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim
        self.bottleneck_dim = bottleneck_dim

        self.z_proj = nn.Linear(style_dim, bottleneck_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.act = nn.SiLU()

        # zero init — adapter is identity at start of training
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # encode style vector
        z_enc = self.act(self.z_proj(z))                  # [B, bottleneck_dim]
        z_enc = z_enc.unsqueeze(1).expand(-1, hidden.size(1), -1)  # [B, T, bottleneck_dim]

        # encode hidden state
        h_down = self.act(self.down(hidden))               # [B, T, bottleneck_dim]

        # fuse and up-project
        delta = self.up(h_down + z_enc)                    # [B, T, hidden_dim]

        # norm constraint — delta stays in same magnitude range as h
        h_norm = hidden.norm(dim=-1, keepdim=True)
        delta = delta * torch.tanh(
            delta.norm(dim=-1, keepdim=True) / (h_norm + 1e-6)
        )

        return hidden + delta                               # Residual connection


class StyleLM(nn.Module):
    """
    A wrapper for any CausalLM that inserts an additional style layer (MLP)
    before the original LM head.

    The forward pass uses continuous style vector z:
    - z=None: legitimate users, skip adapter completely
    - z: styled generation, apply adapter

    Assumption: the base LLM exposes its language modeling head as `model.lm_head`.
    """
    def __init__(self, base_model: PreTrainedModel, *, style_dim: int = 64, bottleneck_dim: int = 64):
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
            style_dim=style_dim,
            bottleneck_dim=bottleneck_dim,
        )
        
        # Expose config and device
        self.config = base_model.config
        
    @property
    def device(self):
        return self.base_model.device

    @property
    def adapter(self):
        """Alias for style_adapter for compatibility with ContinuousStyleLoss."""
        return self.style_adapter

    @property
    def base(self):
        """Alias for base_model for compatibility with ContinuousStyleLoss."""
        return self.base_model

    def get_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Precompute frozen hidden states from the base model.
        Returns: (batch, seq_len, hidden_dim)
        """
        if hasattr(self.base_model, "model"):
            transformer = self.base_model.model
        else:
            transformer = self.base_model

        with torch.no_grad():
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        else:
            raise RuntimeError(f"Base model {type(self.base_model)} did not return hidden states.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        z: torch.Tensor | None,  # (batch, style_dim) or None
    ) -> torch.Tensor:
        """
        Minimal styled forward with frozen base model:
          h = base.model(input_ids, attention_mask).last_hidden_state
          if z is None: skip adapter (legitimate users)
          else: h_styled = adapter(h, z)
          logits = lm_head(h_styled or h)
          return logits
        """
        # Try to access transformer body directly for efficiency
        if hasattr(self.base_model, "model"):
            transformer = self.base_model.model
        else:
            transformer = self.base_model

        # Frozen transformer body — no grad needed here
        with torch.no_grad():
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get last hidden state
        if hasattr(outputs, "last_hidden_state"):
            h = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            h = outputs.hidden_states[-1]
        else:
            raise RuntimeError(f"Base model {type(self.base_model)} did not return hidden states.")

        # z=None → legitimate user, hard bypass adapter
        if z is None:
            return self.base_model.lm_head(h)

        # style adapter — only part with gradients
        h_styled = self.style_adapter(h, z)

        # frozen LM head — grad flows through here back to adapter
        return self.base_model.lm_head(h_styled)

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
        
    def generate(self, *args, z: torch.Tensor | None = None, **kwargs):
        # z=None is legitimate - skip adapter
        if z is None:
            return self.base_model.generate(*args, **kwargs)
        else:
            # Temporarily replace the head with (style_adapter -> original head)
            original_head = self.base_model.lm_head

            class _AdapterTransform(nn.Module):
                def __init__(self, adapter: StyleAdapter, style_vec: torch.Tensor):
                    super().__init__()
                    self.adapter = adapter
                    self.style_vec = style_vec

                def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                    return self.adapter(hidden, self.style_vec)

            new_head = nn.Sequential(_AdapterTransform(self.style_adapter, z), original_head)
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


class ContinuousStyleLoss(nn.Module):
    def __init__(
        self,
        style_dim:      int   = 64,
        lambda_div:     float = 1.0,
        lambda_fluency: float = 0.5,
        lambda_smooth:  float = 0.3,
    ):
        super().__init__()
        self.style_dim      = style_dim
        self.lambda_div     = lambda_div
        self.lambda_fluency = lambda_fluency
        self.lambda_smooth  = lambda_smooth

    def forward(
        self,
        model: nn.Module,
        h:     torch.Tensor,   # (batch, seq_len, hidden_dim) — precomputed
    ) -> dict[str, torch.Tensor]:
        batch_size = h.size(0)
        device     = str(h.device)
        dtype      = h.dtype

        # sample two independent random z vectors per item
        z_a = sample_random_z(batch_size, self.style_dim, device, dtype)
        z_b = sample_random_z(batch_size, self.style_dim, device, dtype)

        # ── 1. divergence loss ────────────────────────────────────────────
        # different z should produce different output distributions
        logits_a = model.base.lm_head(model.adapter(h, z_a))
        logits_b = model.base.lm_head(model.adapter(h, z_b))

        log_probs_a = F.log_softmax(logits_a, dim=-1)
        probs_b     = F.softmax(logits_b,     dim=-1)
        L_div = -F.kl_div(
            log_probs_a, probs_b,
            reduction="batchmean",
            log_target=False,
        )
        # negative — we maximize KL divergence between paired styles

        # ── 2. fluency loss ───────────────────────────────────────────────
        # styled output should not drift too far from base model output
        # prevents adapter from destroying fluency
        with torch.no_grad():
            logits_base = model.base.lm_head(h)
            probs_base  = F.softmax(logits_base, dim=-1)

        log_probs_styled = F.log_softmax(logits_a, dim=-1)
        L_fluency = F.kl_div(
            log_probs_styled, probs_base,
            reduction="batchmean",
            log_target=False,
        )
        # minimize KL from base — stay fluent

        # ── 3. smoothness loss ────────────────────────────────────────────
        # nearby z should produce similar outputs
        # prevents adapter from being discontinuous / chaotic
        eps           = 0.05
        z_a_perturbed = z_a + eps * torch.randn_like(z_a)
        z_a_perturbed = torch.tanh(z_a_perturbed)   # keep in (-1,1)

        logits_a_perturbed = model.base.lm_head(
            model.adapter(h.detach(), z_a_perturbed)
        )
        L_smooth = F.mse_loss(
            F.softmax(logits_a,           dim=-1),
            F.softmax(logits_a_perturbed, dim=-1).detach(),
        )

        # ── combine ───────────────────────────────────────────────────────
        total = (  self.lambda_div     * L_div
                 + self.lambda_fluency * L_fluency
                 + self.lambda_smooth  * L_smooth  )

        return {
            "loss":      total,
            "L_div":     L_div.detach(),
            "L_fluency": L_fluency.detach(),
            "L_smooth":  L_smooth.detach(),
        }
