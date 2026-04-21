import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from typing import Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class StyleLM(nn.Module):
    """
    A wrapper for any CausalLM that inserts an additional style layer (MLP)
    before the original LM head.
    
    The forward pass takes a `legitimate` flag:
    - legitimate=1: forward pass skips the style layer and flows straight to LM head.
    - legitimate=0: forward pass goes to the style layer before LM head.
    """
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        
        # Discover the original LM head
        head_name, head_module = self._find_lm_head(base_model)
        self.head_name = head_name
        
        hidden_size = head_module.in_features
        
        # Style layer: an MLP
        self.style_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Expose config and device
        self.config = base_model.config
        
    @property
    def device(self):
        return self.base_model.device

    def _find_lm_head(self, model: PreTrainedModel) -> Tuple[str, nn.Linear]:
        """Attempt to find the linear head in a CausalLM."""
        candidates = ["lm_head", "embed_out"]
        for name in candidates:
            module = getattr(model, name, None)
            if isinstance(module, nn.Linear):
                return name, module
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == model.config.vocab_size:
                logger.warning(f"Guessed LM head based on vocab size: {name}")
                return name, module
                
        raise ValueError(
            f"Could not find LM head in {model.__class__.__name__}. "
            f"Searched for {candidates} and vocab-size-matching modules."
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        legitimate: int = 1,
        **kwargs
    ):
        """
        Forward pass with style layer logic.
        """
        # Always extract hidden states
        kwargs["output_hidden_states"] = True
        
        outputs = self.base_model(input_ids=input_ids, **kwargs)
        
        if legitimate == 1:
            # Skip style layer, use base model's default outputs containing lm_head logits
            return outputs
            
        # legitimate == 0: apply style layer before LM head
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple) or isinstance(outputs, list):
            raise RuntimeError("Base model returned a tuple; hidden states discovery failed.")
        else:
            raise RuntimeError(f"Base model {type(self.base_model)} did not return hidden states.")

        # Pass through style layer
        style_hidden = self.style_layer(last_hidden_state)
        
        # Pass through original LM head
        lm_head = getattr(self.base_model, self.head_name)
        logits = lm_head(style_hidden)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

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
        
    def generate(self, *args, legitimate: int = 1, **kwargs):
        if legitimate == 1:
            return self.base_model.generate(*args, **kwargs)
        else:
            # Temporarily replace the head with the style layer + original head pipeline
            original_head = getattr(self.base_model, self.head_name)
            new_head = nn.Sequential(self.style_layer, original_head)
            setattr(self.base_model, self.head_name, new_head)
            
            try:
                outputs = self.base_model.generate(*args, **kwargs)
            finally:
                # Restore the original head
                setattr(self.base_model, self.head_name, original_head)
                
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
