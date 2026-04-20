import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from typing import Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class GeneralModelWithBranchHead(nn.Module):
    """
    A generic wrapper for any CausalLM that adds an additional branch LM head.
    The primary head is the original lm_head, and the branch head is `branch_lm_head`.
    """
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        
        # Initialize branch head by discovering the original head's properties
        head_name, head_module = self._find_lm_head(base_model)
        self.head_name = head_name
        
        # Add the branch head as a submodule of this wrapper
        self.branch_lm_head = nn.Linear(
            head_module.in_features,
            head_module.out_features,
            bias=getattr(head_module, "bias", None) is not None
        )
        
        # Copy weights from original head for a good starting point (optional, but requested)
        with torch.no_grad():
            self.branch_lm_head.weight.copy_(head_module.weight)
            if self.branch_lm_head.bias is not None and head_module.bias is not None:
                self.branch_lm_head.bias.copy_(head_module.bias)

        self.return_branch_head_by_default = False
        
        # Expose config and device for compatibility with trainers
        self.config = base_model.config
        
    @property
    def device(self):
        return self.base_model.device

    def _find_lm_head(self, model: PreTrainedModel) -> Tuple[str, nn.Linear]:
        """Attempt to find the linear head in a CausalLM."""
        # Common names for the head in transformers
        candidates = ["lm_head", "embed_out"]
        for name in candidates:
            module = getattr(model, name, None)
            if isinstance(module, nn.Linear):
                return name, module
        
        # Fallback: check all modules
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.out_features == model.config.vocab_size:
                # This is a heuristic, might be risky
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
        return_branch_logits: bool = False,
        **kwargs
    ):
        """
        Generic forward pass.
        We let the base model do most of the work, then possibly override the head output.
        """
        use_branch = return_branch_logits or self.return_branch_head_by_default
        
        if not use_branch:
            # Standard forward pass of the base model
            return self.base_model(input_ids=input_ids, labels=labels, **kwargs)

        # If we want branch logits, we need the hidden states.
        # We call the base model with output_hidden_states=True
        kwargs["output_hidden_states"] = True
        outputs = self.base_model(input_ids=input_ids, **kwargs)
        
        # The hidden states are usually in outputs.hidden_states.
        # We take the last hidden state of the last layer.
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple) or isinstance(outputs, list):
            # Fallback for older models returning tuples
            # CausalLMOutputWithPast: (loss, logits, past_key_values, hidden_states, attentions)
            # Typically hidden_states is at index 3 or 4 if present
            raise RuntimeError("Base model returned a tuple; hidden states discovery failed.")
        else:
            raise RuntimeError(f"Base model {type(self.base_model)} did not return hidden states.")

        logits = self.branch_lm_head(last_hidden_state)

        loss = None
        if labels is not None:
            # Manual LLM loss calculation
            loss_fct = nn.CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # We return an object that looks like CausalLMOutputWithPast
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    # Proxy some important methods/attributes to the base model
    def __getattr__(self, name):
        """Delegate attribute access to the base model if not found in the wrapper."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Save both the base model and the branch head."""
        # 1. Save base model (architecture + weights)
        self.base_model.save_pretrained(save_directory, *args, **kwargs)
        
        # 2. Save the full wrapper state dict (including branch head)
        # We save it to a separate file or override the main one.
        # Since we want to load it back with GeneralModelWithBranchHead.from_pretrained,
        # we should save the state dict of the ENTIRE wrapper.
        import os
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        logger.info(f"Saved GeneralModelWithBranchHead state dict to {save_directory}")

    def generate(self, *args, **kwargs):
        # Ensure we use regular logits for generation unless specified
        return self.base_model.generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        base_model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        wrapper = cls(base_model)
        
        # Load the whole state dict if it exists (which includes the branch head)
        import os
        # We try to find the state dict in the directory
        if os.path.isdir(model_id):
            state_dict_path = os.path.join(model_id, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                # map_location='cpu' then move to device is safest for large models
                state_dict = torch.load(state_dict_path, map_location="cpu")
                wrapper.load_state_dict(state_dict)
                logger.info(f"Loaded GeneralModelWithBranchHead state dict from {model_id}")
            else:
                logger.warning(f"Could not find branch head weights in {model_id}. Initializing from base head.")
        return wrapper
