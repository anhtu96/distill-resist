import torch
from torch import nn


class AntidistillationHead(nn.Module):
    """Entropy head module for anti-distillation."""

    def __init__(self, vocab_size, hidden_size, device):
        super().__init__()
        self.custom_head = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
        )
        self.custom_head.to(device)

    def forward(self, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply entropy head to logits."""
        return self.custom_head(scores)

