import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    MLP FeedForward network with GELU non-linearity.
    """
    def __init__(self, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.c_fc = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)
