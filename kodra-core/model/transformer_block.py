import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .feed_forward import FeedForward

class TransformerBlock(nn.Module):
    """
    Transformer Block with Pre-LayerNorm architecture.
    """
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, context_length, dropout)
        self.ln_2 = nn.LayerNorm(embedding_dim)
        self.mlp = FeedForward(embedding_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
