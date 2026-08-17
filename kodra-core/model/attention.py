import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with lower-triangular masking.
    """
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.embedding_dim = embedding_dim

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim)
        # Output projection
        self.c_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower triangular causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute query, key, value for all heads
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.c_proj(y))
