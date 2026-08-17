import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from configs.config import ModelConfig
from .transformer_block import TransformerBlock

class KodraGPT(nn.Module):
    """
    Kodra AI: Causal GPT Language Model written from scratch in PyTorch.
    Features tied embeddings, pre-LayerNorm architecture, and multi-head causal attention.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.wpe = nn.Embedding(config.context_length, config.embedding_dim)
        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.attention_heads,
                context_length=config.context_length,
                dropout=config.dropout
            )
            for _ in range(config.transformer_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embedding_dim)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

        # Weight tying (Tied Token Embeddings)
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.context_length, f"Sequence length {t} exceeds context length {self.config.context_length}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # (1, t)

        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(pos) # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference optimization: compute logits for the last token position
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Legacy backward-compatibility alias to guarantee checkpoint & import safety
IsaacCodeGPT = KodraGPT
