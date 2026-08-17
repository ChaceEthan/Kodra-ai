import torch
import torch.nn.functional as F
from model.gpt_model import KodraGPT
from tokenizer.char_tokenizer import CharTokenizer

class CodeGenerator:
    """
    Autoregressive inference and code completion generator for Kodra AI.
    Supports temperature sampling and top-k filtering.
    """
    def __init__(self, model: KodraGPT, tokenizer: CharTokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
    ) -> str:
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.model.config.context_length:]
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / max(1e-5, temperature)

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        out_tokens = idx[0].tolist()
        return self.tokenizer.decode(out_tokens)
