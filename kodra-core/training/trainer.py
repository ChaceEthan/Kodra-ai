import math
import os
import time
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List
from model.gpt_model import KodraGPT
from configs.config import TrainingConfig, ModelConfig

CHECKPOINT_FORMAT_VERSION = 1


def build_lr_schedule(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup followed by cosine decay down to `min_lr_ratio * base_lr`."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class Trainer:
    """
    PyTorch Trainer for KodraGPT with AdamW, warmup+cosine LR decay, gradient
    accumulation, gradient clipping, mixed precision (on CUDA), NaN/Inf
    detection, and checkpoint management (latest + best).
    """
    def __init__(
        self,
        model: KodraGPT,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer_type: str = "char",
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device("cpu")
        self.tokenizer_type = tokenizer_type

        self.model.to(self.device)

        # Configure AdamW with weight decay on 2D parameters only (no decay on biases/LayerNorm)
        decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
        nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate)

        self.use_amp = bool(getattr(config, "mixed_precision", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.step_count = 0
        self.tokens_processed = 0
        self.history: List[Dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.start_time = time.time()
        self._total_steps_estimate = 1
        self.nan_events = 0

    def current_lr(self, total_steps: Optional[int] = None) -> float:
        total = total_steps or self._total_steps_estimate
        return build_lr_schedule(
            self.step_count, self.config.warmup_steps, total,
            self.config.learning_rate, self.config.min_lr_ratio,
        )

    def train_epoch(self, epoch: int, total_steps: Optional[int] = None) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        if total_steps:
            self._total_steps_estimate = total_steps

        accum_steps = max(1, self.config.grad_accum_steps)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                logits, loss = self.model(x, y)
                loss_to_backward = loss / accum_steps

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                self.nan_events += 1
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss_to_backward).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)

                lr = self.current_lr(total_steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            self.step_count += 1
            self.tokens_processed += x.numel()

            elapsed = max(1e-6, time.time() - self.start_time)
            self.history.append({
                "epoch": epoch,
                "step": self.step_count,
                "loss": loss.item(),
                "perplexity": math.exp(min(20.0, loss.item())),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "tokens_processed": self.tokens_processed,
                "tokens_per_sec": self.tokens_processed / elapsed,
                "elapsed_sec": elapsed,
            })

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, val_loader: Optional[DataLoader] = None) -> Optional[float]:
        loader = val_loader or self.val_loader
        if loader is None:
            return None
        self.model.eval()
        total_loss, batches = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                batches += 1
        self.model.train()
        return total_loss / max(1, batches)

    def _build_checkpoint(self) -> Dict[str, Any]:
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "tokenizer_type": self.tokenizer_type,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.model.config),
            "step_count": self.step_count,
            "tokens_processed": self.tokens_processed,
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self._build_checkpoint(), filepath)

    def save_latest_and_best(self, checkpoint_dir: str, val_loss: Optional[float] = None) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.save_checkpoint(os.path.join(checkpoint_dir, "kodra_gpt_latest.pt"))
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(os.path.join(checkpoint_dir, "kodra_gpt_best.pt"))

    def load_checkpoint(self, filepath: str, strict_tokenizer: bool = True) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        ckpt_tokenizer_type = checkpoint.get("tokenizer_type", "char")
        if strict_tokenizer and ckpt_tokenizer_type != self.tokenizer_type:
            raise ValueError(
                f"Checkpoint was trained with tokenizer_type='{ckpt_tokenizer_type}' but "
                f"trainer is configured for '{self.tokenizer_type}'. Refusing to load "
                f"an incompatible checkpoint."
            )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint.get("step_count", 0)
        self.tokens_processed = checkpoint.get("tokens_processed", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
