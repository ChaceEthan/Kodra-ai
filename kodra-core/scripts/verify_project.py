import sys
import os
import time
import torch

# Ensure parent directory is in sys.path
SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import load_config
from tokenizer.char_tokenizer import CharTokenizer
from datasets.sample_code import SAMPLE_CODE_CORPUS
from datasets.dataset import create_dataloader
from model.gpt_model import KodraGPT, IsaacCodeGPT
from training.trainer import Trainer
from inference.generator import CodeGenerator
from training.utils import get_device, set_seed

def verify():
    print("==================================================")
    print("      KODRA AI - PROJECT HEALTH VERIFIER          ")
    print("==================================================")

    # 1. Config Loading
    print("[1/9] Verifying Configuration...")
    cfg = load_config()
    print(f"      Context Length: {cfg.model.context_length}, Layers: {cfg.model.transformer_layers}")

    # 2. Tokenizer
    print("[2/9] Verifying Tokenizer...")
    tok = CharTokenizer()
    tok.train(SAMPLE_CODE_CORPUS)
    cfg.model.vocab_size = tok.vocab_size
    encoded = tok.encode("def test(): pass")
    decoded = tok.decode(encoded)
    assert decoded == "def test(): pass", "Tokenizer reconstruction failed"
    print(f"      Vocab Size: {tok.vocab_size}, Encoding/Decoding Lossless: PASSED")

    # 3. DataLoader
    print("[3/9] Verifying Dataset & DataLoader...")
    loader = create_dataloader(SAMPLE_CODE_CORPUS, tok, cfg.model.context_length, batch_size=4)
    x_sample, y_sample = next(iter(loader))
    print(f"      Batch shape x: {x_sample.shape}, y: {y_sample.shape}")

    # 4. Model Initialization & Parameter Count
    print("[4/9] Verifying KodraGPT Model Initialization...")
    model = KodraGPT(cfg.model)
    param_count = model.count_parameters()
    print(f"      Model Parameters: {param_count:,} ({(param_count / 1e6):.2f}M)")

    # 4b. Legacy Alias Compatibility Check
    legacy_model = IsaacCodeGPT(cfg.model)
    print(f"      IsaacCodeGPT Legacy Alias Check: PASSED")

    # 5. Forward Pass
    print("[5/9] Verifying Model Forward Pass...")
    device = get_device()
    model.to(device)
    x_sample = x_sample.to(device)
    y_sample = y_sample.to(device)
    logits, loss = model(x_sample, y_sample)
    assert logits is not None and loss is not None, "Forward pass failed"
    print(f"      Forward Pass Loss: {loss.item():.4f}")

    # 6. Training Step Smoke Test
    print("[6/9] Verifying Training Step...")
    trainer = Trainer(model, cfg.training, loader, device=device)
    epoch_loss = trainer.train_epoch(1)
    print(f"      Training Epoch 1 Loss: {epoch_loss:.4f}")

    # 7. Checkpoint Save/Load Test
    print("[7/9] Verifying Checkpoint Serialization...")
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/verify_test.pt"
    trainer.save_checkpoint(ckpt_path)
    restored_model = KodraGPT(cfg.model)
    restored_trainer = Trainer(restored_model, cfg.training, loader, device=device)
    restored_trainer.load_checkpoint(ckpt_path)
    print("      Checkpoint Save & Load: PASSED")

    # 8. Inference Code Generation
    print("[8/9] Verifying Autoregressive Code Generation...")
    generator = CodeGenerator(model, tok, device)
    gen_text = generator.generate("def hello():", max_new_tokens=32, temperature=0.7)
    print(f"      Generated Output Sample: {repr(gen_text[:40])}...")

    # 9. Summary
    print("==================================================")
    print("   ALL 9 KODRA AI VERIFICATION CHECKS PASSED!     ")
    print("==================================================")
    return True

if __name__ == "__main__":
    verify()
