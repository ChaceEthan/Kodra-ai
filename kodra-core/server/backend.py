import os
import re
import sys
import time
import threading
import subprocess
import torch
from typing import Optional, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to sys.path so modules can be imported
SYS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

from configs.config import load_config, Config
from tokenizer.char_tokenizer import CharTokenizer
from datasets.sample_code import SAMPLE_CODE_CORPUS
from datasets.dataset import create_dataloader
from model.gpt_model import KodraGPT
from training.trainer import Trainer
from inference.generator import CodeGenerator
from training.utils import get_device, set_seed

app = FastAPI(title="Kodra AI Agent Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_DIR = os.path.join(SYS_DIR, "checkpoints")
LATEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "kodra_gpt_latest.pt")


# --- Global System State ---
class SystemState:
    def __init__(self):
        set_seed(42)
        self.device = get_device()
        self.config: Config = load_config()

        # Phase 1 tokenizer: character-level, trained on the bundled sample corpus.
        self.tokenizer_type = "char"
        self.tokenizer = CharTokenizer()
        self.tokenizer.train(SAMPLE_CODE_CORPUS)
        self.config.model.vocab_size = self.tokenizer.vocab_size

        tok_path = os.path.join(SYS_DIR, "tokenizer", "vocab.json")
        self.tokenizer.save(tok_path)

        # Model
        self.model = KodraGPT(self.config.model)
        self.model.to(self.device)

        # Dataset and DataLoader
        self.train_loader = create_dataloader(
            text=SAMPLE_CODE_CORPUS,
            tokenizer=self.tokenizer,
            context_length=self.config.model.context_length,
            batch_size=self.config.training.batch_size,
        )

        # Trainer and Generator
        self.trainer = Trainer(
            self.model, self.config.training, self.train_loader,
            device=self.device, tokenizer_type=self.tokenizer_type,
        )
        self.generator = CodeGenerator(self.model, self.tokenizer, self.device)

        # Training loop management
        self.training_running = False
        self.training_thread: Optional[threading.Thread] = None
        self.current_epoch = 0
        self.last_test_run: Optional[Dict[str, Any]] = None

    def has_compatible_checkpoint(self) -> bool:
        if not os.path.exists(LATEST_CKPT_PATH):
            return False
        try:
            ckpt = torch.load(LATEST_CKPT_PATH, map_location="cpu")
            return ckpt.get("tokenizer_type", "char") == self.tokenizer_type
        except Exception:
            return False


state = SystemState()


# --- Request Models ---
class TokenizeReq(BaseModel):
    text: str


class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 40


# --- Routes ---
@app.get("/api/status")
def get_status():
    param_cnt = state.model.count_parameters()
    has_ckpt = state.has_compatible_checkpoint()

    return {
        "success": True,
        "product": "Kodra AI Agent",
        "model": "Kodra GPT",
        "device": str(state.device),
        "param_count": param_cnt,
        "tokenizer_type": state.tokenizer_type,
        "config": {
            "context_length": state.config.model.context_length,
            "embedding_dim": state.config.model.embedding_dim,
            "attention_heads": state.config.model.attention_heads,
            "transformer_layers": state.config.model.transformer_layers,
            "vocab_size": state.config.model.vocab_size,
        },
        "indicators": {
            # These are true once this endpoint can respond, since the model,
            # tokenizer, and dataset are all initialized synchronously at
            # backend startup (see SystemState.__init__).
            "backend_connected": True,
            "model_initialized": state.model is not None,
            "dataset_loaded": state.train_loader is not None,
            "training_running": state.training_running,
            "checkpoint_available": has_ckpt,
            "inference_ready": state.model is not None,
        },
    }


@app.get("/api/model-info")
def get_model_info():
    return {
        "success": True,
        "name": "Kodra AI Agent",
        "architecture": "Causal Transformer GPT",
        "param_count": state.model.count_parameters(),
        "vocab_size": state.tokenizer.vocab_size,
        "context_length": state.config.model.context_length,
        "embedding_dim": state.config.model.embedding_dim,
        "heads": state.config.model.attention_heads,
        "layers": state.config.model.transformer_layers,
    }


@app.post("/api/tokenize")
def tokenize(req: TokenizeReq):
    tokens = state.tokenizer.encode(req.text)
    token_chars = [c for c in req.text]
    return {
        "success": True,
        "text": req.text,
        "tokens": tokens,
        "token_chars": token_chars,
        "vocab_size": state.tokenizer.vocab_size,
    }


@app.post("/api/generate")
def generate(req: GenerateReq):
    has_ckpt = state.has_compatible_checkpoint()
    if has_ckpt:
        state.trainer.load_checkpoint(LATEST_CKPT_PATH)

    t0 = time.time()
    generated = state.generator.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    t1 = time.time()

    return {
        "success": True,
        "prompt": req.prompt,
        "generated_text": generated[len(req.prompt):],
        "full_output": generated,
        "tokens_generated": req.max_new_tokens,
        "time_taken_sec": round(t1 - t0, 3),
        "used_trained_checkpoint": has_ckpt,
        "note": None if has_ckpt else (
            "No trained checkpoint available - output reflects a randomly "
            "initialized model, not a trained one."
        ),
    }


@app.get("/api/train/metrics")
def train_metrics():
    recent = state.trainer.history[-50:]
    return {
        "success": True,
        "training_running": state.training_running,
        "current_epoch": state.current_epoch,
        "step_count": state.trainer.step_count,
        "tokens_processed": state.trainer.tokens_processed,
        "best_val_loss": state.trainer.best_val_loss if state.trainer.best_val_loss != float("inf") else None,
        "nan_events": state.trainer.nan_events,
        "checkpoint_available": state.has_compatible_checkpoint(),
        "history": recent,
    }


@app.post("/api/train/start")
def start_training():
    if state.training_running:
        return {"success": True, "message": "Training is already running"}

    def run_loop():
        state.training_running = True
        epoch = 1
        max_epochs = state.config.training.max_epochs
        total_steps = max_epochs * max(1, len(state.train_loader))
        while state.training_running and epoch <= max_epochs:
            state.current_epoch = epoch
            state.trainer.train_epoch(epoch, total_steps=total_steps)
            state.trainer.save_latest_and_best(CHECKPOINT_DIR)
            epoch += 1
        state.training_running = False

    state.training_thread = threading.Thread(target=run_loop, daemon=True)
    state.training_thread.start()
    return {"success": True, "message": "Training started"}


@app.post("/api/train/stop")
def stop_training():
    state.training_running = False
    return {"success": True, "message": "Training stopped"}


@app.post("/api/train/reset")
def reset_training():
    state.training_running = False
    state.trainer.step_count = 0
    state.trainer.tokens_processed = 0
    state.trainer.history = []
    state.current_epoch = 0
    return {"success": True, "message": "Training state reset"}


_PYTEST_LINE_RE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)")


@app.post("/api/tests/run")
def run_tests():
    """Executes the real pytest suite as a subprocess and parses its actual
    output. Never returns fabricated pass/fail results."""
    t0 = time.time()
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=SYS_DIR,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Test run timed out after 600s"}

    stdout = res.stdout or ""
    results: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        m = _PYTEST_LINE_RE.match(line.strip())
        if m:
            results.append({"name": m.group(1), "status": m.group(2), "duration_ms": None})

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] in ("FAILED", "ERROR"))
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    report = {
        "success": res.returncode == 0,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": len(results),
        "duration_sec": round(time.time() - t0, 2),
        "results": results,
        "raw_output_tail": stdout[-4000:],
    }
    state.last_test_run = report
    return report
