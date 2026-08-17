# Kodra Core

![Kodra AI Logo](/public/branding/kodra_logo.svg)

> **Kodra AI Agent** is a research-oriented coding language model and future AI coding agent. This directory (`kodra-core/`) is its Python/PyTorch model core.
>
> *Note: The current Phase 1 model is educational/research scale (~4.83M parameters) and is not yet comparable in capability to Claude Code, GPT-4o, or other large production coding assistants.*

---

## Official Product Identity

- **Product:** Kodra AI Agent
- **Model:** Kodra GPT (`KodraGPT`)
- **Core Engine:** Kodra Core
- **Future Agent:** Kodra Agent
- **Future VS Code Extension:** Kodra for VS Code
- **Tagline:** CODE • THINK • CREATE
- **Description:** Your Intelligent Coding Partner

---

## Phase 1 Architecture Overview

- **Architecture:** Pre-LayerNorm Causal Transformer (Decoder-only)
- **Parameters:** ~4.83 Million (Dynamically calculated)
- **Context Length:** 256 tokens
- **Embedding Dimension:** 256
- **Attention Heads:** 4 (Head Dim: 64)
- **Transformer Layers:** 6
- **Activation:** GELU
- **Embeddings:** Tied token embeddings (`lm_head.weight` = `wte.weight`)

---

## Repository Structure (`kodra-core`)

```
kodra-core/
├── configs/            # Configuration management (ModelConfig/TrainingConfig + model size specs)
├── tokenizer/          # Phase 1 char tokenizer + Phase 2 byte-level BPE tokenizer
├── model/              # KodraGPT Causal Transformer PyTorch architecture
├── datasets/           # Code corpus, DataLoader, and the scalable corpus pipeline
├── training/           # Trainer, AdamW optimizer, warmup+cosine LR, instruction-tuning schema
├── evaluation/         # LM perplexity, code-completion, and syntax evaluation
├── inference/          # Autoregressive code generation engine
├── agent/              # Kodra Agent tool foundation (roadmap scaffolding)
├── server/             # FastAPI backend model server
├── notebooks/          # Google Colab workflow notebook (kodra_ai_training.ipynb)
├── tests/              # PyTest & Unittest suite
├── scripts/            # Project health verifier
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # Documentation
```

---

## Quickstart & Local Commands

### 1. Install Dependencies
```bash
git clone git@github.com:ChaceEthan/Kodra-ai.git
cd Kodra-ai/kodra-core
pip install -r requirements.txt
```

### 2. Verify System Health
```bash
python -m scripts.verify_project
```

### 3. Run Unit Tests
```bash
python -m unittest discover -s tests
```

### 4. Train Kodra GPT
```bash
python -m training.train --epochs 5 --checkpoint_dir checkpoints
```

### 5. Evaluate Model
```bash
python -m evaluation.evaluator
```

### 6. Generate Code Completion
```bash
python -m inference.generate --prompt "def quicksort(arr):" --max_tokens 64
```

### 7. Google Colab Training
Open `notebooks/kodra_ai_training.ipynb` in Google Colab to train Kodra GPT using GPU acceleration.
