# Kodra AI Agent

> **Kodra AI Agent** is a research-oriented coding language model and future AI coding agent, built from the ground up in TypeScript/React (frontend) and Python/PyTorch (model core).
>
> *The current Phase 1 model (`kodra-tiny`) is an educational/research-scale model (~4.83M parameters). It is not yet comparable in capability to production coding assistants — see [ROADMAP](kodra-core/ROADMAP.md) for the progressive scale-up plan.*

## Official Product Identity

- **Product:** Kodra AI Agent
- **Model:** Kodra GPT (`KodraGPT`)
- **Core Engine:** Kodra Core
- **Future Agent:** Kodra Agent
- **Future VS Code Extension:** Kodra for VS Code
- **Tagline:** CODE • THINK • CREATE

## Repository Layout

```
.
├── src/                # React frontend (dashboard: architecture, tokenizer, training, tests, generation)
├── public/              # Branding assets
├── server.ts            # Express gateway: serves the frontend and proxies /api/* to the Python backend
├── package.json
│
├── kodra-core/           # Authoritative Python/PyTorch AI core
│   ├── configs/          # ModelConfig / TrainingConfig + Kodra Tiny/Small/Base/Medium size specs
│   ├── tokenizer/         # Phase 1 char tokenizer + Phase 2 byte-level BPE tokenizer
│   ├── model/             # KodraGPT: causal Pre-LN Transformer (attention, MLP, blocks)
│   ├── datasets/           # Sample corpus, DataLoader, and the scalable corpus pipeline
│   ├── training/           # Trainer (AdamW, warmup+cosine LR, grad accum/clip, AMP, checkpoints)
│   ├── evaluation/         # LM perplexity, code-completion, and syntax evaluation
│   ├── inference/          # Autoregressive code generation
│   ├── agent/              # Kodra Agent tool foundation (roadmap scaffolding)
│   ├── server/             # FastAPI backend serving real model/training state to the dashboard
│   ├── scripts/            # Project health verifier
│   ├── notebooks/          # Google Colab training notebook
│   └── tests/              # pytest suite
│
└── README.md
```

## Quickstart

### Frontend + backend (recommended)
```bash
npm install
cd kodra-core && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt   # macOS/Linux
cd ..
npm run dev   # starts the Express gateway on :3000, which spawns the FastAPI backend on :8000
```

### Python core only
```bash
cd kodra-core
python -m scripts.verify_project       # end-to-end health check
python -m pytest tests/ -q             # full test suite
python -m training.train --epochs 5    # train Kodra Tiny locally (CPU-scale only)
python -m evaluation.evaluator --full  # LM + code-completion + syntax evaluation
```

See [kodra-core/README.md](kodra-core/README.md) for full details on the model core.
