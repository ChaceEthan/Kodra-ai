"""
Lightweight runtime-environment configuration for Kodra Core.

This module is deliberately separate from `configs.config` (which holds
MODEL HYPERPARAMETERS such as embedding_dim and learning_rate, and is meant
to be checked into version control via default_config.json). Everything
here is RUNTIME/DEPLOYMENT configuration and secrets: device selection,
directories, CORS origins, and agent tool safety flags. Mixing the two
would make it easy to accidentally commit a secret inside a model config
file, so they are kept in separate modules.

No external dependency (e.g. python-dotenv) is required: this is a small,
dependency-free ".env" parser sufficient for KEY=VALUE lines.
"""
import os
from dataclasses import dataclass
from typing import List, Optional

_ENV_LOADED = False


def load_dotenv_once(path: Optional[str] = None) -> None:
    """Parses a simple KEY=VALUE .env file into os.environ, without
    overriding variables that are already set in the real environment.
    Safe to call multiple times - only loads once per process."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    if path is None:
        # kodra-core/configs/env.py -> repo root is two levels up
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        path = os.path.join(repo_root, ".env")

    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _split_csv(value: str) -> List[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_bool(value: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RuntimeConfig:
    device: str  # "auto" | "cpu" | "cuda"
    checkpoint_dir: str
    data_dir: str
    allowed_origins: List[str]
    require_tool_approval: bool
    enable_terminal_tools: bool
    log_level: str
    model_size: str
    api_key: str
    # Configuration boundary only - see kodra-core/agent/README.md /
    # REPOSITORY VECTOR INDEX status. No embeddings or retrieval are
    # implemented yet; nothing reads these two fields at runtime.
    vector_db_provider: str
    vector_db_path: str


def load_runtime_config(kodra_core_dir: Optional[str] = None) -> RuntimeConfig:
    """Reads runtime/deployment configuration from environment variables
    (populated from a .env file if present). Every value has a safe local
    default so the backend starts even with no .env file at all."""
    load_dotenv_once()

    core_dir = kodra_core_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return RuntimeConfig(
        device=os.environ.get("KODRA_DEVICE", "auto"),
        checkpoint_dir=os.environ.get("KODRA_CHECKPOINT_DIR", os.path.join(core_dir, "checkpoints")),
        data_dir=os.environ.get("KODRA_DATA_DIR", os.path.join(core_dir, "data")),
        allowed_origins=_split_csv(os.environ.get("KODRA_ALLOWED_ORIGINS", "http://localhost:3000")),
        require_tool_approval=_parse_bool(os.environ.get("KODRA_REQUIRE_TOOL_APPROVAL", ""), default=True),
        enable_terminal_tools=_parse_bool(os.environ.get("KODRA_ENABLE_TERMINAL_TOOLS", ""), default=False),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        model_size=os.environ.get("KODRA_MODEL_SIZE", "tiny"),
        api_key=os.environ.get("KODRA_API_KEY", ""),
        vector_db_provider=os.environ.get("VECTOR_DB_PROVIDER", "local"),
        vector_db_path=os.environ.get("VECTOR_DB_PATH", os.path.join(core_dir, "data", "vector_store")),
    )
