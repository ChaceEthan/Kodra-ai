import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(preference: str = "auto") -> torch.device:
    """Resolves a torch device. `preference` is one of "auto", "cpu", "cuda"
    (typically sourced from the KODRA_DEVICE environment variable). "auto"
    picks CUDA/MPS if available, else CPU. An explicit "cuda" that isn't
    actually available falls back to CPU rather than raising, since a
    developer's laptop should never crash the backend for lacking a GPU."""
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
