__all__ = ["Trainer", "get_device", "set_seed"]


def __getattr__(name):
	if name == "Trainer":
		from .trainer import Trainer
		return Trainer
	if name in {"get_device", "set_seed"}:
		from .utils import get_device, set_seed
		return {"get_device": get_device, "set_seed": set_seed}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
