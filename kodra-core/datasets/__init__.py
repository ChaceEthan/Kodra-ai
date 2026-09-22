from .sample_code import SAMPLE_CODE_CORPUS

__all__ = ["CodeDataset", "create_dataloader", "SAMPLE_CODE_CORPUS"]


def __getattr__(name):
	if name in {"CodeDataset", "create_dataloader"}:
		from .dataset import CodeDataset, create_dataloader
		return {"CodeDataset": CodeDataset, "create_dataloader": create_dataloader}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
