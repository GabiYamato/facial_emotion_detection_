"""Device strategy helpers."""

import torch


def get_device(config):
	requested = config.get("project", {}).get("device", "auto")
	if requested != "auto":
		return torch.device(requested)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")
