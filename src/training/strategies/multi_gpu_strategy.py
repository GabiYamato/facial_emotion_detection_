"""Distributed training helpers."""

import torch


def distributed_available():
	return torch.cuda.is_available() and torch.cuda.device_count() > 1


def describe_strategy():
	if distributed_available():
		return f"ddp ({torch.cuda.device_count()} GPUs)"
	return "single-process fallback"
