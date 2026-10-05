"""Reusable model checkpoint manager."""

from pathlib import Path
import torch


class ModelCheckpoint:
	"""Save the best state and the latest resumable state."""

	def __init__(self, directory, model_name, monitor="val_loss", mode="min"):
		if mode not in {"min", "max"}:
			raise ValueError("mode must be 'min' or 'max'")
		self.directory = Path(directory)
		self.directory.mkdir(parents=True, exist_ok=True)
		self.best_path = self.directory / f"best_model_{model_name}.pt"
		self.last_path = self.directory / f"last_checkpoint_{model_name}.pt"
		self.monitor = monitor
		self.mode = mode
		self.best = float("inf") if mode == "min" else -float("inf")

	def save(self, state, value, is_last=True):
		better = value < self.best if self.mode == "min" else value > self.best
		if better:
			self.best = value
			torch.save(state, self.best_path)
		if is_last:
			torch.save(state, self.last_path)
		return better
