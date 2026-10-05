"""Optional TensorBoard logging callback."""


class TensorBoardLogger:
	def __init__(self, log_dir, enabled=True):
		self.writer = None
		if enabled:
			from torch.utils.tensorboard import SummaryWriter
			self.writer = SummaryWriter(log_dir=log_dir)

	def log(self, metrics, step, prefix=""):
		if self.writer is None:
			return
		for name, value in metrics.items():
			if isinstance(value, (int, float)):
				self.writer.add_scalar(f"{prefix}{name}", value, step)

	def close(self):
		if self.writer is not None:
			self.writer.close()
