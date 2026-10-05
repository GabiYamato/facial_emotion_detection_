"""Reusable early-stopping state."""


class EarlyStopping:
	"""Track a monitored value and report when patience is exhausted."""

	def __init__(self, patience=15, mode="min", min_delta=0.0):
		if mode not in {"min", "max"}:
			raise ValueError("mode must be 'min' or 'max'")
		self.patience = patience
		self.mode = mode
		self.min_delta = min_delta
		self.best = float("inf") if mode == "min" else -float("inf")
		self.bad_epochs = 0

	def update(self, value):
		better = (value < self.best - self.min_delta if self.mode == "min"
				  else value > self.best + self.min_delta)
		if better:
			self.best = value
			self.bad_epochs = 0
			return True
		self.bad_epochs += 1
		return False

	@property
	def should_stop(self):
		return self.bad_epochs >= self.patience
