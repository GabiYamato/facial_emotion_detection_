"""Distributed-training entry point with a safe single-process fallback."""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.training.scripts.train_model import prepare_run_config, run_standard_training
from src.training.strategies.multi_gpu_strategy import describe_strategy


def main():
	parser = argparse.ArgumentParser(description="Train using available compute")
	parser.add_argument("--config", default="config/model_config.yml")
	parser.add_argument("--run-name", default=None)
	parser.add_argument("--verbose", type=int, default=1)
	args = parser.parse_args()
	with open(args.config, "r") as stream:
		config = yaml.safe_load(stream)
	if args.verbose:
		print(f"Training strategy: {describe_strategy()}")
	if int(os.environ.get("WORLD_SIZE", "1")) > 1:
		raise NotImplementedError("DDP process-group integration is not enabled; run with one process.")
	config = prepare_run_config(config, args.run_name)
	run_standard_training(config, args.verbose)


if __name__ == "__main__":
	main()
