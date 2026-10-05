"""Resume a standard training run from its latest checkpoint."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.training.scripts.train_model import run_standard_training, prepare_run_config


def main():
	parser = argparse.ArgumentParser(description="Resume facial emotion model training")
	parser.add_argument("checkpoint", help="Path to last_checkpoint_<model>.pt")
	parser.add_argument("--config", default="config/model_config.yml")
	parser.add_argument("--verbose", type=int, default=1)
	args = parser.parse_args()
	with open(args.config, "r") as stream:
		config = yaml.safe_load(stream)
	config = prepare_run_config(config, resume_path=args.checkpoint)
	run_standard_training(config, args.verbose, args.checkpoint)


if __name__ == "__main__":
	main()
