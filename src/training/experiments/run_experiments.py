"""Run selected model experiments into isolated subdirectories."""

import argparse
import copy
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[3]))

from src.models.architectures import get_available_models
from src.training.scripts.train_model import (
    prepare_run_config,
    run_hyperparameter_tuning,
    run_kfold,
    run_nested_kfold,
    run_standard_training,
)


def main():
    parser = argparse.ArgumentParser(description="Run multiple facial-emotion experiments")
    parser.add_argument("--config", default="config/model_config.yml")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--run-name", default="experiment")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        base_config = yaml.safe_load(stream)
    model_names = get_available_models() if "all" in args.models else args.models
    for model_name in model_names:
        config = copy.deepcopy(base_config)
        config["model"]["name"] = model_name
        config = prepare_run_config(config, f"{args.run_name}/{model_name}")
        training = config.get("training", {})
        if training.get("hyperparameter_tuning", {}).get("enabled", False):
            run_hyperparameter_tuning(config, args.verbose)
        elif training.get("nested_kfold", {}).get("enabled", False):
            run_nested_kfold(config, args.verbose)
        elif training.get("kfold", {}).get("enabled", False):
            run_kfold(config, args.verbose)
        else:
            run_standard_training(config, args.verbose)


if __name__ == "__main__":
    main()