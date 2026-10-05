import os
import sys
import yaml
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.data_pipeline.loaders.dataloader import (
    create_dataloaders,
    create_kfold_splits,
    create_kfold_dataloaders,
    compute_class_weights,
)
from src.models.architectures import build_model
from src.training.metrics.metrics import MetricsComputer, TimeTracker
from src.training.callbacks.tensorboard_logging import TensorBoardLogger

try:
    import optuna
except ImportError:
    optuna = None


class Trainer:
    """Handles the training and evaluation loop for a single model instance."""

    def __init__(self, model, config, verbose=1, trial=None, resume_path=None):
        self.config = config
        self.verbose = verbose
        self.trial = trial
        self.train_cfg = config["training"]
        self.device = self._get_device()
        self.model = model.to(self.device)

        # Early stopping and checkpointing
        self.early_stopping_cfg = self.train_cfg.get("early_stopping", {})
        self.checkpointing_cfg = self.train_cfg.get("checkpointing", {})
        self.checkpoint_dir = Path(self.checkpointing_cfg.get("save_dir", "results/models"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / f"best_model_{self.config['model']['name']}.pt"
        self.last_checkpoint_path = self.checkpoint_dir / f"last_checkpoint_{self.config['model']['name']}.pt"
        self.checkpoint_mode = self.checkpointing_cfg.get("mode", "max")
        self.checkpoint_monitor = self.checkpointing_cfg.get("monitor", "val_loss")
        early_mode = self.early_stopping_cfg.get("mode", "min")
        self.best_metric = -float("inf") if early_mode == "max" else float("inf")
        self.checkpoint_best = -float("inf") if self.checkpoint_mode == "max" else float("inf")
        self.patience_counter = 0
        self.best_model_state = None
        self.start_epoch = 0
        self.history = {"train": [], "val": []}

        self.metrics_computer = MetricsComputer(config)
        self.time_tracker = TimeTracker()
        log_cfg = config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "results/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_log_path = self.log_dir / f"epochs_{config['model']['name']}.jsonl"
        tensorboard_cfg = log_cfg.get("tensorboard", {})
        self.tensorboard = TensorBoardLogger(
            tensorboard_cfg.get("log_dir", str(self.log_dir / "tensorboard")),
            enabled=tensorboard_cfg.get("enabled", False),
        )

        # Class weights
        class_weights = compute_class_weights(config)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=self.train_cfg.get("label_smoothing", 0.0),
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self._load_checkpoint(resume_path)

    def _load_checkpoint(self, resume_path):
        if not resume_path:
            return
        checkpoint = torch.load(resume_path, map_location=self.device)
        state = checkpoint.get("model_state", checkpoint)
        self.model.load_state_dict(state)
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler is not None and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.start_epoch = checkpoint.get("epoch", -1) + 1
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.checkpoint_best = checkpoint.get("checkpoint_best", self.checkpoint_best)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.best_model_state = checkpoint.get("best_model_state")
        self.history = checkpoint.get("history", self.history)

    def _get_device(self):
        device_str = self.config["project"].get("device", "auto")
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def _build_optimizer(self):
        opt_name = self.train_cfg.get("optimizer", "adam").lower()
        lr = self.train_cfg.get("learning_rate", 1e-3)
        weight_decay = self.train_cfg.get("weight_decay", 1e-4)
        opt_params = self.train_cfg.get("optimizer_params", {})

        if opt_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                              betas=tuple(opt_params.get("betas", (0.9, 0.999))),
                              eps=opt_params.get("eps", 1e-8))
        elif opt_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                               betas=tuple(opt_params.get("betas", (0.9, 0.999))),
                               eps=opt_params.get("eps", 1e-8))
        elif opt_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay,
                             momentum=opt_params.get("momentum", 0.9),
                             nesterov=opt_params.get("nesterov", True))
        elif opt_name == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

    def _build_scheduler(self):
        sched_cfg = self.train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "none").lower()
        if sched_type == "none":
            return None
        
        params = sched_cfg.get(sched_type, {})
        if sched_type == "cosine_annealing":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.get("T_max", 100), eta_min=params.get("eta_min", 1e-6)
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=params.get("step_size", 20), gamma=params.get("gamma", 0.5)
            )
        elif sched_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=params.get("mode", "min"), factor=params.get("factor", 0.5),
                patience=params.get("patience", 5), min_lr=params.get("min_lr", 1e-7)
            )
        elif sched_type == "cosine_warm_restarts":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=params.get("T_0", 10),
                T_mult=params.get("T_mult", 2), eta_min=params.get("eta_min", 1e-6)
            )
        elif sched_type == "one_cycle":
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=params.get("max_lr", 0.01),
                epochs=self.train_cfg.get("epochs", 100),
                steps_per_epoch=self.train_cfg.get("steps_per_epoch", 1),
                pct_start=params.get("pct_start", 0.3),
                anneal_strategy=params.get("anneal_strategy", "cos"),
            )
        return None

    def _save_checkpoint(self, epoch, path):
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "best_model_state": self.best_model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_metric": self.best_metric,
            "checkpoint_best": self.checkpoint_best,
            "patience_counter": self.patience_counter,
            "history": self.history,
            "model_name": self.config["model"]["name"],
        }, path)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        clip_val = self.train_cfg.get("gradient_clip_value", 0.0)

        pbar = tqdm(train_loader, desc="Training", disable=self.verbose < 1)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            if clip_val > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
            self.optimizer.step()
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader.dataset)
        metrics = self.metrics_computer.compute_all(np.array(all_labels), np.array(all_preds))
        metrics["loss"] = avg_loss
        return metrics

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Evaluating", disable=self.verbose < 1)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader.dataset)
        metrics = self.metrics_computer.compute_all(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        metrics["loss"] = avg_loss
        return metrics

    def run(self, train_loader, val_loader, test_loader=None):
        epochs = self.train_cfg.get("epochs", 100)
        self.time_tracker.start_training()
        
        for epoch in range(self.start_epoch, epochs):
            if self.verbose >= 1:
                print(f"\\nEpoch {epoch+1}/{epochs}")
            
            self.time_tracker.start_epoch()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.time_tracker.end_epoch()

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)
            with self.epoch_log_path.open("a", encoding="utf-8") as log_file:
                json.dump({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}, log_file)
                log_file.write("\n")
            self.tensorboard.log(train_metrics, epoch + 1, "train/")
            self.tensorboard.log(val_metrics, epoch + 1, "val/")

            if self.verbose >= 1:
                print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | Train F1: {train_metrics['f1_macro']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}   | Val Acc: {val_metrics['accuracy']:.4f}   | Val F1: {val_metrics['f1_macro']:.4f}")

            # Scheduler step
            if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                pass
            elif isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # Optuna Pruning
            if self.trial is not None:
                monitor_metric = self.train_cfg.get("hyperparameter_tuning", {}).get("metric", "val_f1_macro").replace("val_", "")
                self.trial.report(val_metrics[monitor_metric], epoch)
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            # Early stopping logic
            if self.early_stopping_cfg.get("enabled", False):
                monitor = self.early_stopping_cfg.get("monitor", "val_loss")
                mode = self.early_stopping_cfg.get("mode", "min")
                
                is_val = monitor.startswith("val_")
                monitor_key = monitor.replace("val_", "")
                current_score = val_metrics[monitor_key] if is_val else train_metrics[monitor_key]

                min_delta = self.early_stopping_cfg.get("min_delta", 0.0)
                is_better = (current_score < self.best_metric - min_delta if mode == "min"
                             else current_score > self.best_metric + min_delta)
                
                if is_better:
                    self.best_metric = current_score
                    self.patience_counter = 0
                    self.best_model_state = deepcopy(self.model.state_dict())
                    
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_cfg.get("patience", 15):
                        if self.verbose >= 1:
                            print(f"Early stopping triggered at epoch {epoch+1}")
                        break

            # Checkpoint monitoring is independent from early stopping.
            if self.checkpointing_cfg.get("enabled", False):
                checkpoint_monitor_key = self.checkpoint_monitor.replace("val_", "")
                checkpoint_values = val_metrics if self.checkpoint_monitor.startswith("val_") else train_metrics
                checkpoint_score = checkpoint_values[checkpoint_monitor_key]
                checkpoint_better = (
                    checkpoint_score > self.checkpoint_best
                    if self.checkpoint_mode == "max"
                    else checkpoint_score < self.checkpoint_best
                )
                if checkpoint_better:
                    self.checkpoint_best = checkpoint_score
                    self.best_model_state = deepcopy(self.model.state_dict())
                    if self.checkpointing_cfg.get("save_best_only", True):
                        self._save_checkpoint(epoch, self.checkpoint_path)
                if not self.checkpointing_cfg.get("save_best_only", True):
                    self._save_checkpoint(
                        epoch,
                        self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    )

            if self.checkpointing_cfg.get("enabled", False):
                self._save_checkpoint(epoch, self.last_checkpoint_path)
        
        self.time_tracker.end_training()
        
        if self.early_stopping_cfg.get("restore_best_weights", True) and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final evaluation
        self.time_tracker.start_eval()
        final_val_metrics = self.evaluate(val_loader)
        final_test_metrics = self.evaluate(test_loader) if test_loader is not None else None
        self.time_tracker.end_eval()
        
        final_val_metrics["timing"] = self.time_tracker.summary()
        final_val_metrics["history"] = self.history
        if final_test_metrics is not None:
            final_val_metrics["test"] = final_test_metrics
        self.tensorboard.close()
        
        # Save results
        if self.config.get("logging", {}).get("save_results_json", True):
            res_dir = Path(self.config["logging"].get("results_dir", "results/tables"))
            res_dir.mkdir(parents=True, exist_ok=True)
            with open(res_dir / f"results_{self.config['model']['name']}.json", "w") as f:
                json.dump(final_val_metrics, f, indent=4)

            if self.config.get("logging", {}).get("save_plots", False):
                self._save_history_plot()

        return final_val_metrics

    def _save_history_plot(self):
        import matplotlib.pyplot as plt

        plot_dir = Path(self.config["logging"].get("plots_dir", "results/plots"))
        plot_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.history["train"]) + 1)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [item["loss"] for item in self.history["train"]], label="train")
        axes[0].plot(epochs, [item["loss"] for item in self.history["val"]], label="validation")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[1].plot(epochs, [item["f1_macro"] for item in self.history["train"]], label="train")
        axes[1].plot(epochs, [item["f1_macro"] for item in self.history["val"]], label="validation")
        axes[1].set_title("Macro F1")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_dir / f"history_{self.config['model']['name']}.png")
        plt.close(figure)


def run_standard_training(config, verbose, resume_path=None):
    print("Running standard training...")
    train_loader, val_loader, test_loader = create_dataloaders(config, verbose)
    config["training"]["steps_per_epoch"] = len(train_loader)
    model = build_model(config)
    trainer = Trainer(model, config, verbose, resume_path=resume_path)
    metrics = trainer.run(train_loader, val_loader, test_loader)
    print(f"Final Validation Accuracy: {metrics['accuracy']:.4f}")
    return metrics


def run_kfold(config, verbose):
    print("Running K-Fold Cross Validation...")
    kfold_cfg = config["training"]["kfold"]
    splits = create_kfold_splits(config, n_splits=kfold_cfg.get("n_splits", 5), 
                                 shuffle=kfold_cfg.get("shuffle", True), 
                                 stratified=kfold_cfg.get("stratified", True))
    
    fold_results = []
    for i, (train_idx, val_idx) in enumerate(splits):
        if verbose >= 1:
            print(f"\\n{'='*30}\\nFold {i+1}/{len(splits)}\\n{'='*30}")
        
        fold_config = prepare_run_config(deepcopy(config), f"{Path(config['logging'].get('run_dir', 'kfold')).name}/fold_{i + 1}")
        train_loader, val_loader = create_kfold_dataloaders(fold_config, train_idx, val_idx, verbose)
        fold_config["training"]["steps_per_epoch"] = len(train_loader)
        model = build_model(fold_config)
        trainer = Trainer(model, fold_config, verbose)
        metrics = trainer.run(train_loader, val_loader)
        fold_results.append(metrics)
    
    mc = MetricsComputer(config)
    aggregated = mc.aggregate_fold_results(fold_results)
    
    if config.get("logging", {}).get("save_results_json", True):
        res_dir = Path(config["logging"].get("results_dir", "results/tables"))
        res_dir.mkdir(parents=True, exist_ok=True)
        with open(res_dir / f"kfold_results_{config['model']['name']}.json", "w") as f:
            json.dump(aggregated, f, indent=4)
            
    if verbose >= 1:
        print(f"\\nK-Fold Aggregated Accuracy: {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    return aggregated


def run_nested_kfold(config, verbose):
    """Run inner model selection inside each outer evaluation fold."""
    nested_cfg = config["training"]["nested_kfold"]
    data_cfg = config["data"]
    full_dataset = create_kfold_splits(
        config, n_splits=nested_cfg.get("outer_splits", 5),
        shuffle=nested_cfg.get("shuffle", True),
        stratified=nested_cfg.get("stratified", True),
    )
    labels_dataset = __import__(
        "src.data_pipeline.loaders.dataset_classes.emotion_dataset",
        fromlist=["EmotionDataset"],
    ).EmotionDataset(data_cfg["raw_dir"], split="all", config=data_cfg)
    labels = labels_dataset.get_labels()
    outer_results = []
    base_run = Path(config.get("logging", {}).get("run_dir", "results/runs/nested"))

    for outer_index, (outer_train, outer_val) in enumerate(full_dataset):
        inner_labels = labels[outer_train]
        inner_splitter = StratifiedKFold(
            n_splits=nested_cfg.get("inner_splits", 3),
            shuffle=nested_cfg.get("shuffle", True),
            random_state=config["project"].get("seed", 42),
        )
        inner_results = []
        for inner_index, (inner_train_rel, inner_val_rel) in enumerate(
                inner_splitter.split(np.zeros(len(inner_labels)), inner_labels)):
            inner_config = deepcopy(config)
            inner_config = prepare_run_config(
                inner_config,
                str(base_run / f"outer_{outer_index + 1}" / f"inner_{inner_index + 1}"),
            )
            inner_train = outer_train[inner_train_rel]
            inner_val = outer_train[inner_val_rel]
            train_loader, val_loader = create_kfold_dataloaders(
                inner_config, inner_train, inner_val, verbose
            )
            inner_config["training"]["steps_per_epoch"] = len(train_loader)
            trainer = Trainer(build_model(inner_config), inner_config, verbose)
            inner_results.append(trainer.run(train_loader, val_loader))

        selection_metric = nested_cfg.get("selection_metric", "f1_macro")
        selected = max(inner_results, key=lambda result: result.get(selection_metric, float("-inf")))
        outer_config = deepcopy(config)
        outer_config = prepare_run_config(
            outer_config, str(base_run / f"outer_{outer_index + 1}")
        )
        train_loader, val_loader = create_kfold_dataloaders(
            outer_config, outer_train, outer_val, verbose
        )
        outer_config["training"]["steps_per_epoch"] = len(train_loader)
        trainer = Trainer(build_model(outer_config), outer_config, verbose)
        outer_result = trainer.run(train_loader, val_loader)
        outer_result["inner_selection"] = {
            "metric": selection_metric,
            "selected_value": selected.get(selection_metric),
            "inner_results": inner_results,
        }
        outer_results.append(outer_result)

    aggregated = MetricsComputer(config).aggregate_fold_results(outer_results)
    result_dir = Path(config["logging"].get("results_dir", "results/tables"))
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f"nested_kfold_results_{config['model']['name']}.json", "w") as stream:
        json.dump(aggregated, stream, indent=4)
    return aggregated


def run_hyperparameter_tuning(config, verbose):
    if optuna is None:
        raise ImportError("Optuna is not installed. Please install it to run hyperparameter tuning.")
    
    tuning_cfg = config["training"]["hyperparameter_tuning"]
    print("Running Hyperparameter Tuning with Optuna...")

    def objective(trial):
        # Override config with trial suggestions
        trial_config = deepcopy(config)
        
        spaces = tuning_cfg.get("search_spaces", {})
        for param, space in spaces.items():
            stype = space.get("type")
            if stype == "loguniform":
                val = trial.suggest_float(param, space["low"], space["high"], log=True)
            elif stype == "uniform":
                val = trial.suggest_float(param, space["low"], space["high"])
            elif stype == "categorical":
                val = trial.suggest_categorical(param, space["choices"])
            
            # Map param to config
            if param in ["learning_rate", "batch_size", "weight_decay", "optimizer"]:
                trial_config["training"][param] = val
            elif param == "dropout":
                model_name = trial_config["model"]["name"]
                trial_config["model"][model_name]["dropout"] = val
                
        # Run standard training for evaluation
        train_loader, val_loader, _ = create_dataloaders(trial_config, verbose=0)
        trial_config["training"]["steps_per_epoch"] = len(train_loader)
        model = build_model(trial_config)
        trainer = Trainer(model, trial_config, verbose=0, trial=trial)
        
        try:
            metrics = trainer.run(train_loader, val_loader)
        except optuna.exceptions.TrialPruned:
            raise optuna.exceptions.TrialPruned()
            
        monitor_metric = tuning_cfg.get("metric", "val_f1_macro").replace("val_", "")
        return metrics[monitor_metric]

    run_dir = Path(config.get("logging", {}).get("run_dir", "results/runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{config['model']['name']}_study",
        direction=tuning_cfg.get("direction", "maximize"),
        storage=f"sqlite:///{(run_dir / 'optuna.db').resolve().as_posix()}",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=tuning_cfg.get("n_trials", 30), timeout=tuning_cfg.get("timeout", None))

    print(f"\\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save tuning results
    res_dir = Path(config["training"]["hyperparameter_tuning"].get("tuning_results", "src/training/hyperparameters/tuning_results"))
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / f"optuna_best_params_{config['model']['name']}.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    with open(res_dir / f"optuna_trials_{config['model']['name']}.json", "w") as f:
        json.dump([
            {"number": item.number, "value": item.value, "params": item.params,
             "state": item.state.name}
            for item in study.trials
        ], f, indent=4)


def prepare_run_config(config, run_name=None, resume_path=None):
    """Route all generated artifacts into one explicit run directory."""
    logging_cfg = config.setdefault("logging", {})
    if resume_path:
        run_dir = Path(resume_path).resolve().parent.parent
    else:
        selected_name = run_name or logging_cfg.get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")
        selected_path = Path(selected_name)
        run_dir = selected_path if selected_path.is_absolute() else Path(logging_cfg.get("runs_dir", "results/runs")) / selected_path
    logging_cfg["run_dir"] = str(run_dir)
    logging_cfg["results_dir"] = str(run_dir / "tables")
    logging_cfg["log_dir"] = str(run_dir / "logs")
    logging_cfg["plots_dir"] = str(run_dir / "plots")
    config.setdefault("training", {}).setdefault("checkpointing", {})["save_dir"] = str(run_dir / "models")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial Emotion Detection Model")
    parser.add_argument("--config", type=str, default="config/model_config.yml", help="Path to YAML config")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0=silent, 1=progress/info)")
    parser.add_argument("--run-name", type=str, default=None, help="Name for the isolated output run directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to a last_checkpoint_*.pt file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config = prepare_run_config(config, args.run_name, args.resume)

    # Set seeds for reproducibility
    seed = config["project"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if config["project"].get("deterministic", True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config["training"].get("hyperparameter_tuning", {}).get("enabled", False):
        run_hyperparameter_tuning(config, args.verbose)
    elif config["training"].get("nested_kfold", {}).get("enabled", False):
        run_nested_kfold(config, args.verbose)
    elif config["training"].get("kfold", {}).get("enabled", False):
        run_kfold(config, args.verbose)
    else:
        run_standard_training(config, args.verbose, args.resume)
