# Facial Emotion Detection

## Training setup

Install the dependencies in `config/requirements.txt` into the active Python
environment, then run commands from the repository root.

```powershell
python -m pip install -r config/requirements.txt
python src/training/scripts/train_model.py --help
```

## Standard training

```powershell
python src/training/scripts/train_model.py `
	--config config/model_config.yml `
	--run-name cnn_baseline `
	--verbose 1
```

Set `model.name` in the YAML file to one of `cnn`, `rnn`, `lstm`,
`lstm_attention`, `bilstm`, `bilstm_attention`, `residual_bilstm`, `cnn_lstm`,
`cnn_bilstm`, `vision_transformer`, or `swin_transformer`.

Each run is isolated under `results/runs/<run-name>/` and contains `models/`,
`tables/`, `logs/`, and `plots/`. Epoch metrics are written to a JSONL file;
the result JSON contains the final validation metrics, optional test metrics,
timing, and complete in-memory history. Best and latest checkpoints include
model, optimizer, scheduler, epoch, patience, and history state.

Without `--run-name`, a timestamped run directory is generated. Reusing an
explicit run name intentionally writes into that run directory.

## Resume training

```powershell
python src/training/scripts/resume_training.py `
	results/runs/cnn_baseline/models/last_checkpoint_cnn.pt `
	--config config/model_config.yml
```

Resume loads the model, optimizer, scheduler, best values, patience state, and
history. There is no resume support for a process that failed before writing a
checkpoint.

## K-fold and nested K-fold

Set `training.kfold.enabled: true` or `training.nested_kfold.enabled: true` in
the YAML configuration. K-fold artifacts are isolated per fold; the aggregate
JSON is saved in the run's `tables/` directory. Nested K-fold uses inner folds
for metric-based selection and outer folds for evaluation.

## Hyperparameter tuning

Set `training.hyperparameter_tuning.enabled: true`. Optuna persists its study
in the run directory as `optuna.db`, along with best parameters and all trial
records. Optuna is optional unless this mode is enabled.

## Multiple models

```powershell
python src/training/experiments/run_experiments.py `
	--config config/model_config.yml `
	--models cnn lstm vision_transformer `
	--run-name comparison
```

Each model is written to `results/runs/comparison/<model>/`, so successful
results remain separate. A model failure stops the runner at that model; prior
model directories remain intact and can be rerun individually.

## Experiment 1

Experiment 1 runs every registered model with the self-contained nested
cross-validation and class-balanced configuration:

```powershell
python src/training/experiments/run_experiments.py `
	--config src/training/experiments/experiment1/config.yml `
	--models all `
	--run-name experiment1 `
	--verbose 1
```

Results are written under `results/runs/experiment1/<model>/`. Each model has
five outer folds and three inner folds, so this is a large run before the
configured early stopping is applied.

## Logging and plots

Set `logging.tensorboard.enabled: true` for TensorBoard event files. Set
`logging.save_plots: true` for loss and macro-F1 history plots. `--verbose 0`
silences progress output but does not disable saved artifacts.

## Distributed entry point

```powershell
python src/training/scripts/distributed_train.py --config config/model_config.yml
```

This entry point detects multiple GPUs but currently refuses a multi-process
launch until DDP process-group integration is added. Single-process execution
uses the same trainer and output isolation as standard training.
