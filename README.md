# Facial Emotion Detection

## Training setup

Create a project-local virtual environment, install the dependencies in
`config/requirements.txt`, and run commands from the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r config/requirements.txt
.\.venv\Scripts\python.exe src/training/scripts/train_model.py --help
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

## Repository structure

```text
config/
	model_config.yml             Master configuration example
	requirements.txt             Python dependencies
data/
	raw/<class>/                 Input images grouped by class folder
src/
	data_pipeline/               Dataset, transforms, splits, and loaders
	models/architectures/        Model implementations and model registry
	training/
		callbacks/                 Checkpoint, early-stopping, and TensorBoard helpers
		experiments/               Reusable experiment configs and runners
		metrics/                   Metric computation and fold aggregation
		scripts/                   Standard, resume, and distributed entry points
		strategies/                Device and distributed strategy helpers
	testing/                     Unit and integration tests
results/
	runs/<run-name>/              Generated training artifacts
docs/                           Checklists and project documentation
notebooks/                      Analysis and result visualization notebooks
```

The reusable implementation lives under `src/`. Experiment-specific choices
belong under `src/training/experiments/`; generated outputs belong under
`results/runs/`, not beside the source configuration.

## Data layout

With `data.use_predefined_split: true`, images must use `train_` and `test_`
filename prefixes:

```text
data/raw/
	1/train_00001.jpg
	1/test_00001.jpg
	2/train_00002.jpg
	2/test_00002.jpg
	...
	7/train_00007.jpg
```

The folder-to-class mapping is defined by `data.class_to_folder`. If
`use_predefined_split` is false, the loader scans all images and creates
stratified train, validation, and test partitions using `val_split` and
`test_split`.

## Configuration examples

The master example is [config/model_config.yml](config/model_config.yml).
Runnable experiment configs are:

- [Experiment 1 config](src/training/experiments/experiment1/config.yml):
	full nested-K-fold benchmark across all models.
- [Experiment 2 config](src/training/experiments/experiment2/config.yml):
	short CNN smoke test with augmentation disabled.

Configs are standalone YAML files. To change an experiment, copy its config,
change `model.name` or a training section, and use a new `--run-name`.

Important switches:

```yaml
augmentation:
	enabled: false

training:
	class_balancing:
		enabled: true
		method: class_weights       # class_weights or oversampling
	nested_kfold:
		enabled: true
		outer_splits: 5
		inner_splits: 3
	early_stopping:
		enabled: true
	checkpointing:
		enabled: true

logging:
	save_plots: true
	tensorboard:
		enabled: false
```

Setting augmentation to false disables random flips, rotations, affine
transforms, color jitter, blur, and random erasing. Normalization still runs.

## Artifact layout

For `--run-name experiment1/cnn`, artifacts are written below:

```text
results/runs/experiment1/cnn/
	models/
		best_model_cnn.pt
		last_checkpoint_cnn.pt
	tables/
		results_cnn.json
		nested_kfold_results_cnn.json
	logs/
		epochs_cnn.jsonl
		tensorboard/              When TensorBoard is enabled
	plots/
		history_cnn.png           When plot saving is enabled
```

Checkpoints include model, optimizer, scheduler, epoch, best metric,
patience, and training history. Reusing an explicit run name overwrites files
inside that run; a new run name preserves the previous run.

## Command reference

### `train_model.py`

```text
--config PATH       YAML configuration; default: config/model_config.yml
--verbose LEVEL     0 disables progress, 1 prints progress; default: 1
--run-name NAME     Explicit output directory name
--resume PATH       Latest checkpoint to resume from
```

### `run_experiments.py`

```text
--config PATH       YAML configuration
--models NAMES      One or more model names, or all
--run-name NAME     Parent output directory; default: experiment
--verbose LEVEL     Progress verbosity; default: 1
```

### `resume_training.py`

```text
checkpoint PATH     Path to last_checkpoint_<model>.pt
--config PATH       YAML configuration
--verbose LEVEL     Progress verbosity
```

### `distributed_train.py`

```text
--config PATH
--run-name NAME
--verbose LEVEL
```

## Typical workflow

1. Run Experiment 2 to verify the environment and dataset.
2. Inspect `results/runs/experiment2_cnn_smoke/tables/`.
3. Run one model from Experiment 1 to estimate runtime.
4. Run all models with `--models all`.
5. Use the saved JSON files or [visualize_results.ipynb](notebooks/results/visualize_results.ipynb) for comparisons.
6. Resume from `last_checkpoint_<model>.pt` if a run stops after a checkpoint.

## Validation

Compile the source tree:

```powershell
python -m compileall -q src
```

Run the available unit tests:

```powershell
python -m unittest discover -s src/testing
```

Install dependencies before running training or tests that import the full
trainer:

```powershell
.\.venv\Scripts\python.exe -m pip install -r config/requirements.txt
```
