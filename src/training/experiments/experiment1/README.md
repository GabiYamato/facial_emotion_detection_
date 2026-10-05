# Experiment 1: Full Model Benchmark

Experiment 1 compares every registered architecture with the same data split,
training policy, class balancing, and evaluation settings.

## Run every model

From the repository root:

```powershell
python src/training/experiments/run_experiments.py `
  --config src/training/experiments/experiment1/config.yml `
  --models all `
  --run-name experiment1 `
  --verbose 1
```

The configuration uses five outer folds and three inner folds. This produces a
large number of training jobs: 15 inner jobs plus one outer job per outer fold
for each model.

## Run selected models

```powershell
python src/training/experiments/run_experiments.py `
  --config src/training/experiments/experiment1/config.yml `
  --models cnn lstm vision_transformer `
  --run-name experiment1_selected
```

## Run one model independently

```powershell
python src/training/scripts/train_model.py `
  --config src/training/experiments/experiment1/config.yml `
  --run-name experiment1/cnn
```

The runner overwrites artifacts only when the same explicit run name and model
path are reused. Different model names have separate directories.