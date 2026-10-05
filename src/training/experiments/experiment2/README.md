# Experiment 2: Development Baseline

Experiment 2 is a ten-epoch CNN baseline with augmentation disabled. It is
intended for checking data loading, model construction, checkpointing, and
result generation before starting the full Experiment 1 benchmark.

```powershell
python src/training/scripts/train_model.py `
  --config src/training/experiments/experiment2/config.yml `
  --run-name experiment2_cnn_smoke `
  --verbose 1
```

Expected output is under:

```text
results/runs/experiment2_cnn_smoke/
  models/
  tables/
  logs/
  plots/
```