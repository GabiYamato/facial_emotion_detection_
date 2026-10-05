# Training Implementation Checklist

This checklist tracks the training-system completion work.

- [x] Add isolated run directories and explicit run names.
- [x] Save complete checkpoints, including optimizer, scheduler, epoch, and history state.
- [x] Implement resume training from a checkpoint.
- [x] Apply checkpoint monitor, mode, and `min_delta` independently from early stopping.
- [x] Save epoch history and persistent logs.
- [x] Evaluate and save test-set metrics after training.
- [x] Implement the configured schedulers, including warm restarts and one-cycle scheduling.
- [x] Implement nested K-fold cross-validation.
- [x] Implement a multi-model experiment runner using reusable core modules.
- [x] Implement callback modules and use them from the trainer.
- [x] Implement distributed-training entry-point behavior with a documented single-process fallback.
- [x] Honor `use_predefined_split` and validate data/config assumptions.
- [x] Persist Optuna studies and trial results.
- [x] Add plots and TensorBoard output when enabled.
- [x] Add focused tests for early stopping and experiment isolation.
- [x] Document setup, commands, artifacts, and failure/retry behavior.

Remaining limitation: true multi-process DDP is not enabled; the distributed
entry point deliberately rejects `WORLD_SIZE > 1` rather than silently running
incorrectly.