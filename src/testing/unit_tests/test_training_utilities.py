import tempfile
import unittest
from pathlib import Path

from src.training.callbacks.early_stopping import EarlyStopping

try:
    from src.training.scripts.train_model import prepare_run_config
except ModuleNotFoundError:
    prepare_run_config = None


class TrainingUtilitiesTests(unittest.TestCase):
    def test_early_stopping_honors_min_delta_and_patience(self):
        callback = EarlyStopping(patience=2, mode="min", min_delta=0.1)
        self.assertTrue(callback.update(1.0))
        self.assertFalse(callback.update(0.95))
        self.assertTrue(callback.update(0.7))
        self.assertFalse(callback.update(0.69))
        self.assertTrue(callback.should_stop is False)
        self.assertFalse(callback.update(0.68))
        self.assertTrue(callback.should_stop)

    def test_prepare_run_config_isolates_outputs(self):
        if prepare_run_config is None:
            self.skipTest("project dependencies are not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"logging": {"runs_dir": temp_dir}, "training": {"checkpointing": {}}}
            prepared = prepare_run_config(config, "sample")
            run_dir = Path(temp_dir) / "sample"
            self.assertEqual(Path(prepared["logging"]["results_dir"]), run_dir / "tables")
            self.assertEqual(Path(prepared["training"]["checkpointing"]["save_dir"]), run_dir / "models")


if __name__ == "__main__":
    unittest.main()