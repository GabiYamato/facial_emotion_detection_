"""
Comprehensive metrics computation for training and evaluation.

Supports:
  - Accuracy, F1, Precision, Recall (macro / weighted / per-class)
  - AUC-ROC (macro / per-class)
  - FPR per class
  - 95% confidence intervals via bootstrap
  - Standard deviation across folds
  - Computational time tracking
  - Confusion matrix and classification report
"""

from __future__ import annotations

import time
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from scipy import stats


class MetricsComputer:
    """
    Computes, aggregates, and exports evaluation metrics.

    Args:
        config: Full project configuration dictionary.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.eval_cfg = config.get("evaluation", {})
        self.num_classes = config["data"].get("num_classes", 7)
        self.class_names = config["data"].get(
            "class_names",
            [f"Class_{i}" for i in range(self.num_classes)],
        )
        self.ci_level = self.eval_cfg.get("confidence_interval", 0.95)
        self.bootstrap_n = self.eval_cfg.get("bootstrap_iterations", 1000)

    # ------------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------------
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute all configured metrics.

        Args:
            y_true: Ground-truth labels, shape ``(N,)``.
            y_pred: Predicted labels, shape ``(N,)``.
            y_prob: Predicted probabilities, shape ``(N, C)``.  Required for
                    AUC and FPR.

        Returns:
            Dictionary of metric name → value.
        """
        results: Dict = {}

        results["accuracy"] = float(accuracy_score(y_true, y_pred))
        results["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        results["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        results["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        results["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        results["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        results["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        if self.eval_cfg.get("per_class_metrics", True):
            per_class = self._per_class_metrics(y_true, y_pred)
            results["per_class"] = per_class

        # AUC & FPR (require probabilities)
        if y_prob is not None:
            auc_results = self._compute_auc(y_true, y_prob)
            results.update(auc_results)
            fpr_results = self._compute_fpr(y_true, y_prob)
            results.update(fpr_results)

        # Confusion matrix
        if self.eval_cfg.get("confusion_matrix", True):
            cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
            results["confusion_matrix"] = cm.tolist()

        # Classification report
        if self.eval_cfg.get("classification_report", True):
            report = classification_report(
                y_true, y_pred,
                target_names=self.class_names,
                zero_division=0,
                output_dict=True,
            )
            results["classification_report"] = report

        # Bootstrap confidence intervals
        results["confidence_intervals"] = self._bootstrap_ci(y_true, y_pred, y_prob)

        return results

    # ------------------------------------------------------------------
    # Per-class
    # ------------------------------------------------------------------
    def _per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        precs = precision_score(y_true, y_pred, average=None, zero_division=0)
        recs = recall_score(y_true, y_pred, average=None, zero_division=0)

        per_class = {}
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            per_class[name] = {
                "f1": float(f1s[i]) if i < len(f1s) else 0.0,
                "precision": float(precs[i]) if i < len(precs) else 0.0,
                "recall": float(recs[i]) if i < len(recs) else 0.0,
            }
        return per_class

    # ------------------------------------------------------------------
    # AUC-ROC
    # ------------------------------------------------------------------
    def _compute_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        results: Dict = {}
        try:
            # One-hot encode y_true for multi-class AUC
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            if y_bin.shape[1] == 1:
                y_bin = np.hstack([1 - y_bin, y_bin])

            results["auc_macro"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
            results["auc_weighted"] = float(roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr"))

            # Per-class AUC
            auc_per_class = {}
            for i in range(self.num_classes):
                name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                try:
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    auc_per_class[name] = float(auc(fpr, tpr))
                except Exception:
                    auc_per_class[name] = None
            results["auc_per_class"] = auc_per_class

            # ROC curve data (for plotting)
            if self.eval_cfg.get("roc_curves", True):
                roc_data = {}
                for i in range(self.num_classes):
                    name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
                    try:
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                    except Exception:
                        pass
                results["roc_curves"] = roc_data

        except Exception as e:
            results["auc_macro"] = None
            results["auc_error"] = str(e)

        return results

    # ------------------------------------------------------------------
    # FPR per class
    # ------------------------------------------------------------------
    def _compute_fpr(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
        if y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        fpr_per_class = {}
        for i in range(self.num_classes):
            name = self.class_names[i] if i < len(self.class_names) else f"class_{i}"
            try:
                fp = np.sum((y_prob[:, i] >= 0.5) & (y_bin[:, i] == 0))
                tn = np.sum((y_prob[:, i] < 0.5) & (y_bin[:, i] == 0))
                fpr_per_class[name] = float(fp / (fp + tn + 1e-9))
            except Exception:
                fpr_per_class[name] = None

        return {"fpr_per_class": fpr_per_class}

    # ------------------------------------------------------------------
    # Bootstrap confidence intervals
    # ------------------------------------------------------------------
    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute bootstrap CIs for accuracy, f1, precision, recall, AUC."""
        alpha = 1 - self.ci_level
        n = len(y_true)
        rng = np.random.RandomState(42)

        metrics_lists: Dict[str, list] = defaultdict(list)

        for _ in range(self.bootstrap_n):
            idx = rng.choice(n, size=n, replace=True)
            yt, yp = y_true[idx], y_pred[idx]

            metrics_lists["accuracy"].append(accuracy_score(yt, yp))
            metrics_lists["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
            metrics_lists["precision_macro"].append(precision_score(yt, yp, average="macro", zero_division=0))
            metrics_lists["recall_macro"].append(recall_score(yt, yp, average="macro", zero_division=0))

            if y_prob is not None:
                try:
                    from sklearn.preprocessing import label_binarize
                    yb = label_binarize(yt, classes=list(range(self.num_classes)))
                    if yb.shape[1] == 1:
                        yb = np.hstack([1 - yb, yb])
                    yprob = y_prob[idx]
                    metrics_lists["auc_macro"].append(
                        roc_auc_score(yb, yprob, average="macro", multi_class="ovr")
                    )
                except Exception:
                    pass

        ci_results = {}
        for name, values in metrics_lists.items():
            arr = np.array(values)
            lo = np.percentile(arr, 100 * alpha / 2)
            hi = np.percentile(arr, 100 * (1 - alpha / 2))
            ci_results[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "ci_level": self.ci_level,
            }
        return ci_results

    # ------------------------------------------------------------------
    # Fold aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
        """
        Aggregate metrics across k folds, computing mean ± std.

        Args:
            fold_results: List of per-fold metric dictionaries.

        Returns:
            Aggregated results with mean, std, and individual fold values.
        """
        scalar_keys = [
            "accuracy", "f1_macro", "f1_weighted",
            "precision_macro", "precision_weighted",
            "recall_macro", "recall_weighted",
            "auc_macro", "auc_weighted",
        ]

        aggregated: Dict = {"n_folds": len(fold_results)}

        for key in scalar_keys:
            values = []
            for fr in fold_results:
                v = fr.get(key)
                if v is not None:
                    values.append(v)
            if values:
                arr = np.array(values)
                aggregated[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "values": [float(v) for v in arr],
                }

        # aggregate timing
        for time_key in ["train_time", "eval_time", "epoch_times"]:
            values = [fr.get(time_key) for fr in fold_results if time_key in fr]
            if values:
                if isinstance(values[0], list):
                    # epoch_times: list of lists
                    aggregated[time_key] = {
                        "per_fold": values,
                        "mean_per_epoch": np.mean(values, axis=0).tolist() if values else [],
                    }
                else:
                    arr = np.array(values, dtype=float)
                    aggregated[time_key] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                    }

        # keep individual fold results
        aggregated["folds"] = fold_results

        return aggregated


class TimeTracker:
    """Track training / evaluation / per-epoch timing."""

    def __init__(self):
        self.epoch_times: List[float] = []
        self.train_start: float = 0.0
        self.train_end: float = 0.0
        self.eval_start: float = 0.0
        self.eval_end: float = 0.0
        self._epoch_start: float = 0.0

    def start_training(self):
        self.train_start = time.time()

    def end_training(self):
        self.train_end = time.time()

    def start_epoch(self):
        self._epoch_start = time.time()

    def end_epoch(self):
        self.epoch_times.append(time.time() - self._epoch_start)

    def start_eval(self):
        self.eval_start = time.time()

    def end_eval(self):
        self.eval_end = time.time()

    def summary(self) -> Dict:
        return {
            "total_train_time_sec": self.train_end - self.train_start,
            "total_eval_time_sec": self.eval_end - self.eval_start,
            "epoch_times_sec": self.epoch_times,
            "mean_epoch_time_sec": float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
            "total_run_time_sec": (self.eval_end or self.train_end) - self.train_start,
        }
