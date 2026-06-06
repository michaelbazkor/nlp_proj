"""Evaluation metrics for pipeline predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)


def auc_to_cohens_d(auc: float) -> float:
    """Transform AUC to Cohen's d (Salgado 2018 approximation)."""
    from math import sqrt

    return sqrt(2) * _norm_ppf(auc)


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF."""
    from math import erfc, sqrt

    p = min(max(p, 1e-6), 1 - 1e-6)
    return sqrt(2) * _erfinv(2 * p - 1)


def _erfinv(x: float) -> float:
    """Approximate inverse error function (Winitzki)."""
    from math import log, sqrt

    a = 0.147
    sign = 1 if x >= 0 else -1
    x = abs(x)
    ln = log(1 - x * x)
    first = 2 / (np.pi * a) + ln / 2
    second = ln / a
    return sign * sqrt(sqrt(first * first - second) - first)


def evaluate_results(
    results_path: str | Path,
    binary_sd_threshold: int = 1,
) -> dict[str, Any]:
    """Compute metrics comparing predictions to ground truth."""
    df = pd.read_csv(results_path)
    report: dict[str, Any] = {"n_users": len(df), "metrics": {}}

    continuous_pairs = [
        ("FOMO", "pred_FOMO"),
        ("BFI_N", "pred_BFI_N"),
        ("Lonely", "pred_Lonely"),
        ("Brooding", "pred_Brooding"),
        ("Worry", "pred_Worry"),
        ("SWL", "pred_SWL"),
    ]

    for true_col, pred_col in continuous_pairs:
        t_col = f"true_{true_col}"
        if t_col not in df.columns or pred_col not in df.columns:
            continue
        y_true = df[t_col].astype(float)
        y_pred = df[pred_col].astype(float)
        mask = y_true.notna() & y_pred.notna()
        if mask.sum() < 2:
            continue
        pearson = float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])
        mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
        report["metrics"][true_col] = {"pearson_r": pearson, "mae": mae}

    if "true_MDD" in df.columns and "pred_MDD" in df.columns:
        y_true = df["true_MDD"].astype(int)
        y_pred = df["pred_MDD"].astype(int)
        report["metrics"]["MDD"] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

    if "true_SD" in df.columns and "pred_SD" in df.columns:
        y_true_bin = (df["true_SD"] >= binary_sd_threshold).astype(int)
        y_pred_bin = (df["pred_SD"] >= binary_sd_threshold).astype(int)
        y_pred_score = df["pred_SD"].astype(float)

        try:
            auc = float(roc_auc_score(y_true_bin, y_pred_score))
            cohens_d = auc_to_cohens_d(auc)
        except ValueError:
            auc = None
            cohens_d = None

        report["metrics"]["SD_binary"] = {
            "threshold": binary_sd_threshold,
            "auc": auc,
            "cohens_d": cohens_d,
            "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
            "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        }
        report["metrics"]["SD_continuous"] = {
            "mae": float(mean_absolute_error(df["true_SD"], df["pred_SD"])),
        }

    return report


def save_evaluation_report(
    results_path: str | Path,
    output_dir: str | Path,
    binary_sd_threshold: int = 1,
) -> Path:
    """Evaluate and save JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_results(results_path, binary_sd_threshold=binary_sd_threshold)
    out_path = output_dir / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved: {out_path}")
    return out_path
