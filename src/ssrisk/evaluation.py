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
    binary_sd_positive_min: int = 5,
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
        mask = df["true_MDD"].notna() & df["pred_MDD"].notna()
        if mask.sum() >= 1:
            y_true = df.loc[mask, "true_MDD"].astype(int)
            y_pred = df.loc[mask, "pred_MDD"].astype(int)
            report["metrics"]["MDD"] = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }

    if "true_SD" in df.columns and "pred_SD" in df.columns:
        sd_mask = df["true_SD"].notna() & df["pred_SD"].notna()
        if sd_mask.sum() >= 1:
            y_true_sd = df.loc[sd_mask, "true_SD"].astype(float)
            y_pred_sd = df.loc[sd_mask, "pred_SD"].astype(float)
            y_true_bin = (y_true_sd >= binary_sd_positive_min).astype(int)
            y_pred_bin = (y_pred_sd >= binary_sd_positive_min).astype(int)

            try:
                auc = float(roc_auc_score(y_true_bin, y_pred_sd)) if sd_mask.sum() >= 2 else None
                cohens_d = auc_to_cohens_d(auc) if auc is not None else None
            except ValueError:
                auc = None
                cohens_d = None

            report["metrics"]["SD_binary"] = {
                "positive_min": binary_sd_positive_min,
                "positive_range": "[5, 6]",
                "negative_range": "[0, 4]",
                "auc": auc,
                "cohens_d": cohens_d,
                "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
            }
            report["metrics"]["SD_continuous"] = {
                "mae": float(mean_absolute_error(y_true_sd, y_pred_sd)),
            }

    return report


def save_evaluation_report(
    results_path: str | Path,
    output_dir: str | Path,
    binary_sd_positive_min: int = 5,
) -> Path:
    """Evaluate and save JSON report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate_results(results_path, binary_sd_positive_min=binary_sd_positive_min)
    out_path = output_dir / "evaluation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved: {out_path}")
    return out_path
