from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftThresholds:
    mean_shift_std: float = 0.50
    missing_rate_shift: float = 0.10
    categorical_l1_shift: float = 0.30
    psi: float = 0.20


def _psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna()
    cur = pd.to_numeric(current, errors="coerce").dropna()
    if ref.empty or cur.empty or ref.nunique() < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(1, ref_counts.sum()), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(1, cur_counts.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _categorical_l1(reference: pd.Series, current: pd.Series) -> float:
    ref_dist = reference.fillna("__missing__").astype(str).value_counts(normalize=True)
    cur_dist = current.fillna("__missing__").astype(str).value_counts(normalize=True)
    categories = set(ref_dist.index).union(cur_dist.index)
    return float(
        sum(abs(cur_dist.get(cat, 0.0) - ref_dist.get(cat, 0.0)) for cat in categories)
    )


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    thresholds: DriftThresholds | None = None,
) -> dict:
    thresholds = thresholds or DriftThresholds()
    numeric_results = {}
    categorical_results = {}
    alerts = []

    for feature in numeric_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref = pd.to_numeric(reference_df[feature], errors="coerce")
        cur = pd.to_numeric(current_df[feature], errors="coerce")
        ref_mean = float(ref.mean()) if not math.isnan(ref.mean()) else 0.0
        cur_mean = float(cur.mean()) if not math.isnan(cur.mean()) else 0.0
        ref_std = float(ref.std()) if ref.std() and not math.isnan(ref.std()) else 1.0
        mean_shift_std = abs(cur_mean - ref_mean) / max(ref_std, 1e-9)
        missing_shift = abs(float(cur.isna().mean()) - float(ref.isna().mean()))
        psi_value = _psi(ref, cur)
        flagged = (
            mean_shift_std > thresholds.mean_shift_std
            or missing_shift > thresholds.missing_rate_shift
            or psi_value > thresholds.psi
        )
        numeric_results[feature] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "mean_shift_std": float(mean_shift_std),
            "missing_rate_shift": float(missing_shift),
            "psi": psi_value,
            "alert": flagged,
        }
        if flagged:
            alerts.append(
                {
                    "feature": feature,
                    "type": "numeric_drift",
                    "details": numeric_results[feature],
                }
            )

    for feature in categorical_features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        missing_shift = abs(
            float(current_df[feature].isna().mean())
            - float(reference_df[feature].isna().mean())
        )
        l1_shift = _categorical_l1(reference_df[feature], current_df[feature])
        flagged = (
            missing_shift > thresholds.missing_rate_shift
            or l1_shift > thresholds.categorical_l1_shift
        )
        categorical_results[feature] = {
            "l1_distribution_shift": l1_shift,
            "missing_rate_shift": missing_shift,
            "alert": flagged,
        }
        if flagged:
            alerts.append(
                {
                    "feature": feature,
                    "type": "categorical_drift",
                    "details": categorical_results[feature],
                }
            )

    return {
        "summary": {
            "reference_rows": int(len(reference_df)),
            "current_rows": int(len(current_df)),
            "alert_count": len(alerts),
            "has_alerts": bool(alerts),
            "thresholds": thresholds.__dict__,
        },
        "numeric": numeric_results,
        "categorical": categorical_results,
        "alerts": alerts,
    }
