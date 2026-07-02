"""
Drift Detection via Population Stability Index (PSI)
======================================================
PSI measures how much a feature's distribution has shifted between a reference
period and a current window. It's the industry standard in financial ML for
detecting model degradation before it shows up in performance metrics.

PSI interpretation (industry standard thresholds):
  PSI < 0.10  → No significant change, model stable
  PSI 0.10–0.20 → Moderate shift, investigate
  PSI > 0.20  → Significant shift, retrain required

In this project we also track:
  - Prediction distribution drift (did the model's outputs shift?)
  - Rolling RMSE (did performance actually degrade?)
  - A combined "drift score" that triggers an alert

The structural break in the synthetic data (STRUCTURAL_BREAK_DATE) should
cause PSI alerts to fire within 1–2 rolling windows after the break.
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PSI_THRESHOLD_WARNING, PSI_THRESHOLD_CRITICAL,
    PSI_BINS, DRIFT_REFERENCE_DAYS, DRIFT_WINDOW_DAYS,
    STRUCTURAL_BREAK_DATE,
)


# ── PSI Core ──────────────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    Compute PSI between reference and current distributions.

    PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

    Parameters
    ----------
    reference : Samples from reference (stable) period
    current   : Samples from current window to evaluate
    bins      : Number of equal-width bins (default 10 per industry standard)

    Returns
    -------
    psi_value : float
    """
    # Use reference distribution to define bin edges
    # (bins defined by reference are the correct approach)
    eps = 1e-6   # avoid log(0)
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())

    bin_edges  = np.linspace(min_val, max_val + eps, bins + 1)

    ref_counts, _  = np.histogram(reference, bins=bin_edges)
    cur_counts, _  = np.histogram(current,   bins=bin_edges)

    # Convert to proportions
    ref_pct = ref_counts / (len(reference) + eps) + eps
    cur_pct = cur_counts / (len(current)   + eps) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def psi_status(psi_value: float) -> str:
    if psi_value < PSI_THRESHOLD_WARNING:
        return "✅ STABLE"
    elif psi_value < PSI_THRESHOLD_CRITICAL:
        return "⚠️  WARNING"
    else:
        return "🚨 CRITICAL"


# ── Rolling Drift Monitor ──────────────────────────────────────────────────────

def rolling_drift_report(
    feature_df: pd.DataFrame,
    feature_names: list[str],
    predictions_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute rolling PSI for each feature over time.

    For each rolling window of DRIFT_WINDOW_DAYS, compare the feature
    distribution against the reference period (first DRIFT_REFERENCE_DAYS).

    Parameters
    ----------
    feature_df     : Full feature DataFrame with timestamp
    feature_names  : Feature columns to monitor
    predictions_df : Optional predictions DataFrame (to include pred drift)

    Returns
    -------
    drift_df : DataFrame with one row per (window_date, feature), PSI value, and status
    """
    df = feature_df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    all_dates = sorted(df["date"].unique())

    # Reference period: first DRIFT_REFERENCE_DAYS of data
    reference_dates = set(all_dates[:DRIFT_REFERENCE_DAYS])
    ref_mask        = df["date"].isin(reference_dates)
    ref_df          = df[ref_mask]

    # Rolling windows (step by 7 days)
    step        = 7
    records     = []
    window_end  = DRIFT_REFERENCE_DAYS + DRIFT_WINDOW_DAYS

    while window_end <= len(all_dates):
        window_dates = all_dates[window_end - DRIFT_WINDOW_DAYS : window_end]
        cur_mask     = df["date"].isin(window_dates)
        cur_df       = df[cur_mask]
        window_date  = window_dates[-1]  # label by end of window

        # Monitor a subset of key features (PSI for all 43 would be noisy)
        key_features = [f for f in feature_names if any(
            kw in f for kw in ["price_lag", "price_roll_mean", "temperature", "load_mw", "price_range"]
        )]

        for feat in key_features:
            ref_vals = ref_df[feat].dropna().values
            cur_vals = cur_df[feat].dropna().values

            if len(ref_vals) < 30 or len(cur_vals) < 30:
                continue

            psi = compute_psi(ref_vals, cur_vals)
            records.append({
                "window_date":    window_date,
                "feature":        feat,
                "psi":            round(psi, 4),
                "status":         psi_status(psi),
                "is_post_break":  int(str(window_date) >= STRUCTURAL_BREAK_DATE),
            })

        window_end += step

    drift_df = pd.DataFrame(records)
    return drift_df


def summarize_drift(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate PSI by date: return max PSI, mean PSI, and # of critical features
    for each rolling window. This is the "dashboard view" of drift over time.
    """
    summary = (
        drift_df.groupby("window_date")
        .agg(
            max_psi      = ("psi", "max"),
            mean_psi     = ("psi", "mean"),
            n_warning    = ("status", lambda s: (s == "⚠️  WARNING").sum()),
            n_critical   = ("status", lambda s: (s == "🚨 CRITICAL").sum()),
            is_post_break= ("is_post_break", "max"),
        )
        .reset_index()
    )
    summary["overall_status"] = summary["max_psi"].apply(psi_status)
    return summary


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Kolmogorov-Smirnov test as a second-opinion on drift.
    Returns test statistic and p-value. p < 0.05 = distributions differ.
    """
    stat, pval = stats.ks_2samp(reference, current)
    return {"ks_stat": round(stat, 4), "ks_pval": round(pval, 4), "drifted": pval < 0.05}


def first_alert_date(summary_df: pd.DataFrame) -> str | None:
    """Return the first date a WARNING or CRITICAL status fires after the break."""
    post = summary_df[summary_df["is_post_break"] == 1]
    alerts = post[(post["n_warning"] > 0) | (post["n_critical"] > 0)]
    if alerts.empty:
        return None
    return str(alerts["window_date"].min())


if __name__ == "__main__":
    from data.synthetic import generate
    from features.engineer import build_feature_matrix, get_feature_df_with_meta

    print("Generating synthetic data...")
    raw = generate(save=False)

    print("Engineering features...")
    X, y, feat_names = build_feature_matrix(raw)
    feat_df = get_feature_df_with_meta(raw)

    print("Running drift detection...")
    drift_df   = rolling_drift_report(feat_df, feat_names)
    summary_df = summarize_drift(drift_df)

    alert_date = first_alert_date(summary_df)

    print(f"\n  Structural break: {STRUCTURAL_BREAK_DATE}")
    print(f"  First drift alert: {alert_date}")
    print(f"\n  Drift Summary (last 10 windows):")
    print(summary_df.tail(10).to_string(index=False))

    # Show most drifted features
    top_drift = (
        drift_df.groupby("feature")["psi"].max()
        .sort_values(ascending=False)
        .head(8)
    )
    print(f"\n  Top drifted features (max PSI):")
    for feat, psi in top_drift.items():
        print(f"    {feat:40s}  PSI={psi:.4f}  {psi_status(psi)}")
