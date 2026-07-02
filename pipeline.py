"""
Energy Price Forecaster — End-to-End Pipeline
==============================================
Run this to execute the full pipeline:
  1. Generate synthetic ERCOT-style price data
  2. Engineer features (lags, rolling stats, calendar, cyclical encoding)
  3. Walk-forward validation with LightGBM (no future leakage)
  4. Detect distributional drift via PSI (catches the structural break)
  5. Log everything to MLflow
  6. Save visualizations to /plots/

Usage:
  python pipeline.py                   # Full run
  python pipeline.py --no-mlflow       # Skip MLflow logging
  python pipeline.py --no-plots        # Skip plot generation

Then:
  mlflow ui --backend-store-uri mlruns  # View experiment dashboard
  python -m uvicorn api.main:app        # Start prediction API
"""

import argparse
import time
import sys
import pandas as pd

print("=" * 65)
print("  Energy Price Forecaster — Pipeline")
print("=" * 65)

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
parser.add_argument("--no-plots",  action="store_true", help="Skip plot generation")
parser.add_argument("--seed",      type=int, default=42, help="Random seed for data generation")
args = parser.parse_args()

# ── Step 1: Generate Data ────────────────────────────────────────────────────
print("\n[1/6] Generating synthetic ERCOT price data...")
t0 = time.time()

from data.synthetic import generate
raw_df = generate(seed=args.seed, save=True)
print(f"      Done in {time.time()-t0:.1f}s")

# ── Step 2: Feature Engineering ──────────────────────────────────────────────
print("\n[2/6] Engineering features...")
t0 = time.time()

from features.engineer import build_feature_matrix, get_feature_df_with_meta
X, y, feature_names = build_feature_matrix(raw_df)
feature_df          = get_feature_df_with_meta(raw_df)

print(f"      {X.shape[0]:,} samples × {X.shape[1]} features")
print(f"      Done in {time.time()-t0:.1f}s")

# ── Step 3: Walk-Forward Validation ──────────────────────────────────────────
print("\n[3/6] Running walk-forward validation...")
t0 = time.time()

from models.forecaster import walk_forward_validate, train_final_model, get_feature_importance
import numpy as np

predictions_df, fold_metrics = walk_forward_validate(feature_df, feature_names, verbose=True)

wfv_elapsed = time.time() - t0
avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
avg_mape = np.mean([m["mape"] for m in fold_metrics])
print(f"      {len(fold_metrics)} folds  |  Avg RMSE: {avg_rmse:.2f} $/MWh  |  Avg MAPE: {avg_mape:.2f}%")
print(f"      Done in {wfv_elapsed:.1f}s")

# ── Step 4: Train Final Model ─────────────────────────────────────────────────
print("\n[4/6] Training final model on all data...")
t0 = time.time()

final_model       = train_final_model(X, y, feature_names, save=True)
feature_importance = get_feature_importance(final_model, feature_names)

print(f"\n      Top 5 Features:")
for _, row in feature_importance.head(5).iterrows():
    print(f"        {row['feature']:35s} {row['importance']:6,}")
print(f"      Done in {time.time()-t0:.1f}s")

# ── Step 5: Drift Detection ───────────────────────────────────────────────────
print("\n[5/6] Running drift detection (PSI)...")
t0 = time.time()

from drift.detector import rolling_drift_report, summarize_drift, first_alert_date
from config import STRUCTURAL_BREAK_DATE

drift_df   = rolling_drift_report(feature_df, feature_names)
summary_df = summarize_drift(drift_df)
alert_date = first_alert_date(summary_df)

pre_break_max_psi  = summary_df[summary_df["is_post_break"] == 0]["max_psi"].max()
post_break_max_psi = summary_df[summary_df["is_post_break"] == 1]["max_psi"].max()

print(f"\n      Structural break date:   {STRUCTURAL_BREAK_DATE}")
print(f"      First PSI alert fired:   {alert_date}")
print(f"      Pre-break max PSI:       {pre_break_max_psi:.4f}  (stable baseline)")
print(f"      Post-break max PSI:      {post_break_max_psi:.4f}  (drift confirmed)")
print(f"      Done in {time.time()-t0:.1f}s")

# ── Step 6a: MLflow Logging ───────────────────────────────────────────────────
if not args.no_mlflow:
    print("\n[6a/6] Logging to MLflow...")
    t0 = time.time()
    try:
        from tracking.mlflow_logger import log_wfv_run
        run_id = log_wfv_run(
            fold_metrics        = fold_metrics,
            drift_summary       = summary_df,
            feature_importance  = feature_importance,
            model               = final_model,
            feature_names       = feature_names,
            run_name            = "wfv_lgbm_pipeline",
        )
        print(f"      Done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"      ⚠️  MLflow logging failed: {e}")
else:
    print("\n[6a/6] Skipping MLflow (--no-mlflow)")

# ── Step 6b: Visualizations ───────────────────────────────────────────────────
if not args.no_plots:
    print("\n[6b/6] Generating visualizations...")
    t0 = time.time()
    try:
        from visualizations import (
            plot_price_series, plot_wfv_results,
            plot_drift_dashboard, plot_feature_importance,
            plot_intraday_patterns,
        )
        print("  Plotting...")
        plot_price_series(raw_df)
        plot_wfv_results(predictions_df, fold_metrics)
        plot_drift_dashboard(summary_df, drift_df)
        plot_feature_importance(feature_importance)
        plot_intraday_patterns(raw_df)
        print(f"      Done in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"      ⚠️  Plotting failed: {e}")
else:
    print("\n[6b/6] Skipping plots (--no-plots)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Pipeline Complete")
print("=" * 65)
print(f"  Data:          {len(raw_df):,} hourly rows (2 years)")
print(f"  Features:      {len(feature_names)}")
print(f"  WFV Folds:     {len(fold_metrics)}")
print(f"  Avg RMSE:      {avg_rmse:.2f} $/MWh")
print(f"  Avg MAPE:      {avg_mape:.2f}%")
print(f"  Break date:    {STRUCTURAL_BREAK_DATE}")
print(f"  First alert:   {alert_date}  ({(pd.Timestamp(alert_date) - pd.Timestamp(STRUCTURAL_BREAK_DATE)).days} days after break)")
print()
print("  Next steps:")
print("    mlflow ui --backend-store-uri mlruns")
print("    python -m uvicorn api.main:app --reload")
print("=" * 65)
