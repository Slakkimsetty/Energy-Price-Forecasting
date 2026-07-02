"""
MLflow Experiment Tracking
============================
Logs walk-forward validation results, model parameters, and drift metrics
to a local MLflow tracking server. Run `mlflow ui` to view the dashboard.

Each pipeline run creates:
  - A parent run with summary metrics + params
  - One child run per WFV fold with per-fold RMSE/MAE/MAPE
  - Saved artifacts: feature importance CSV, drift report CSV
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    LIGHTGBM_PARAMS, STRUCTURAL_BREAK_DATE,
)


def setup_mlflow() -> None:
    """Initialize MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    print(f"✅ MLflow tracking → {MLFLOW_TRACKING_URI}")
    print(f"   Experiment: {MLFLOW_EXPERIMENT}")


def log_wfv_run(
    fold_metrics:   list[dict],
    drift_summary:  pd.DataFrame,
    feature_importance: pd.DataFrame,
    model,
    feature_names:  list[str],
    run_name:       str = "wfv_pipeline",
) -> str:
    """
    Log a complete WFV pipeline run to MLflow.

    Returns the run ID.
    """
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as parent_run:
        run_id = parent_run.info.run_id

        # ── Parameters ────────────────────────────────────────────────────────
        mlflow.log_params({
            "model_type":         "LightGBM",
            "n_features":         len(feature_names),
            "n_folds":            len(fold_metrics),
            "structural_break":   STRUCTURAL_BREAK_DATE,
            **{f"lgbm_{k}": v for k, v in LIGHTGBM_PARAMS.items()},
        })

        # ── Aggregate metrics ─────────────────────────────────────────────────
        avg_rmse = float(np.mean([m["rmse"] for m in fold_metrics]))
        avg_mae  = float(np.mean([m["mae"]  for m in fold_metrics]))
        avg_mape = float(np.mean([m["mape"] for m in fold_metrics]))
        mlflow.log_metrics({
            "avg_rmse": avg_rmse,
            "avg_mae":  avg_mae,
            "avg_mape": avg_mape,
        })

        # ── Drift metrics ──────────────────────────────────────────────────────
        if not drift_summary.empty:
            pre  = drift_summary[drift_summary["is_post_break"] == 0]
            post = drift_summary[drift_summary["is_post_break"] == 1]
            if not pre.empty:
                mlflow.log_metric("drift_pre_break_avg_psi",  pre["mean_psi"].mean())
                mlflow.log_metric("drift_pre_break_max_psi",  pre["max_psi"].max())
            if not post.empty:
                mlflow.log_metric("drift_post_break_avg_psi", post["mean_psi"].mean())
                mlflow.log_metric("drift_post_break_max_psi", post["max_psi"].max())
                mlflow.log_metric("drift_post_critical_windows", int((post["n_critical"] > 0).sum()))

        # ── Artifacts ─────────────────────────────────────────────────────────
        # Feature importance
        fi_path = "/tmp/feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path, "reports")

        # Drift report
        dr_path = "/tmp/drift_report.csv"
        drift_summary.to_csv(dr_path, index=False)
        mlflow.log_artifact(dr_path, "reports")

        # Fold metrics JSON
        fm_path = "/tmp/fold_metrics.json"
        with open(fm_path, "w") as f:
            json.dump(fold_metrics, f, indent=2)
        mlflow.log_artifact(fm_path, "reports")

        # Log the LightGBM model itself
        mlflow.lightgbm.log_model(model, "lgbm_model")

        # ── Per-fold child runs ───────────────────────────────────────────────
        for fm in fold_metrics:
            with mlflow.start_run(run_name=f"fold_{fm['fold']:03d}", nested=True):
                mlflow.log_metrics({
                    "rmse":       fm["rmse"],
                    "mae":        fm["mae"],
                    "mape":       fm["mape"],
                    "train_size": fm["train_size"],
                })
                mlflow.log_params({
                    "test_start": fm["test_start"],
                    "test_end":   fm["test_end"],
                })

        print(f"\n✅ MLflow run logged")
        print(f"   Run ID:   {run_id}")
        print(f"   Avg RMSE: {avg_rmse:.2f} $/MWh")
        print(f"   Avg MAPE: {avg_mape:.2f}%")
        print(f"\n   View dashboard: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

        return run_id


if __name__ == "__main__":
    print("MLflow tracking module — run pipeline.py to log a full run.")
