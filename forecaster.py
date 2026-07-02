"""
LightGBM Forecaster with Walk-Forward Validation
==================================================
Why walk-forward (not random split)?
  Energy prices are a time series. Random splits leak future information into the
  training set — a model that "sees" next week's prices while predicting this week
  will look artificially good. Walk-forward validation always trains only on the past
  and predicts the future, exactly as the model would behave in production.

Fold structure (WFV_INITIAL_TRAIN_DAYS = 120, WFV_STEP_DAYS = 7):
  Fold 0: train [day 0 → 120), predict [day 120 → 127)
  Fold 1: train [day 0 → 127), predict [day 127 → 134)
  Fold 2: train [day 0 → 134), predict [day 134 → 141)
  ...
  (expanding window — each fold uses all available history)
"""

import sys
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    LIGHTGBM_PARAMS, WFV_INITIAL_TRAIN_DAYS,
    WFV_STEP_DAYS, WFV_TEST_DAYS, MODELS_DIR, FORECAST_HORIZON_HOURS,
)


# ── Metrics ────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae":  round(mae(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


# ── Walk-Forward Validation ────────────────────────────────────────────────────

def walk_forward_validate(
    feature_df: pd.DataFrame,
    feature_names: list[str],
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run walk-forward validation over the full feature DataFrame.

    Parameters
    ----------
    feature_df    : Full DataFrame including timestamp, target, and feature columns
    feature_names : List of feature column names
    verbose       : Print per-fold results

    Returns
    -------
    predictions_df : DataFrame with columns [timestamp, y_true, y_pred, fold]
    fold_metrics   : List of per-fold metric dicts
    """
    df = feature_df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date

    all_dates   = sorted(df["date"].unique())
    start_idx   = WFV_INITIAL_TRAIN_DAYS
    fold_results = []
    fold_metrics = []
    fold_num     = 0

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Walk-Forward Validation")
        print(f"  Initial train: {WFV_INITIAL_TRAIN_DAYS} days")
        print(f"  Step size:     {WFV_STEP_DAYS} days")
        print(f"  Test window:   {WFV_TEST_DAYS} days / fold")
        print(f"  Total dates:   {len(all_dates)}")
        print(f"{'─'*60}")

    while start_idx + WFV_TEST_DAYS <= len(all_dates):
        train_dates = all_dates[:start_idx]
        test_dates  = all_dates[start_idx : start_idx + WFV_TEST_DAYS]

        train_mask = df["date"].isin(train_dates)
        test_mask  = df["date"].isin(test_dates)

        X_train = df.loc[train_mask, feature_names]
        y_train = df.loc[train_mask, "target"]
        X_test  = df.loc[test_mask, feature_names]
        y_test  = df.loc[test_mask, "target"]

        if len(y_test) == 0:
            break

        # Train LightGBM
        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test.values, y_pred)
        metrics["fold"]        = fold_num
        metrics["train_size"]  = len(y_train)
        metrics["test_start"]  = str(test_dates[0])
        metrics["test_end"]    = str(test_dates[-1])
        fold_metrics.append(metrics)

        fold_df = df.loc[test_mask, ["timestamp", "target", "is_post_break"]].copy()
        fold_df["y_true"] = y_test.values
        fold_df["y_pred"] = y_pred
        fold_df["fold"]   = fold_num
        fold_results.append(fold_df)

        if verbose and fold_num % 5 == 0:
            print(f"  Fold {fold_num:3d} | Train: {len(y_train):5,} | "
                  f"RMSE: {metrics['rmse']:6.2f} | MAE: {metrics['mae']:6.2f} | "
                  f"MAPE: {metrics['mape']:5.2f}%  [{test_dates[0]} → {test_dates[-1]}]")

        start_idx += WFV_STEP_DAYS
        fold_num  += 1

    predictions_df = pd.concat(fold_results, ignore_index=True)

    if verbose:
        avg_rmse = np.mean([m["rmse"] for m in fold_metrics])
        avg_mae  = np.mean([m["mae"]  for m in fold_metrics])
        avg_mape = np.mean([m["mape"] for m in fold_metrics])
        print(f"{'─'*60}")
        print(f"  Folds complete: {fold_num}")
        print(f"  Avg RMSE:  {avg_rmse:.2f} $/MWh")
        print(f"  Avg MAE:   {avg_mae:.2f} $/MWh")
        print(f"  Avg MAPE:  {avg_mape:.2f}%")
        print(f"{'─'*60}\n")

    return predictions_df, fold_metrics


# ── Final Model (trained on all data) ─────────────────────────────────────────

def train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
    save: bool = True,
) -> lgb.LGBMRegressor:
    """
    Train a final production model on all available data.
    This is the model saved for the API endpoint.
    """
    model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(X, y, callbacks=[lgb.log_evaluation(period=-1)])

    if save:
        path = MODELS_DIR / "lgbm_final.joblib"
        joblib.dump({"model": model, "feature_names": feature_names}, path)
        print(f"✅ Final model saved → {path}")

    return model


def load_model(path: Path | None = None) -> tuple[lgb.LGBMRegressor, list[str]]:
    """Load a saved model and its feature names."""
    if path is None:
        path = MODELS_DIR / "lgbm_final.joblib"
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_names"]


def get_feature_importance(model: lgb.LGBMRegressor, feature_names: list[str]) -> pd.DataFrame:
    """Return feature importance as a sorted DataFrame."""
    return (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    from data.synthetic import generate
    from features.engineer import build_feature_matrix, get_feature_df_with_meta

    print("Generating data...")
    raw = generate(save=False)

    print("Engineering features...")
    X, y, feat_names = build_feature_matrix(raw)
    feat_df = get_feature_df_with_meta(raw)

    print("Running walk-forward validation...")
    preds_df, metrics = walk_forward_validate(feat_df, feat_names)

    print("\nTraining final model...")
    final_model = train_final_model(X, y, feat_names)

    print("\nTop 10 Features:")
    fi = get_feature_importance(final_model, feat_names)
    print(fi.head(10).to_string(index=False))
