"""
Feature Engineering
====================
Transforms raw hourly price/weather data into a feature matrix for LightGBM.

Features created:
  Calendar    : hour, day_of_week, month, quarter, is_weekend, is_peak_hour
  Lag         : price at t-1h, t-2h, t-3h, t-6h, t-12h, t-24h, t-48h, t-168h
  Rolling     : mean/std of price over 6h, 24h, 48h, 168h windows
  Interaction : temp × is_peak_hour, load × hour_sin/cos
  Cyclical    : sin/cos encoding of hour-of-day and day-of-year (avoids ordinality)
  Weather     : temperature_f, load_mw
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LAG_HOURS, ROLLING_WINDOWS, FORECAST_HORIZON_HOURS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features from the timestamp column."""
    ts = df["timestamp"]

    df["hour"]         = ts.dt.hour
    df["day_of_week"]  = ts.dt.dayofweek
    df["month"]        = ts.dt.month
    df["quarter"]      = ts.dt.quarter
    df["day_of_year"]  = ts.dt.dayofyear
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Peak hours = 7–9 AM and 5–8 PM (when prices are highest)
    df["is_morning_peak"] = df["hour"].between(7, 9).astype(int)
    df["is_evening_peak"] = df["hour"].between(17, 20).astype(int)
    df["is_peak_hour"]    = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)
    df["is_overnight"]    = df["hour"].between(0, 5).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features as (sin, cos) pairs so the model understands
    that hour 23 and hour 0 are adjacent — a plain integer doesn't convey that.
    """
    df["hour_sin"]       = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]       = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]        = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]        = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]      = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]      = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]        = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]        = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = LAG_HOURS) -> pd.DataFrame:
    """
    Add lagged price values.
    IMPORTANT: We lag by at least FORECAST_HORIZON_HOURS to prevent leakage —
    when predicting t+24, we can only use data available up to t.
    """
    adjusted_lags = [max(lag, FORECAST_HORIZON_HOURS) for lag in lags]
    for lag in adjusted_lags:
        df[f"price_lag_{lag}h"] = df["price_mwh"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Rolling mean and std of price over multiple windows.
    min_periods=1 avoids NaN at the start of the series.
    """
    for w in windows:
        shifted = df["price_mwh"].shift(FORECAST_HORIZON_HOURS)
        df[f"price_roll_mean_{w}h"] = shifted.rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{w}h"]  = shifted.rolling(w, min_periods=1).std().fillna(0)
        df[f"price_roll_max_{w}h"]  = shifted.rolling(w, min_periods=1).max()
        df[f"price_roll_min_{w}h"]  = shifted.rolling(w, min_periods=1).min()
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-feature interactions that encode domain knowledge."""
    # Temperature × peak hour: hot peak hours are the most expensive
    df["temp_x_peak"]     = df["temperature_f"] * df["is_peak_hour"]
    # Load × evening peak
    df["load_x_evening"]  = df["load_mw"] * df["is_evening_peak"]
    # Price spread: rolling range indicates market volatility
    if "price_roll_max_24h" in df.columns and "price_roll_min_24h" in df.columns:
        df["price_range_24h"] = df["price_roll_max_24h"] - df["price_roll_min_24h"]

    return df


def build_feature_matrix(df: pd.DataFrame, target_col: str = "price_mwh") -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Full pipeline: raw DataFrame → (X, y, feature_names).

    Parameters
    ----------
    df         : Raw DataFrame with timestamp, price_mwh, temperature_f, load_mw columns
    target_col : Column to predict

    Returns
    -------
    X              : Feature DataFrame (NaN rows dropped)
    y              : Target Series aligned with X
    feature_names  : List of feature column names
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)

    # Define the target: price 24 hours ahead
    df["target"] = df[target_col].shift(-FORECAST_HORIZON_HOURS)

    # Drop rows where we don't have enough history or future
    df = df.dropna().reset_index(drop=True)

    # Feature columns = everything except metadata and target
    exclude = {"timestamp", "price_mwh", "target", "is_post_break"}
    feature_names = [c for c in df.columns if c not in exclude]

    X = df[feature_names]
    y = df["target"]

    return X, y, feature_names


def get_feature_df_with_meta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as build_feature_matrix but returns a single DataFrame with
    timestamp and is_post_break columns retained (useful for drift analysis).
    """
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_interaction_features(df)
    df["target"] = df["price_mwh"].shift(-FORECAST_HORIZON_HOURS)
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick smoke test
    import sys
    sys.path.insert(0, "..")
    from data.synthetic import generate

    raw = generate(save=False)
    X, y, feat_names = build_feature_matrix(raw)
    print(f"✅ Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"   Target range:  ${y.min():.1f} – ${y.max():.1f}/MWh")
    print(f"\n   Features:\n   " + "\n   ".join(feat_names))
