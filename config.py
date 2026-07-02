"""
Central configuration for the Energy Price Forecaster project.
All tunable knobs live here — no magic numbers scattered through the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR    = Path(__file__).parent
DATA_DIR    = ROOT_DIR / "data" / "raw"
MODELS_DIR  = ROOT_DIR / "models" / "saved"
PLOTS_DIR   = ROOT_DIR / "plots"

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── External API keys (optional — project runs on synthetic data without these) ─
EIA_API_KEY = os.getenv("EIA_API_KEY", "")       # https://www.eia.gov/opendata/
OPENMETEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# ── Data settings ─────────────────────────────────────────────────────────────
# Period: 2 years of hourly data
SYNTHETIC_START = "2022-01-01"
SYNTHETIC_END   = "2023-12-31"

# Structural break date — simulates a market rule change mid-dataset
# The drift detector's job is to catch this automatically
STRUCTURAL_BREAK_DATE = "2023-01-15"
BREAK_PRICE_SHIFT     = 18.0    # $/MWh additive shift after rule change
BREAK_VOLATILITY_MULT = 1.6     # Volatility multiplier post-break

# ── Feature engineering ────────────────────────────────────────────────────────
# Lag features (hours back)
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]   # 168 = 1 week

# Rolling window sizes (hours)
ROLLING_WINDOWS = [6, 24, 48, 168]

# Forecast horizon
FORECAST_HORIZON_HOURS = 24   # day-ahead pricing

# ── Model ─────────────────────────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective":       "regression",
    "metric":          "rmse",
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "verbosity":       -1,
    "random_state":    42,
}

# ── Walk-forward validation ────────────────────────────────────────────────────
WFV_INITIAL_TRAIN_DAYS = 120   # Seed the first model with 4 months
WFV_STEP_DAYS          = 7     # Roll forward 1 week at a time
WFV_TEST_DAYS          = 7     # Predict the next 7 days per fold

# ── Drift detection ────────────────────────────────────────────────────────────
# Population Stability Index thresholds (industry standard)
PSI_THRESHOLD_WARNING  = 0.10   # 0.10–0.20 = moderate shift, investigate
PSI_THRESHOLD_CRITICAL = 0.20   # >0.20     = significant shift, retrain
PSI_BINS               = 10
DRIFT_REFERENCE_DAYS   = 90     # Use last 90 days as reference distribution
DRIFT_WINDOW_DAYS      = 14     # Evaluate drift on rolling 14-day windows

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT    = "energy-price-forecasting"

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
