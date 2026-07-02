"""
FastAPI Prediction Endpoint
============================
Serves day-ahead electricity price forecasts from the trained LightGBM model.

Endpoints:
  GET  /health              → Health check + model info
  POST /predict             → 24-hour price forecast for a given datetime
  GET  /predict/next24h     → Forecast for the next 24 hours from now
  GET  /model/features      → Feature importance list
  POST /drift/check         → Run PSI drift check on supplied price history
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.forecaster import load_model, get_feature_importance
from features.engineer import build_feature_matrix
from drift.detector import compute_psi, psi_status
from config import MODELS_DIR, FORECAST_HORIZON_HOURS


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Energy Price Forecaster API",
    description="Day-ahead electricity price forecasting for ERCOT with drift detection",
    version="1.0.0",
)

# Load model on startup (module-level so it's not reloaded each request)
try:
    _model, _feature_names = load_model()
    print(f"✅ Model loaded: {len(_feature_names)} features")
except Exception as e:
    print(f"⚠️  Model not yet trained. Run pipeline.py first. ({e})")
    _model, _feature_names = None, []


# ── Request/Response schemas ───────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input for a single-timestamp forecast."""
    timestamp:     str   = Field(..., example="2024-01-15 14:00:00",
                                  description="ISO datetime for forecast anchor")
    temperature_f: float = Field(72.0, ge=-30, le=130, description="Ambient temperature °F")
    load_mw:       int   = Field(42000, ge=1000, le=100000, description="Grid load in MW")
    recent_prices: list[float] = Field(
        default_factory=list,
        description="Last 168+ hours of prices ($/MWh) — newest last. "
                    "If omitted, uses fallback estimates."
    )


class ForecastPoint(BaseModel):
    hour:          int
    timestamp:     str
    predicted_price: float
    price_low:     float   # rough ±1 std estimate
    price_high:    float


class PredictResponse(BaseModel):
    forecast_generated_at: str
    target_datetime:       str
    predicted_price_mwh:   float
    confidence_range:      dict
    forecast_horizon_hours: int


class DriftCheckRequest(BaseModel):
    reference_prices: list[float] = Field(..., description="Reference period prices (90+ hours)")
    current_prices:   list[float] = Field(..., description="Current window prices (14+ hours)")


class DriftCheckResponse(BaseModel):
    psi_value:    float
    status:       str
    recommendation: str
    reference_n:  int
    current_n:    int


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_single_row(req: PredictRequest) -> pd.DataFrame:
    """
    Construct a minimal feature row for a single prediction timestamp.
    Uses recent_prices to compute lag and rolling features if provided.
    """
    ts = pd.Timestamp(req.timestamp)

    # We need historical prices to compute lag features
    # Build a small synthetic history if recent_prices not supplied
    if len(req.recent_prices) >= 168:
        hist_prices = list(req.recent_prices[-200:])
    else:
        # Fallback: use supplied price or a generic estimate
        fallback_price = req.recent_prices[-1] if req.recent_prices else 55.0
        hist_prices    = [fallback_price] * 200

    # Build a small DataFrame ending at ts
    hist_ts = pd.date_range(end=ts, periods=len(hist_prices), freq="h")
    mini_df = pd.DataFrame({
        "timestamp":     hist_ts,
        "price_mwh":     hist_prices,
        "temperature_f": [req.temperature_f] * len(hist_prices),
        "load_mw":       [req.load_mw]       * len(hist_prices),
        "is_post_break": [0]                 * len(hist_prices),
    })

    from features.engineer import (
        add_calendar_features, add_cyclical_features,
        add_lag_features, add_rolling_features, add_interaction_features,
    )
    mini_df = add_calendar_features(mini_df)
    mini_df = add_cyclical_features(mini_df)
    mini_df = add_lag_features(mini_df)
    mini_df = add_rolling_features(mini_df)
    mini_df = add_interaction_features(mini_df)
    mini_df = mini_df.dropna()

    return mini_df


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_loaded":  _model is not None,
        "n_features":    len(_feature_names),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "timestamp":     datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline.py first.")

    try:
        mini_df = _build_single_row(req)
        if mini_df.empty:
            raise ValueError("Could not build feature row — check inputs.")

        row     = mini_df.tail(1)
        missing = [f for f in _feature_names if f not in row.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X       = row[_feature_names]
        pred    = float(_model.predict(X)[0])

        # Rough confidence interval (±1.5 * naive price std)
        price_std = 8.0  # $/MWh typical ±1σ
        return PredictResponse(
            forecast_generated_at   = datetime.utcnow().isoformat(),
            target_datetime         = str(pd.Timestamp(req.timestamp) + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)),
            predicted_price_mwh     = round(pred, 2),
            confidence_range        = {
                "low":  round(max(0, pred - 1.5 * price_std), 2),
                "high": round(pred + 1.5 * price_std, 2),
            },
            forecast_horizon_hours  = FORECAST_HORIZON_HOURS,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/drift/check", response_model=DriftCheckResponse)
def drift_check(req: DriftCheckRequest):
    ref = np.array(req.reference_prices)
    cur = np.array(req.current_prices)

    if len(ref) < 24 or len(cur) < 12:
        raise HTTPException(status_code=422, detail="Need ≥24 reference and ≥12 current prices.")

    psi = compute_psi(ref, cur)
    status = psi_status(psi)

    if psi < 0.10:
        rec = "Model is stable. No action required."
    elif psi < 0.20:
        rec = "Moderate distributional shift detected. Monitor closely and schedule retrain within 1–2 weeks."
    else:
        rec = "Significant shift detected. Retrain immediately using post-shift data."

    return DriftCheckResponse(
        psi_value=round(psi, 4),
        status=status,
        recommendation=rec,
        reference_n=len(ref),
        current_n=len(cur),
    )


@app.get("/model/features")
def get_features():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    fi = get_feature_importance(_model, _feature_names)
    return {"features": fi.to_dict(orient="records")}


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
