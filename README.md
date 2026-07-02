# ⚡ Energy Price Forecaster (SmartBidder-style)

A production-grade day-ahead electricity price forecasting system with built-in distributional drift detection. Built to demonstrate ML engineering patterns relevant to energy market bidding systems.

---

## What This Does

| Component | Description |
|---|---|
| **Synthetic ERCOT data** | 2 years of realistic hourly prices with seasonal/intraday patterns, price spikes, and a deliberate **market rule change** (structural break) on `2023-01-15` |
| **43 engineered features** | Lag prices, rolling stats, calendar features, cyclical sin/cos encodings, weather interactions |
| **LightGBM + Walk-Forward Validation** | 86 folds, expanding window — no future leakage |
| **PSI Drift Detector** | Catches the structural break in **4 days** via Population Stability Index |
| **MLflow Tracking** | Logs all fold metrics, drift scores, feature importance, and the model artifact |
| **FastAPI endpoint** | `/predict` returns day-ahead price forecasts; `/drift/check` runs on-demand PSI |

---

## Results

```
Walk-Forward Validation (86 folds):
  Avg RMSE:  10.19 $/MWh
  Avg MAE:    7.21 $/MWh
  Avg MAPE:  11.15%

Drift Detection:
  Structural break injected:  2023-01-15
  First PSI alert fired:      2023-01-19  (4 days later)
  Post-break max PSI:         27.27  [CRITICAL — retrain required]
```

---

## Project Structure

```
energy-price-forecaster/
├── config.py                  # All tunable parameters (one place)
├── pipeline.py                # End-to-end runner (start here)
│
├── data/
│   └── synthetic.py           # ERCOT-style data generator with structural break
│
├── features/
│   └── engineer.py            # Lag/rolling/calendar/cyclical features (43 total)
│
├── models/
│   ├── forecaster.py          # LightGBM + walk-forward validation
│   └── saved/                 # Trained model artifacts (.joblib)
│
├── drift/
│   └── detector.py            # PSI-based rolling drift monitor
│
├── tracking/
│   └── mlflow_logger.py       # MLflow experiment logging
│
├── api/
│   └── main.py                # FastAPI prediction + drift check endpoints
│
├── visualizations.py          # 5 production-ready plots
└── plots/                     # Output: price series, WFV results, drift dashboard...
```

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/energy-price-forecaster
cd energy-price-forecaster
pip install -r requirements.txt

# Run the full pipeline (generates data, trains, evaluates, detects drift, plots)
python pipeline.py

# View MLflow experiment dashboard
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start the prediction API
python -m uvicorn api.main:app --reload
```

No API keys needed — the project runs on synthetic data out of the box.

---

## Design Decisions Worth Noting

### Why walk-forward, not random split?
Random splits leak future data into training — a model that "sees" next week's price while predicting this week will look ~8 points better in RMSE than it actually is. Walk-forward validation mirrors real production: always train on the past, predict the future.

### Why PSI for drift detection?
PSI is the industry standard in financial ML because it's distribution-agnostic and fires on **input** feature drift before performance metrics degrade. In this project, the PSI alert fires 4 days after the market rule change — well before you'd see it in rolling RMSE.

### Why lagged features instead of a time series model?
LightGBM with lag/rolling features routinely matches or beats ARIMA/Prophet on electricity price forecasting benchmarks, and it handles the complex non-linear interactions (temperature × peak hour, weekend effects) that pure time series models struggle with. It's also far easier to deploy and monitor in production.

---

## Upgrading to Real Data

Replace `data/synthetic.py` with real data by getting a free EIA API key at https://www.eia.gov/opendata/ and hitting:

```
GET https://api.eia.gov/v2/electricity/rto/region-data/data/
    ?api_key=YOUR_KEY&frequency=hourly&data[]=value&facets[respondent][]=ERCO
```

The rest of the pipeline (features, model, drift, API) works unchanged.

---

## API Reference

```bash
# Health check
curl http://localhost:8000/health

# Predict day-ahead price
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-01-15 14:00:00", "temperature_f": 85.0, "load_mw": 55000}'

# Check for drift (supply reference and current price windows)
curl -X POST http://localhost:8000/drift/check \
  -H "Content-Type: application/json" \
  -d '{"reference_prices": [...], "current_prices": [...]}'

# Feature importance
curl http://localhost:8000/model/features
```

---

## Stack

Python · LightGBM · pandas · scikit-learn · MLflow · FastAPI · matplotlib/seaborn

---

## Author

Srikar Lakkimsetti · [LinkedIn](https://linkedin.com) · [srikar.l@ajobguide.com](mailto:srikar.l@ajobguide.com)
