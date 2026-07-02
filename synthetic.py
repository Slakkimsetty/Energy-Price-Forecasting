"""
Synthetic Electricity Price Generator
======================================
Produces 2 years of realistic hourly electricity prices for ERCOT (Texas grid),
including:
  - Seasonal price cycles (winter/summer peaks)
  - Intraday demand curves (morning ramp, evening peak)
  - Weekend demand reduction
  - Random price spikes (supply squeezes, weather events)
  - Temperature-correlated demand
  - A deliberate STRUCTURAL BREAK at STRUCTURAL_BREAK_DATE:
      * Price level shifts up by BREAK_PRICE_SHIFT $/MWh
      * Volatility increases by BREAK_VOLATILITY_MULT
    This simulates a market rule change (e.g., new capacity requirement).
    The drift detector's job is to catch it.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SYNTHETIC_START, SYNTHETIC_END,
    STRUCTURAL_BREAK_DATE,
    BREAK_PRICE_SHIFT, BREAK_VOLATILITY_MULT,
    DATA_DIR,
)


def _seasonal_temp(hour_of_year: np.ndarray) -> np.ndarray:
    """Synthetic temperature (°F) with Texas-style seasonal pattern."""
    # Peak summer ~100°F in July, mild winter ~45°F
    base = 72.0
    amplitude = 27.0
    phase_shift = np.pi  # peaks in July (~hour 4380)
    return base + amplitude * np.sin(2 * np.pi * hour_of_year / 8760 + phase_shift)


def _intraday_price_curve(hour: np.ndarray) -> np.ndarray:
    """
    Intraday price shape ($/MWh additive):
    - Morning ramp: 6–9 AM
    - Midday dip: 11 AM–2 PM (solar generation)
    - Evening peak: 5–8 PM (highest prices)
    - Overnight trough: midnight–5 AM
    """
    curve = np.zeros_like(hour, dtype=float)
    curve += np.where((hour >= 6)  & (hour < 9),  12.0, 0)   # morning ramp
    curve += np.where((hour >= 11) & (hour < 14), -5.0, 0)   # solar dip
    curve += np.where((hour >= 17) & (hour < 20), 20.0, 0)   # evening peak
    curve += np.where((hour >= 0)  & (hour < 5),  -8.0, 0)   # overnight trough
    return curve


def _weekly_shape(day_of_week: np.ndarray) -> np.ndarray:
    """Weekends have ~10% lower demand → lower prices."""
    return np.where(day_of_week >= 5, -6.0, 0.0)


def _temperature_effect(temp: np.ndarray) -> np.ndarray:
    """
    U-shaped relationship: both very hot and very cold days drive high prices.
    Minimum around 65°F (comfort zone).
    """
    comfort = 65.0
    return 0.04 * (temp - comfort) ** 2


def _price_spikes(n: int, rng: np.random.Generator, base_rate: float = 0.003) -> np.ndarray:
    """Rare but large price spikes (supply squeeze, grid emergency)."""
    spikes = np.zeros(n)
    spike_idx = rng.choice(n, size=int(n * base_rate), replace=False)
    spike_magnitudes = rng.exponential(scale=80, size=len(spike_idx))
    spikes[spike_idx] = spike_magnitudes
    return spikes


def generate(seed: int = 42, save: bool = True) -> pd.DataFrame:
    """
    Generate synthetic hourly electricity price DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, price_mwh, temperature_f, load_mw, is_post_break
    """
    rng = np.random.default_rng(seed)

    # Build hourly index
    idx = pd.date_range(start=SYNTHETIC_START, end=SYNTHETIC_END, freq="h")
    n   = len(idx)

    hour_of_year = (idx.dayofyear - 1) * 24 + idx.hour
    temp         = _seasonal_temp(hour_of_year.values)
    # Add daily noise to temperature
    temp        += rng.normal(0, 4, n)

    # Base price
    base_price   = 35.0  # $/MWh floor

    # Seasonal component (summer/winter highs)
    seasonal     = 10.0 * np.sin(2 * np.pi * hour_of_year.values / 8760 + np.pi) + \
                    8.0 * np.cos(4 * np.pi * hour_of_year.values / 8760)

    # Intraday + weekly shape
    intraday     = _intraday_price_curve(idx.hour.values)
    weekly       = _weekly_shape(idx.dayofweek.values)

    # Temperature-driven demand effect
    temp_effect  = _temperature_effect(temp)

    # Random noise
    noise        = rng.normal(0, 3.5, n)

    # Occasional spikes
    spikes       = _price_spikes(n, rng, base_rate=0.003)

    # Assemble pre-break price
    price        = base_price + seasonal + intraday + weekly + temp_effect + noise + spikes

    # ── Apply structural break ────────────────────────────────────────────────
    break_ts     = pd.Timestamp(STRUCTURAL_BREAK_DATE)
    is_post      = idx >= break_ts

    # Shift mean upward
    price[is_post] += BREAK_PRICE_SHIFT

    # Increase volatility — resample the noise component with higher std
    post_noise    = rng.normal(0, 3.5 * BREAK_VOLATILITY_MULT, n)
    price[is_post] = (price[is_post]
                      - noise[is_post]          # remove original noise
                      + post_noise[is_post])    # replace with higher-vol noise

    # Prices can't go negative (well, rarely — clamp at 0 for simplicity)
    price         = np.maximum(price, 0.0)

    # Synthetic load (MW) — correlated with temp + intraday pattern
    load_base     = 40_000
    load          = (load_base
                     + 100 * temp_effect
                     + 500 * (intraday / 20.0)
                     + rng.normal(0, 800, n))
    load[is_post] *= 1.03   # slight load increase post-break

    df = pd.DataFrame({
        "timestamp":     idx,
        "price_mwh":     price.round(2),
        "temperature_f": temp.round(1),
        "load_mw":       load.round(0).astype(int),
        "is_post_break": is_post.astype(int),
    })

    if save:
        path = DATA_DIR / "synthetic_prices.parquet"
        df.to_parquet(path, index=False)
        print(f"✅ Synthetic data saved → {path}")
        print(f"   Rows: {len(df):,}  |  Pre-break: {(~is_post).sum():,}  |  Post-break: {is_post.sum():,}")
        print(f"   Price range: ${price.min():.1f} – ${price.max():.1f} /MWh")
        print(f"   Pre-break avg:  ${price[~is_post].mean():.2f}/MWh")
        print(f"   Post-break avg: ${price[is_post].mean():.2f}/MWh  (↑ +{BREAK_PRICE_SHIFT:.0f} expected)")

    return df


if __name__ == "__main__":
    df = generate()
    print(df.head(10))
