"""
Visualizations
==============
All plots for the project. Each function saves a PNG to PLOTS_DIR.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR, STRUCTURAL_BREAK_DATE, PSI_THRESHOLD_WARNING, PSI_THRESHOLD_CRITICAL

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary":  "#2563EB",   # blue
    "accent":   "#F59E0B",   # amber
    "danger":   "#DC2626",   # red
    "success":  "#16A34A",   # green
    "neutral":  "#6B7280",   # gray
    "pre":      "#60A5FA",   # light blue (pre-break)
    "post":     "#F97316",   # orange (post-break)
}

break_ts = pd.Timestamp(STRUCTURAL_BREAK_DATE)


def _save(name: str, fig: plt.Figure) -> Path:
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Saved: {path.name}")
    return path


# ── 1. Price Time Series with Structural Break ────────────────────────────────

def plot_price_series(df: pd.DataFrame) -> Path:
    """Full 2-year price series with break line and rolling mean."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Hourly prices
    ax = axes[0]
    pre  = df[df["timestamp"] < break_ts]
    post = df[df["timestamp"] >= break_ts]

    ax.plot(pre["timestamp"],  pre["price_mwh"],  color=COLORS["pre"],  alpha=0.6, lw=0.4, label="Pre-break")
    ax.plot(post["timestamp"], post["price_mwh"], color=COLORS["post"], alpha=0.6, lw=0.4, label="Post-break")

    # 7-day rolling mean
    roll = df.set_index("timestamp")["price_mwh"].rolling("7D").mean()
    ax.plot(roll.index, roll.values, color=COLORS["danger"], lw=2.0, label="7-day rolling mean")

    ax.axvline(break_ts, color="black", ls="--", lw=1.5, label=f"Market Rule Change\n({STRUCTURAL_BREAK_DATE})")
    ax.set_ylabel("Price ($/MWh)", fontsize=11)
    ax.set_title("ERCOT Synthetic Day-Ahead Electricity Prices", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, None)

    # Daily price distribution (violin)
    ax2 = axes[1]
    df["month_label"] = df["timestamp"].dt.strftime("%b %Y")
    months_order = df.sort_values("timestamp")["month_label"].unique()

    monthly_data = [df[df["month_label"] == m]["price_mwh"].values for m in months_order]
    colors_list  = [COLORS["pre"] if pd.Timestamp(m) < break_ts else COLORS["post"]
                    for m in pd.to_datetime(months_order, format="%b %Y")]

    vp = ax2.violinplot(monthly_data, positions=range(len(months_order)), widths=0.7, showmedians=True)
    for i, (body, col) in enumerate(zip(vp["bodies"], colors_list)):
        body.set_facecolor(col)
        body.set_alpha(0.6)

    ax2.set_xticks(range(len(months_order)))
    ax2.set_xticklabels(months_order, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Price ($/MWh)", fontsize=11)
    ax2.set_title("Monthly Price Distribution (Blue=Pre-break, Orange=Post-break)", fontsize=11)

    fig.tight_layout()
    return _save("01_price_series", fig)


# ── 2. Walk-Forward Validation Results ────────────────────────────────────────

def plot_wfv_results(predictions_df: pd.DataFrame, fold_metrics: list[dict]) -> Path:
    """Actual vs predicted prices + per-fold RMSE over time."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))

    # Panel 1: Actual vs Predicted (last 60 days for clarity)
    ax = axes[0]
    tail = predictions_df[predictions_df["timestamp"] >= predictions_df["timestamp"].max() - pd.Timedelta(days=60)]
    ax.plot(tail["timestamp"], tail["y_true"], color=COLORS["neutral"], lw=1.0, label="Actual", alpha=0.8)
    ax.plot(tail["timestamp"], tail["y_pred"], color=COLORS["primary"], lw=1.0, label="Predicted", alpha=0.85)
    ax.set_title("Actual vs Predicted Electricity Prices (Last 60 Days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price ($/MWh)", fontsize=10)
    ax.legend(fontsize=9)

    # Panel 2: RMSE per fold over time
    ax2 = axes[1]
    fm_df = pd.DataFrame(fold_metrics)
    fm_df["test_start"] = pd.to_datetime(fm_df["test_start"])
    ax2.plot(fm_df["test_start"], fm_df["rmse"], color=COLORS["primary"], lw=1.5, marker="o", ms=3)
    ax2.axvline(break_ts, color=COLORS["danger"], ls="--", lw=1.5, label=f"Break ({STRUCTURAL_BREAK_DATE})")
    ax2.axhline(fm_df["rmse"].mean(), color=COLORS["neutral"], ls=":", lw=1.2, label=f"Mean RMSE = {fm_df['rmse'].mean():.2f}")
    ax2.set_title("Walk-Forward Validation: RMSE Per Fold", fontsize=12, fontweight="bold")
    ax2.set_ylabel("RMSE ($/MWh)", fontsize=10)
    ax2.legend(fontsize=9)

    # Panel 3: Residuals
    ax3 = axes[2]
    residuals = predictions_df["y_true"] - predictions_df["y_pred"]
    ax3.scatter(predictions_df["timestamp"], residuals, alpha=0.15, s=2,
                c=predictions_df["is_post_break"].map({0: COLORS["pre"], 1: COLORS["post"]}))
    ax3.axhline(0, color="black", lw=1.0)
    ax3.axvline(break_ts, color=COLORS["danger"], ls="--", lw=1.5)
    ax3.set_title("Residuals Over Time (Blue=Pre-break, Orange=Post-break)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Residual ($/MWh)", fontsize=10)

    fig.tight_layout()
    return _save("02_wfv_results", fig)


# ── 3. Drift Detection Dashboard ──────────────────────────────────────────────

def plot_drift_dashboard(summary_df: pd.DataFrame, drift_df: pd.DataFrame) -> Path:
    """Rolling PSI over time with threshold lines and break marker."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # Panel 1: Max PSI over time
    ax = axes[0]
    summary_df = summary_df.copy()
    summary_df["window_date"] = pd.to_datetime(summary_df["window_date"])

    pre  = summary_df[summary_df["window_date"] < break_ts]
    post = summary_df[summary_df["window_date"] >= break_ts]

    ax.plot(pre["window_date"],  pre["max_psi"],  color=COLORS["pre"],  lw=2.0, label="Max PSI (pre-break)")
    ax.plot(post["window_date"], post["max_psi"], color=COLORS["post"], lw=2.0, label="Max PSI (post-break)")
    ax.plot(summary_df["window_date"], summary_df["mean_psi"],
            color=COLORS["neutral"], lw=1.2, ls="--", alpha=0.7, label="Mean PSI")

    ax.axhline(PSI_THRESHOLD_WARNING,  color=COLORS["accent"], ls="--", lw=1.2, label=f"Warning ({PSI_THRESHOLD_WARNING})")
    ax.axhline(PSI_THRESHOLD_CRITICAL, color=COLORS["danger"],  ls="--", lw=1.5, label=f"Critical ({PSI_THRESHOLD_CRITICAL})")
    ax.axvline(break_ts, color="black", ls="-", lw=2.0, label=f"Market Rule Change ({STRUCTURAL_BREAK_DATE})")

    ax.fill_between(summary_df["window_date"], PSI_THRESHOLD_CRITICAL, summary_df["max_psi"].clip(lower=PSI_THRESHOLD_CRITICAL),
                    alpha=0.15, color=COLORS["danger"])
    ax.fill_between(summary_df["window_date"], PSI_THRESHOLD_WARNING, summary_df["max_psi"].clip(lower=PSI_THRESHOLD_WARNING, upper=PSI_THRESHOLD_CRITICAL),
                    alpha=0.10, color=COLORS["accent"])

    ax.set_title("Drift Detection: Rolling PSI Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Population Stability Index (PSI)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # Panel 2: Per-feature PSI heatmap (top features)
    ax2 = axes[1]
    top_feats = (
        drift_df.groupby("feature")["psi"].max()
        .sort_values(ascending=False)
        .head(8).index.tolist()
    )
    pivot = drift_df[drift_df["feature"].isin(top_feats)].pivot_table(
        index="feature", columns="window_date", values="psi", aggfunc="mean"
    )
    pivot.columns = pd.to_datetime(pivot.columns)
    # Thin out columns for readability
    cols_to_show = pivot.columns[::4]
    pivot = pivot[cols_to_show]

    sns.heatmap(
        pivot, ax=ax2, cmap="YlOrRd", linewidths=0.3,
        xticklabels=[c.strftime("%b %Y") for c in pivot.columns],
        cbar_kws={"label": "PSI"},
        vmin=0, vmax=1.0,
    )
    ax2.set_title("PSI Heatmap by Feature and Window (Top 8 Drifted Features)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return _save("03_drift_dashboard", fig)


# ── 4. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20) -> Path:
    """Horizontal bar chart of top N feature importances."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fi = feature_importance.head(top_n).iloc[::-1]  # reverse for horizontal bar

    colors = [COLORS["primary"] if "price" in f else
              COLORS["accent"]  if "temp"  in f or "load" in f else
              COLORS["success"] for f in fi["feature"]]

    bars = ax.barh(fi["feature"], fi["importance"], color=colors, alpha=0.85)

    # Value labels
    for bar, val in zip(bars, fi["importance"]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)

    ax.set_xlabel("Feature Importance (LightGBM gain)", fontsize=10)
    ax.set_title(f"Top {top_n} Feature Importances\n(Blue=Price features, Amber=Weather/Load, Green=Calendar)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save("04_feature_importance", fig)


# ── 5. Intraday Price Patterns ────────────────────────────────────────────────

def plot_intraday_patterns(df: pd.DataFrame) -> Path:
    """Average price by hour of day: pre vs post break, and by day of week."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"]  = df["timestamp"].dt.dayofweek
    df["period"] = df["is_post_break"].map({0: "Pre-break", 1: "Post-break"})

    # Panel 1: Hourly pattern
    ax = axes[0]
    for period, color in [("Pre-break", COLORS["pre"]), ("Post-break", COLORS["post"])]:
        sub = df[df["period"] == period].groupby("hour")["price_mwh"].agg(["mean", "std"])
        ax.plot(sub.index, sub["mean"], color=color, lw=2.5, label=period, marker="o", ms=4)
        ax.fill_between(sub.index, sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.15, color=color)

    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Avg Price ($/MWh)", fontsize=10)
    ax.set_title("Intraday Price Pattern: Pre vs Post Market Rule Change", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, 24, 2))

    # Panel 2: Day of week pattern
    ax2 = axes[1]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for period, color in [("Pre-break", COLORS["pre"]), ("Post-break", COLORS["post"])]:
        sub = df[df["period"] == period].groupby("dow")["price_mwh"].mean()
        ax2.bar(np.arange(7) + (0 if period == "Pre-break" else 0.35),
                sub.values, width=0.35, color=color, alpha=0.8, label=period)

    ax2.set_xticks(np.arange(7) + 0.175)
    ax2.set_xticklabels(dow_names)
    ax2.set_ylabel("Avg Price ($/MWh)", fontsize=10)
    ax2.set_title("Average Price by Day of Week", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)

    fig.tight_layout()
    return _save("05_intraday_patterns", fig)
