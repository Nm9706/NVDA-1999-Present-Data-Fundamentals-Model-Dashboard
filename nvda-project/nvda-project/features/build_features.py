"""
Feature engineering for the NVDA classification task.

Produces TWO feature sets from the master table, because of the coverage
gap surfaced by the ETL step:

  1. long_history  : price/technical + analyst-momentum features only.
                      Usable across the FULL 1999-2026 span (~6,900 rows).
                      This is the set to use for a robust, well-powered model.

  2. fundamentals_enriched : adds quarterly/annual fundamentals on top.
                      Only usable from ~2023 onward (~800-900 rows), since
                      that's when fundamentals data starts. Smaller sample,
                      richer features -- good for showing the classic
                      bias-variance / sample-size tradeoff in your writeup.

Target:
    target_up_1d  : 1 if next day's close > today's close, else 0
    target_up_5d  : 1 if close 5 trading days ahead > today's close, else 0

Run:
    python features/build_features.py
Output:
    data/processed/features_long_history.parquet
    data/processed/features_fundamentals_enriched.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def add_technical_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    px = df["Adj_Close"]

    df["ret_1d"] = px.pct_change(1)
    for lag in (2, 3, 5, 10, 21):
        df[f"ret_{lag}d"] = px.pct_change(lag)

    for w in (5, 10, 21, 63):
        df[f"sma_{w}"] = px.rolling(w).mean()
        df[f"sma_ratio_{w}"] = px / df[f"sma_{w}"]
        df[f"vol_{w}"] = df["ret_1d"].rolling(w).std()

    df["rsi_14"] = rsi(px, 14)
    macd_line, signal_line, hist = macd(px)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    df["vol_change_5d"] = df["Volume"].pct_change(5)
    df["vol_zscore_21"] = (df["Volume"] - df["Volume"].rolling(21).mean()) / df["Volume"].rolling(21).std()

    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]

    return df


def add_targets(df):
    px = df["Adj_Close"]
    df["fwd_ret_1d"] = px.shift(-1) / px - 1
    df["fwd_ret_5d"] = px.shift(-5) / px - 1
    df["target_up_1d"] = (df["fwd_ret_1d"] > 0).astype(int)
    df["target_up_5d"] = (df["fwd_ret_5d"] > 0).astype(int)
    # rows at the very end have no future price -> can't be labeled
    df.loc[df["fwd_ret_1d"].isna(), "target_up_1d"] = np.nan
    df.loc[df["fwd_ret_5d"].isna(), "target_up_5d"] = np.nan
    return df


# Price/technical only -- every one of these is computable from day 1 of
# the price series (modulo rolling-window warmup), so this set stretches
# across the FULL 1999-2026 history. Earnings-date and analyst-action data
# only exist from 2020/2016 onward in this dataset, so those features live
# in the enriched set below instead of truncating this one.
LONG_HISTORY_COLS = [
    "Date", "Adj_Close", "Volume",
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_21d",
    "sma_ratio_5", "sma_ratio_10", "sma_ratio_21", "sma_ratio_63",
    "vol_5", "vol_10", "vol_21", "vol_63",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "vol_change_5d", "vol_zscore_21", "high_low_range",
    "target_up_1d", "target_up_5d", "fwd_ret_1d", "fwd_ret_5d",
]

# Added on top of LONG_HISTORY_COLS for the enriched set: earnings proximity
# and analyst rolling activity. These are only populated from ~2016-2020
# onward, which is part of why the enriched set has far fewer rows.
EVENT_COLS = [
    "days_to_next_earnings", "days_since_last_earnings",
    "analyst_actions_30d", "analyst_score_30d",
    "analyst_actions_90d", "analyst_score_90d",
]

# Annual fundamentals go back to FY2022 (~4.5 years of daily rows once
# forward-filled) -- quarterly fundamentals only go back to Q1 FY2026
# (~1 year), so leaning on annual keeps the enriched sample much larger.
# Swap/add the commented-out quarterly lines if you want fresher (but far
# fewer) fundamentals -- that's a real bias-variance tradeoff worth
# discussing in your writeup, not just a technical footnote.
FUNDAMENTAL_EXTRA_COLS = [
    "fy_inc_Total_Revenue",
    "fy_inc_Gross_Profit",
    "fy_inc_Diluted_EPS",
    "fy_inc_Research_And_Development",
    "fy_inc_Operating_Income",
    "fy_cf_Free_Cash_Flow",
    "fy_cf_Stock_Based_Compensation",
    "fy_bal_Total_Debt",
    "fy_bal_Cash_And_Cash_Equivalents",
    # "q_inc_Total_Revenue", "q_inc_Diluted_EPS",  # uncomment for quarterly (shrinks sample to ~2025+)
]


def main():
    df = pd.read_parquet(PROC / "master_daily.parquet")
    df = add_technical_features(df)
    df = add_targets(df)

    long_hist = df[LONG_HISTORY_COLS].dropna(
        subset=[c for c in LONG_HISTORY_COLS if c not in ("target_up_1d", "target_up_5d", "fwd_ret_1d", "fwd_ret_5d")]
    )
    long_hist = long_hist.dropna(subset=["target_up_1d"])  # drop unlabeled trailing rows
    long_hist.to_parquet(PROC / "features_long_history.parquet", index=False)

    available_extra = [c for c in FUNDAMENTAL_EXTRA_COLS if c in df.columns]
    enriched_cols = LONG_HISTORY_COLS + EVENT_COLS + available_extra
    enriched = df[enriched_cols].copy()
    # YoY revenue growth (based on the annual figures) is more informative
    # for a classifier than the raw revenue level, which just tracks company size
    if "fy_inc_Total_Revenue" in enriched.columns:
        enriched["yoy_revenue_growth"] = enriched["fy_inc_Total_Revenue"].pct_change()
    enriched = enriched.dropna(subset=available_extra + ["target_up_1d"])
    enriched.to_parquet(PROC / "features_fundamentals_enriched.parquet", index=False)

    print(f"long_history:            {long_hist.shape[0]} rows x {long_hist.shape[1]} cols  "
          f"({long_hist['Date'].min().date()} -> {long_hist['Date'].max().date()})")
    print(f"fundamentals_enriched:   {enriched.shape[0]} rows x {enriched.shape[1]} cols  "
          f"({enriched['Date'].min().date()} -> {enriched['Date'].max().date()})")
    print(f"target_up_1d balance (long_history):  {long_hist['target_up_1d'].mean():.3f} pct up")


if __name__ == "__main__":
    main()
