"""
ETL pipeline for the NVDA dataset (1999-present).

Builds a single daily "master table" from 16 raw CSVs:
  price          -> daily OHLCV backbone (1999-2026)
  dividends      -> merged as-of (last dividend known as of each day)
  splits         -> merged as-of
  earnings_dates -> days since/until nearest earnings event
  upgrades/downgrades -> rolling counts of analyst actions, trailing N days
  annual/quarterly fundamentals -> merged as-of REPORT DATE, forward-filled
                                    (so a quarter's numbers only become
                                    "known" the day they were reported --
                                    this is what prevents lookahead bias)

IMPORTANT DATA COVERAGE NOTE (read before modeling):
  - Price data:        1999-01-22 -> 2026-06-15   (~6,900 trading days)
  - Upgrades/downgrades: 2016-08 -> 2026-06        (982 events)
  - Earnings dates:     2020-08 -> 2026-08          (25 events)
  - Annual fundamentals: 2022-01 -> 2026-01         (5 fiscal years)
  - Quarterly fundamentals: 2025-04 -> 2026-04      (5 quarters)

  Fundamentals only cover the last ~5 years. Rows before the first
  reported fundamentals will have NaN in those feature columns by
  design (not a bug) -- see features/build_features.py for how the
  two feature sets ("long-history" vs "fundamentals-enriched") are
  built to work around this.

Run:
    python pipeline/etl.py
Output:
    data/processed/master_daily.parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def _read(name, parse_dates):
    df = pd.read_csv(RAW / name)
    for c in parse_dates:
        # utc=True then drop tz so all timestamps are comparable/naive
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce").dt.tz_localize(None)
    return df


def load_price():
    df = _read("NVDA_price.csv", ["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset="Date")
    return df


def load_dividends():
    df = _read("NVDA_dividends.csv", ["Date"])
    return df.sort_values("Date")


def load_splits():
    df = _read("NVDA_splits.csv", ["Date"])
    return df.sort_values("Date")


def load_earnings_dates():
    df = _read("NVDA_earnings_dates.csv", ["Earnings_Date"])
    df = df.rename(columns={"Earnings_Date": "Date"})
    return df.sort_values("Date")


def load_upgrades_downgrades():
    df = _read("NVDA_upgrades_downgrades.csv", ["GradeDate"])
    df = df.rename(columns={"GradeDate": "Date"})
    # +1 buy-side action, -1 sell-side action, 0 neutral/unclear -- used later
    # for a simple rolling "analyst momentum" feature
    bullish = {"Buy", "Strong Buy", "Outperform", "Overweight", "Positive", "Add"}
    bearish = {"Sell", "Strong Sell", "Underperform", "Underweight", "Negative", "Reduce"}
    df["action_score"] = df["ToGrade"].apply(
        lambda g: 1 if g in bullish else (-1 if g in bearish else 0)
    )
    return df.sort_values("Date")


def load_fundamentals(name):
    """Annual or quarterly income/balance/cashflow -- wide format, one row
    per report date. Reported columns are prefixed to avoid collisions
    when we join income+balance+cashflow together."""
    df = _read(name, ["Date"])
    return df.sort_values("Date")


def build_fundamentals_panel(prefix, income_file, balance_file, cashflow_file):
    inc = load_fundamentals(income_file).add_prefix("inc_").rename(columns={"inc_Date": "Date"})
    bal = load_fundamentals(balance_file).add_prefix("bal_").rename(columns={"bal_Date": "Date"})
    cf = load_fundamentals(cashflow_file).add_prefix("cf_").rename(columns={"cf_Date": "Date"})
    panel = inc.merge(bal, on="Date", how="outer").merge(cf, on="Date", how="outer")
    panel = panel.sort_values("Date").add_prefix(prefix).rename(columns={f"{prefix}Date": "Date"})
    return panel.sort_values("Date")


def asof_merge_forward_filled(base, other, cols, label):
    """Merge `other`'s columns onto `base` using the LAST known value as of
    each price date. This is the key leakage-prevention step: a fundamental
    value is only visible starting the day it was actually reported."""
    other = other[["Date"] + cols].dropna(subset=["Date"]).sort_values("Date")
    merged = pd.merge_asof(base.sort_values("Date"), other, on="Date", direction="backward")
    return merged


def add_days_to_from_earnings(base, earnings):
    dates = earnings["Date"].sort_values().reset_index(drop=True)
    base = base.sort_values("Date").reset_index(drop=True)

    idx_next = np.searchsorted(dates.values, base["Date"].values, side="left")
    idx_prev = idx_next - 1

    next_dates = dates.reindex(idx_next).values
    prev_dates = dates.reindex(idx_prev.clip(min=0)).values
    prev_dates = np.where(idx_prev >= 0, prev_dates, np.datetime64("NaT"))

    base["days_to_next_earnings"] = (pd.to_datetime(next_dates) - base["Date"]).dt.days
    base["days_since_last_earnings"] = (base["Date"] - pd.to_datetime(prev_dates)).dt.days
    return base


def add_rolling_analyst_activity(base, ud, windows=(30, 90)):
    ud = ud[["Date", "action_score"]].dropna().sort_values("Date")
    base = base.sort_values("Date").reset_index(drop=True)
    for w in windows:
        col_count = f"analyst_actions_{w}d"
        col_score = f"analyst_score_{w}d"
        counts = []
        scores = []
        j0 = 0
        dates = ud["Date"].values
        scores_arr = ud["action_score"].values
        for i, d in enumerate(base["Date"].values):
            lo = d - np.timedelta64(w, "D")
            # advance window start
            j0 = np.searchsorted(dates, lo, side="left")
            j1 = np.searchsorted(dates, d, side="right")
            window_scores = scores_arr[j0:j1]
            counts.append(len(window_scores))
            scores.append(window_scores.sum() if len(window_scores) else 0)
        base[col_count] = counts
        base[col_score] = scores
    return base


def main():
    price = load_price()
    dividends = load_dividends()
    splits = load_splits()
    earnings = load_earnings_dates()
    ud = load_upgrades_downgrades()

    annual_fund = build_fundamentals_panel(
        "fy_", "NVDA_income.csv", "NVDA_balance.csv", "NVDA_cashflow.csv"
    )
    quarterly_fund = build_fundamentals_panel(
        "q_", "NVDA_q_income.csv", "NVDA_q_balance.csv", "NVDA_q_cashflow.csv"
    )

    df = price.copy()

    # dividends / splits: as-of, plus a same-day flag (event on that date)
    df["ex_dividend_today"] = df["Date"].isin(dividends["Date"]).astype(int)
    df["split_today"] = df["Date"].isin(splits["Date"]).astype(int)
    df = asof_merge_forward_filled(df, dividends, ["Dividends"], "div")
    df = asof_merge_forward_filled(df, splits, ["Splits"], "split")
    df["Dividends"] = df["Dividends"].fillna(0.0)

    # earnings proximity
    df = add_days_to_from_earnings(df, earnings)

    # analyst rolling activity (this only becomes meaningful from 2016 on --
    # see coverage note at top of file)
    df = add_rolling_analyst_activity(df, ud, windows=(30, 90))

    # fundamentals -- as-of REPORT DATE so no lookahead. NaN before first report.
    fund_cols_annual = [c for c in annual_fund.columns if c != "Date"]
    fund_cols_quarterly = [c for c in quarterly_fund.columns if c != "Date"]
    df = asof_merge_forward_filled(df, annual_fund, fund_cols_annual, "annual_fund")
    df = asof_merge_forward_filled(df, quarterly_fund, fund_cols_quarterly, "quarterly_fund")

    df = df.sort_values("Date").reset_index(drop=True)

    out_path = OUT / "master_daily.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Master table: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Date range:   {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Saved to:     {out_path}")

    # quick coverage report so it's obvious in the console, not just the docstring
    first_fund = df.loc[df["fy_inc_Total_Revenue"].notna(), "Date"].min()
    first_analyst = df.loc[df["analyst_actions_30d"] > 0, "Date"].min()
    print(f"First date with annual fundamentals available: {first_fund.date()}")
    print(f"First date with analyst activity available:    {first_analyst.date()}")


if __name__ == "__main__":
    main()
