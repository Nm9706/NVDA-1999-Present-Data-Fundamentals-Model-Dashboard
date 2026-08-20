"""
NVDA Project Dashboard (Streamlit)

Run from the project root:
    streamlit run app/dashboard.py

Reads only from data/processed/ -- it never retrains models live, it just
visualizes what pipeline/etl.py, features/build_features.py, and
models/train.py already produced. Run those three scripts first if the
processed files don't exist yet.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

st.set_page_config(page_title="NVDA Project Dashboard", layout="wide")


@st.cache_data
def load_data():
    master = pd.read_parquet(PROC / "master_daily.parquet")
    long_hist = pd.read_parquet(PROC / "features_long_history.parquet")
    preds = pd.read_parquet(PROC / "predictions.parquet")
    with open(PROC / "model_metrics.json") as f:
        metrics = json.load(f)
    ud_raw = pd.read_csv(ROOT / "data" / "raw" / "NVDA_upgrades_downgrades.csv")
    ud_raw["GradeDate"] = pd.to_datetime(ud_raw["GradeDate"], utc=True, errors="coerce").dt.tz_localize(None)
    earnings = pd.read_csv(ROOT / "data" / "raw" / "NVDA_earnings_dates.csv")
    earnings["Earnings_Date"] = pd.to_datetime(earnings["Earnings_Date"], utc=True, errors="coerce").dt.tz_localize(None)
    return master, long_hist, preds, metrics, ud_raw, earnings


master, long_hist, preds, metrics, ud_raw, earnings = load_data()

st.title("NVDA: 1999-Present -- Data, Fundamentals & Model Dashboard")

tab_price, tab_model, tab_analyst, tab_backtest = st.tabs(
    ["Price & Fundamentals", "Model Performance", "Analyst Activity", "Backtest"]
)

# ---------------------------------------------------------------- Price tab
with tab_price:
    st.subheader("Price history with key events")
    log_scale = st.checkbox("Log scale", value=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=master["Date"], y=master["Adj_Close"], name="Adj Close", line=dict(width=1.3)))

    split_rows = master[master["split_today"] == 1]
    fig.add_trace(go.Scatter(
        x=split_rows["Date"], y=split_rows["Adj_Close"], mode="markers", name="Stock split",
        marker=dict(symbol="triangle-up", size=10, color="orange"),
    ))
    div_rows = master[master["ex_dividend_today"] == 1]
    fig.add_trace(go.Scatter(
        x=div_rows["Date"], y=div_rows["Adj_Close"], mode="markers", name="Ex-dividend",
        marker=dict(symbol="circle", size=5, color="green"),
    ))
    fig.add_trace(go.Scatter(
        x=earnings["Earnings_Date"], y=[master["Adj_Close"].max() * 1.02] * len(earnings),
        mode="markers", name="Earnings date",
        marker=dict(symbol="diamond", size=7, color="red"),
    ))
    if log_scale:
        fig.update_yaxes(type="log")
    fig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Note: split/earnings/dividend markers only render where the underlying event data "
        "exists in this dataset (earnings from 2020, analyst data from 2016)."
    )

    st.subheader("Revenue & margin trend (annual)")
    fy_cols = [c for c in master.columns if c.startswith("fy_inc_")]
    fy = master.dropna(subset=["fy_inc_Total_Revenue"]).drop_duplicates(subset="fy_inc_Total_Revenue")
    fy_view = fy[["Date", "fy_inc_Total_Revenue", "fy_inc_Gross_Profit", "fy_inc_Diluted_EPS"]].copy()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=fy_view["Date"], y=fy_view["fy_inc_Total_Revenue"], name="Total Revenue"))
    fig2.add_trace(go.Bar(x=fy_view["Date"], y=fy_view["fy_inc_Gross_Profit"], name="Gross Profit"))
    fig2.update_layout(barmode="group", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- Model tab
with tab_model:
    st.subheader("Walk-forward validation results (next-day direction)")
    rows = []
    for name, res in metrics.items():
        a = res["average"]
        rows.append({"model": name, "accuracy": a["accuracy"], "f1": a["f1"], "roc_auc": a.get("roc_auc")})
    metrics_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "All models land close to the majority-class baseline and ROC-AUC sits near 0.5 -- "
        "consistent with market efficiency for a single-stock, next-day horizon. That's the "
        "expected/honest result, not a bug."
    )

    st.subheader("Rolling accuracy over time (Random Forest)")
    preds_sorted = preds.sort_values("Date").copy()
    preds_sorted["correct_rf"] = (preds_sorted["pred_rf"] == preds_sorted["y_true"]).astype(int)
    preds_sorted["rolling_acc"] = preds_sorted["correct_rf"].rolling(60).mean()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=preds_sorted["Date"], y=preds_sorted["rolling_acc"], name="60-day rolling accuracy"))
    fig3.add_hline(y=0.5, line_dash="dash", annotation_text="Coin flip")
    fig3.update_layout(height=350, yaxis_title="Accuracy")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Confusion matrix (Random Forest, all out-of-sample folds)")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(preds["y_true"], preds["pred_rf"])
    cm_df = pd.DataFrame(cm, index=["Actual Down", "Actual Up"], columns=["Pred Down", "Pred Up"])
    st.dataframe(cm_df, use_container_width=True)

# ------------------------------------------------------------- Analyst tab
with tab_analyst:
    st.subheader("Analyst actions vs price")
    ud_ts = ud_raw.dropna(subset=["GradeDate"]).sort_values("GradeDate")
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=master["Date"], y=master["Adj_Close"], name="Adj Close", yaxis="y1", line=dict(width=1)))
    fig4.add_trace(go.Scatter(
        x=ud_ts["GradeDate"], y=ud_ts["currentPriceTarget"], mode="markers", name="Price target set",
        marker=dict(size=4, color="purple"), yaxis="y1",
    ))
    fig4.update_layout(height=450, hovermode="x unified", yaxis=dict(type="log", title="Price ($)"))
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Upgrade / downgrade counts by action type")
    action_counts = ud_raw["Action"].value_counts()
    fig5 = go.Figure(go.Bar(x=action_counts.index, y=action_counts.values))
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)

# ------------------------------------------------------------ Backtest tab
with tab_backtest:
    st.subheader("Simple strategy vs buy & hold")
    st.caption("Strategy: go long only on days the Random Forest predicts 'up' (prob_rf > 0.5), else hold cash. "
               "Uses only out-of-sample (walk-forward test) predictions -- no lookahead.")

    bt = preds.merge(long_hist[["Date", "fwd_ret_1d"]], on="Date", how="left").sort_values("Date")
    bt["strategy_ret"] = np.where(bt["prob_rf"] > 0.5, bt["fwd_ret_1d"], 0.0)
    bt["strategy_equity"] = (1 + bt["strategy_ret"].fillna(0)).cumprod()
    bt["buy_hold_equity"] = (1 + bt["fwd_ret_1d"].fillna(0)).cumprod()

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=bt["Date"], y=bt["strategy_equity"], name="Model-gated strategy"))
    fig6.add_trace(go.Scatter(x=bt["Date"], y=bt["buy_hold_equity"], name="Buy & hold"))
    fig6.update_layout(height=450, yaxis_title="Growth of $1 (out-of-sample period)")
    st.plotly_chart(fig6, use_container_width=True)

    final_strat = bt["strategy_equity"].iloc[-1]
    final_bh = bt["buy_hold_equity"].iloc[-1]
    col1, col2 = st.columns(2)
    col1.metric("Model-gated strategy", f"{final_strat:.2f}x")
    col2.metric("Buy & hold", f"{final_bh:.2f}x")
    st.caption(
        "This backtest ignores transaction costs, slippage, and taxes, and only covers the "
        "walk-forward test windows (not the full history). Treat it as a teaching illustration "
        "of tying predictions back to an economic outcome, not a trading recommendation."
    )
