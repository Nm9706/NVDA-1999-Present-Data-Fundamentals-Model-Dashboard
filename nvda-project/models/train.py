"""
Train and evaluate classifiers to predict NVDA next-day direction.

Uses WALK-FORWARD (expanding window) validation, not k-fold -- shuffling
time series data would leak future information into training and produce
misleadingly high accuracy. Each fold trains only on data strictly before
the test window.

Models compared:
    - Naive baseline (predict "yesterday's direction repeats")
    - Majority-class baseline (always predict the more common class)
    - Logistic Regression
    - Random Forest
    - XGBoost

Run:
    python models/train.py
Output:
    data/processed/predictions.parquet   (out-of-sample predictions, all folds)
    data/processed/model_metrics.json    (per-model, per-fold metrics)
    models/artifacts/xgb_model.json      (final model, trained on all but last fold)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

FEATURE_COLS = None  # set in main() once we know the dataframe


def make_walk_forward_folds(n_rows, n_folds=5, min_train_frac=0.5):
    """Expanding-window splits: each fold's test block comes strictly after
    its train block. min_train_frac reserves the first chunk of history as
    a warmup so the earliest fold still has a reasonable amount to train on."""
    min_train = int(n_rows * min_train_frac)
    remaining = n_rows - min_train
    fold_size = remaining // n_folds
    folds = []
    for i in range(n_folds):
        train_end = min_train + i * fold_size
        test_end = train_end + fold_size if i < n_folds - 1 else n_rows
        folds.append((slice(0, train_end), slice(train_end, test_end)))
    return folds


def naive_baseline(y_train, y_test, prev_direction):
    """Predict that tomorrow repeats today's realized direction."""
    return prev_direction.astype(int).values


def majority_baseline(y_train, y_test):
    majority = int(y_train.mean() > 0.5)
    return np.full(len(y_test), majority)


def evaluate(y_true, y_pred, y_prob=None):
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    return out


def main():
    df = pd.read_parquet(PROC / "features_long_history.parquet")
    df = df.sort_values("Date").reset_index(drop=True)

    global FEATURE_COLS
    exclude = {"Date", "Adj_Close", "target_up_1d", "target_up_5d", "fwd_ret_1d", "fwd_ret_5d"}
    FEATURE_COLS = [c for c in df.columns if c not in exclude]

    X = df[FEATURE_COLS].values
    y = df["target_up_1d"].values.astype(int)
    prev_dir = (df["ret_1d"] > 0).astype(int)

    folds = make_walk_forward_folds(len(df), n_folds=5, min_train_frac=0.6)

    results = {"naive": [], "majority": [], "logreg": [], "random_forest": [], "xgboost": []}
    all_preds = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        dates_test = df["Date"].values[test_idx]
        prev_dir_test = prev_dir.values[test_idx]

        # --- baselines ---
        pred_naive = naive_baseline(y_train, y_test, pd.Series(prev_dir_test))
        results["naive"].append(evaluate(y_test, pred_naive))

        pred_majority = majority_baseline(y_train, y_test)
        results["majority"].append(evaluate(y_test, pred_majority))

        # --- scale for logreg ---
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

        logreg = LogisticRegression(max_iter=1000, C=1.0)
        logreg.fit(X_train_s, y_train)
        prob_lr = logreg.predict_proba(X_test_s)[:, 1]
        pred_lr = (prob_lr > 0.5).astype(int)
        results["logreg"].append(evaluate(y_test, pred_lr, prob_lr))

        rf = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                     random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        prob_rf = rf.predict_proba(X_test)[:, 1]
        pred_rf = (prob_rf > 0.5).astype(int)
        results["random_forest"].append(evaluate(y_test, pred_rf, prob_rf))

        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        pred_xgb = (prob_xgb > 0.5).astype(int)
        results["xgboost"].append(evaluate(y_test, pred_xgb, prob_xgb))

        all_preds.append(pd.DataFrame({
            "Date": dates_test,
            "fold": fold_i,
            "y_true": y_test,
            "pred_naive": pred_naive,
            "pred_majority": pred_majority,
            "pred_logreg": pred_lr, "prob_logreg": prob_lr,
            "pred_rf": pred_rf, "prob_rf": prob_rf,
            "pred_xgb": pred_xgb, "prob_xgb": prob_xgb,
        }))

        if fold_i == len(folds) - 1:
            xgb_model.save_model(str(ARTIFACTS / "xgb_model.json"))
            import joblib
            joblib.dump(scaler, ARTIFACTS / "scaler.joblib")
            joblib.dump(logreg, ARTIFACTS / "logreg_model.joblib")
            joblib.dump(rf, ARTIFACTS / "rf_model.joblib")
            with open(ARTIFACTS / "feature_cols.json", "w") as f:
                json.dump(FEATURE_COLS, f)

    preds_df = pd.concat(all_preds, ignore_index=True)
    preds_df.to_parquet(PROC / "predictions.parquet", index=False)

    summary = {}
    for model_name, fold_results in results.items():
        avg = {k: float(np.mean([r.get(k, np.nan) for r in fold_results])) for k in ("accuracy", "f1", "roc_auc")}
        summary[model_name] = {"per_fold": fold_results, "average": avg}

    with open(PROC / "model_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"{'Model':<15} {'Avg Accuracy':<15} {'Avg F1':<10} {'Avg ROC-AUC':<12}")
    for model_name, res in summary.items():
        a = res["average"]
        auc = a.get("roc_auc")
        auc_str = f"{auc:.3f}" if auc is not None and not np.isnan(auc) else "n/a"
        print(f"{model_name:<15} {a['accuracy']:<15.3f} {a['f1']:<10.3f} {auc_str:<12}")

    print(f"\nOut-of-sample predictions saved: {PROC / 'predictions.parquet'} ({len(preds_df)} rows)")
    print(f"Metrics saved: {PROC / 'model_metrics.json'}")
    print(f"Final-fold model artifacts saved to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
