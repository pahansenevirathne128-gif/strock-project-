from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.data import fetch_ohlcv
from src.features import make_features, split_xy


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def main():
    # Config (Week 4: keep it explicit + repeatable)
    ticker = "^GSPC"   # SPY also fine if you want ETF volume behaviour
    start = "2000-01-01"
    n_splits = 5
    holdout_frac = 0.20
    random_state = 42

    # 1) Load data
    df = fetch_ohlcv(ticker, start=start)

    # 2) Make features + targets
    feat_df = make_features(df)
    X, y_reg, y_clf, feature_cols = split_xy(feat_df)

    if "Close" in feat_df.columns:
        close_today = feat_df.loc[X.index, "Close"].astype(float).values
        naive_rmse_all = rmse(y_reg.astype(float).values, close_today)
        print("\n=== Baseline (Naive persistence) ===")
        print(f"REG naive RMSE (all rows) = {naive_rmse_all:.4f}")

    # Safety checks (helps catch silent issues)
    if len(X) < 200:
        raise ValueError(f"Not enough rows after feature engineering: {len(X)}")

    # Ensure feature column names are unique
    if X.columns.duplicated().any():
        dupes = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns detected: {dupes}")

    # 3) Holdout split (Week 4: last chunk of time is test set)
    n = len(X)
    test_size = max(1, int(n * holdout_frac))
    train_end = n - test_size

    X_train_all, X_test = X.iloc[:train_end], X.iloc[train_end:]
    yreg_train_all, yreg_test = y_reg.iloc[:train_end], y_reg.iloc[train_end:]
    yclf_train_all, yclf_test = y_clf.iloc[:train_end], y_clf.iloc[train_end:]

    # 4) Time-series CV on TRAIN ONLY (no leakage)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Regression model: next-day close
    reg_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    # Classification model: probability next day is UP
    # Note: StandardScaler is unnecessary for RandomForest, so we remove it to avoid confusion.
    clf_model = Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            max_depth=6
        ))
    ])

    # CV metrics storage
    reg_rmses, reg_maes, reg_r2s = [], [], []
    clf_accs, clf_aucs, clf_briers, clf_precs, clf_recs = [], [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train_all), start=1):
        X_tr, X_va = X_train_all.iloc[tr_idx], X_train_all.iloc[va_idx]
        yreg_tr, yreg_va = yreg_train_all.iloc[tr_idx], yreg_train_all.iloc[va_idx]
        yclf_tr, yclf_va = yclf_train_all.iloc[tr_idx], yclf_train_all.iloc[va_idx]

        # --- regression ---
        reg_model.fit(X_tr, yreg_tr)
        yreg_pred = reg_model.predict(X_va)

        fold_rmse = rmse(yreg_va, yreg_pred)
        fold_mae = float(mean_absolute_error(yreg_va, yreg_pred))
        fold_r2 = float(r2_score(yreg_va, yreg_pred))

        reg_rmses.append(fold_rmse)
        reg_maes.append(fold_mae)
        reg_r2s.append(fold_r2)

        # --- classification ---
        clf_model.fit(X_tr, yclf_tr)

        yprob = clf_model.predict_proba(X_va)[:, 1]
        yhat = (yprob >= 0.5).astype(int)

        fold_acc = float(accuracy_score(yclf_va, yhat))
        fold_brier = float(brier_score_loss(yclf_va, yprob))
        fold_prec = float(precision_score(yclf_va, yhat, zero_division=0))
        fold_rec = float(recall_score(yclf_va, yhat, zero_division=0))

        # roc_auc requires both classes present in y_true in that fold
        try:
            fold_auc = float(roc_auc_score(yclf_va, yprob))
        except ValueError:
            fold_auc = float("nan")

        clf_accs.append(fold_acc)
        clf_briers.append(fold_brier)
        clf_precs.append(fold_prec)
        clf_recs.append(fold_rec)
        clf_aucs.append(fold_auc)

        print(f"\nFold {fold}")
        print(f"  REG  RMSE={fold_rmse:.4f}  MAE={fold_mae:.4f}  R2={fold_r2:.4f}")
        print(
            f"  CLF  ACC ={fold_acc:.4f}  AUC={fold_auc:.4f}  "
            f"BRIER={fold_brier:.4f}  PREC={fold_prec:.4f}  REC={fold_rec:.4f}"
        )

    print("\n=== CV Summary (TRAIN ONLY) ===")
    print(
        f"REG  RMSE mean={np.mean(reg_rmses):.4f}  "
        f"MAE mean={np.mean(reg_maes):.4f}  "
        f"R2 mean={np.mean(reg_r2s):.4f}"
    )
    print(
        f"CLF  ACC mean={np.mean(clf_accs):.4f}  "
        f"AUC mean={_safe_mean(clf_aucs):.4f}  "
        f"BRIER mean={np.mean(clf_briers):.4f}  "
        f"PREC mean={np.mean(clf_precs):.4f}  "
        f"REC mean={np.mean(clf_recs):.4f}"
    )

    # 5) Fit on TRAIN and evaluate on HOLDOUT test (Week 4: real “out-of-sample” check)
    reg_model.fit(X_train_all, yreg_train_all)
    clf_model.fit(X_train_all, yclf_train_all)

    yreg_test_pred = reg_model.predict(X_test)
    reg_test_rmse = rmse(yreg_test, yreg_test_pred)
    reg_test_mae = float(mean_absolute_error(yreg_test, yreg_test_pred))
    reg_test_r2 = float(r2_score(yreg_test, yreg_test_pred))

    yclf_test_prob = clf_model.predict_proba(X_test)[:, 1]
    yclf_test_pred = (yclf_test_prob >= 0.5).astype(int)

    clf_test_acc = float(accuracy_score(yclf_test, yclf_test_pred))
    clf_test_brier = float(brier_score_loss(yclf_test, yclf_test_prob))
    clf_test_prec = float(precision_score(yclf_test, yclf_test_pred, zero_division=0))
    clf_test_rec = float(recall_score(yclf_test, yclf_test_pred, zero_division=0))
    try:
        clf_test_auc = float(roc_auc_score(yclf_test, yclf_test_prob))
    except ValueError:
        clf_test_auc = float("nan")

    print("\n=== HOLDOUT Test (last 20% of time) ===")
    print(f"REG  RMSE={reg_test_rmse:.4f}  MAE={reg_test_mae:.4f}  R2={reg_test_r2:.4f}")
    print(
        f"CLF  ACC ={clf_test_acc:.4f}  AUC={clf_test_auc:.4f}  "
        f"BRIER={clf_test_brier:.4f}  PREC={clf_test_prec:.4f}  REC={clf_test_rec:.4f}"
    )

    # 6) Fit on FULL data (after evaluation) and save artifacts
    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)

    # Try to capture feature-date range if present
    date_start, date_end = None, None
    if "date" in feat_df.columns:
        try:
            d = feat_df.loc[X.index, "date"]
            date_start = str(d.iloc[0])
            date_end = str(d.iloc[-1])
        except Exception:
            pass

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "start": start,
        "n_rows": int(len(X)),
        "holdout_frac": holdout_frac,
        "date_start": date_start,
        "date_end": date_end,
        "feature_cols": feature_cols,
        "cv_train_only": {
            "reg_rmse_mean": float(np.mean(reg_rmses)),
            "reg_mae_mean": float(np.mean(reg_maes)),
            "reg_r2_mean": float(np.mean(reg_r2s)),
            "clf_acc_mean": float(np.mean(clf_accs)),
            "clf_auc_mean": _safe_mean(clf_aucs),
            "clf_brier_mean": float(np.mean(clf_briers)),
            "clf_prec_mean": float(np.mean(clf_precs)),
            "clf_rec_mean": float(np.mean(clf_recs)),
        },
        "holdout_test": {
            "reg_rmse": reg_test_rmse,
            "reg_mae": reg_test_mae,
            "reg_r2": reg_test_r2,
            "clf_acc": clf_test_acc,
            "clf_auc": clf_test_auc,
            "clf_brier": clf_test_brier,
            "clf_prec": clf_test_prec,
            "clf_rec": clf_test_rec,
        },
        "models": {
            "regressor": "Ridge(alpha=1.0) + StandardScaler",
            "classifier": "RandomForestClassifier(n_estimators=400, max_depth=6)",
            "random_state": random_state,
        },
    }

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        {"model": reg_model, "features": feature_cols, "ticker": ticker, "target": "next_close", "meta": meta},
        models_dir / "reg_next_close.joblib"
    )
    joblib.dump(
        {"model": clf_model, "features": feature_cols, "ticker": ticker, "target": "prob_up_next_day", "meta": meta},
        models_dir / "clf_prob_up.joblib"
    )

    # Optional: easy-to-read metadata file
    (models_dir / "week4_train_metadata.json").write_text(json.dumps(meta, indent=2))

    # 7) Latest prediction (based on most recent feature row)
    x_last = X.iloc[[-1]]
    pred_close = float(reg_model.predict(x_last)[0])
    prob_up = float(clf_model.predict_proba(x_last)[0, 1])

    # last close from raw df (aligns with your original behaviour)
    last_close = float(df["Close"].iloc[-1])

    print("\n=== Latest Prediction (trained on FULL data) ===")
    print(f"Ticker: {ticker}")
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted next close: {pred_close:.2f}")
    print(f"Probability next day UP: {prob_up:.3f}")

    print("\nSaved:")
    print(f" - {models_dir / 'reg_next_close.joblib'}")
    print(f" - {models_dir / 'clf_prob_up.joblib'}")
    print(f" - {models_dir / 'week4_train_metadata.json'}")


if __name__ == "__main__":
    main()