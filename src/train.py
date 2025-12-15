from pathlib import Path
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

from src.data import fetch_ohlcv
from src.features import make_features, split_xy

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    ticker = "^GSPC"   # S&P 500 index; if you want cleaner volume features use "SPY"
    start = "2000-01-01"

    df = fetch_ohlcv(ticker, start=start)
    feat_df = make_features(df)
    X, y_reg, y_clf, feature_cols = split_xy(feat_df)

    # Walk-forward style CV
    tscv = TimeSeriesSplit(n_splits=5)

    reg_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])

    clf_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            max_depth=6
        ))
    ])

    # Evaluate via CV (time-series split)
    reg_rmses, reg_maes, reg_r2s = [], [], []
    clf_accs, clf_aucs = [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        yreg_train, yreg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        yclf_train, yclf_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        reg_model.fit(X_train, yreg_train)
        yreg_pred = reg_model.predict(X_test)

        reg_rmses.append(rmse(yreg_test, yreg_pred))
        reg_maes.append(float(mean_absolute_error(yreg_test, yreg_pred)))
        reg_r2s.append(float(r2_score(yreg_test, yreg_pred)))

        clf_model.fit(X_train, yclf_train)
        yclf_prob = clf_model.predict_proba(X_test)[:, 1]
        yclf_pred = (yclf_prob >= 0.5).astype(int)

        clf_accs.append(float(accuracy_score(yclf_test, yclf_pred)))
        # roc_auc needs both classes present; handle edge cases
        try:
            clf_aucs.append(float(roc_auc_score(yclf_test, yclf_prob)))
        except ValueError:
            pass

        print(f"\nFold {fold}")
        print(f"  REG  RMSE={reg_rmses[-1]:.4f}  MAE={reg_maes[-1]:.4f}  R2={reg_r2s[-1]:.4f}")
        print(f"  CLF  ACC ={clf_accs[-1]:.4f}  AUC={(clf_aucs[-1] if clf_aucs else float('nan')):.4f}")

    print("\n=== CV Summary ===")
    print(f"REG  RMSE mean={np.mean(reg_rmses):.4f}  MAE mean={np.mean(reg_maes):.4f}  R2 mean={np.mean(reg_r2s):.4f}")
    print(f"CLF  ACC  mean={np.mean(clf_accs):.4f}  AUC mean={(np.mean(clf_aucs) if clf_aucs else float('nan')):.4f}")

    # Fit on full data at the end (so we can save a “latest” model)
    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(
        {"model": reg_model, "features": feature_cols, "ticker": ticker, "target": "next_close"},
        models_dir / "reg_next_close.joblib"
    )
    joblib.dump(
        {"model": clf_model, "features": feature_cols, "ticker": ticker, "target": "prob_up_next_day"},
        models_dir / "clf_prob_up.joblib"
    )

    # Make a “today” prediction (last available row)
    x_last = X.iloc[[-1]]
    pred_close = float(reg_model.predict(x_last)[0])
    prob_up = float(clf_model.predict_proba(x_last)[0, 1])
    last_close = float(df["Close"].iloc[-1])

    print("\n=== Latest Prediction ===")
    print(f"Last close: {last_close:.2f}")
    print(f"Predicted next close: {pred_close:.2f}")
    print(f"Probability next day UP: {prob_up:.3f}")

if __name__ == "__main__":
    main()