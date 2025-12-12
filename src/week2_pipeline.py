from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def download_spy(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Downloads SPY OHLCV data from Yahoo Finance.
    We use SPY as a practical proxy for the S&P 500 price series.
    """
    df = yf.download("SPY", start=start, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data returned. Check internet connection or ticker.")
    df = df.reset_index()
    # Standardize columns
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates leak-free features using only information available up to day t.
    Target will be created separately by shifting future close.
    """
    out = df.copy()

    # Core price column
    out["close"] = out["Close"].astype(float)

    # Simple returns (from t-1 to t)
    out["ret_1"] = out["close"].pct_change()

    # Moving averages computed at time t (only uses past and current)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_20"] = out["close"].rolling(20).mean()

    # Volatility proxy (rolling std of returns)
    out["vol_10"] = out["ret_1"].rolling(10).std()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # Volume features
    out["volume"] = out["Volume"].astype(float)
    out["vol_ma_10"] = out["volume"].rolling(10).mean()

    # Clean rows where rolling windows are not ready
    out = out.dropna().reset_index(drop=True)
    return out


def train_predict_direct_horizon(
    feat_df: pd.DataFrame,
    horizon: int,
    train_end_date: str = "2022-12-31",
) -> tuple[pd.DataFrame, float]:
    """
    Direct model: features at time t predict close at time t+horizon.
    Returns predictions dataframe and train residual std (for a simple probability estimate).
    """
    df = feat_df.copy()

    # Create target: future close
    df[f"y_h{horizon}"] = df["close"].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    feature_cols = ["close", "ret_1", "ma_5", "ma_10", "ma_20", "vol_10", "vol_20", "volume", "vol_ma_10"]

    train_mask = df["date"] <= pd.to_datetime(train_end_date)
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[~train_mask].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        raise RuntimeError("Not enough train/test data. Change train_end_date or start date.")

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df[f"y_h{horizon}"].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df[f"y_h{horizon}"].to_numpy()

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Residual std on TRAIN (used for simple probabilistic estimate)
    train_pred = model.predict(X_train)
    resid = y_train - train_pred
    resid_std = float(np.std(resid, ddof=1))
    if resid_std == 0.0:
        resid_std = 1e-9

    test_pred = model.predict(X_test)

    # Metrics
    rmse = math.sqrt(mean_squared_error(y_test, test_pred))
    mae = mean_absolute_error(y_test, test_pred)

    preds = test_df[["date", "close"]].copy()
    preds[f"pred_close_h{horizon}"] = test_pred
    preds[f"actual_close_h{horizon}"] = y_test
    preds[f"rmse_h{horizon}"] = rmse
    preds[f"mae_h{horizon}"] = mae

    # “Probability” example: P(future close > today close) under Normal(resid_std)
    # This is a simple baseline assumption (NOT a guarantee).
    z = (preds["close"] - preds[f"pred_close_h{horizon}"]) / resid_std  # P(Y > close_today) = 1 - Phi((close_today - pred)/std)
    x = (preds["close"] - preds[f"pred_close_h{horizon}"]) / resid_std
    preds[f"prob_up_h{horizon}"] = 1.0 - norm.cdf(x)

    return preds, resid_std


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 1) Download + save raw
    raw = download_spy(start="2000-01-01")
    raw_path = DATA_RAW / "spy_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Saved raw data -> {raw_path}")

    # 2) Features
    feat = make_features(raw)
    feat_path = DATA_PROCESSED / "spy_features.csv"
    feat.to_csv(feat_path, index=False)
    print(f"Saved features -> {feat_path}")

    # 3) Train + predict horizons 1..7
    all_preds = []
    for h in range(1, 8):
        preds_h, resid_std = train_predict_direct_horizon(feat, horizon=h, train_end_date="2022-12-31")
        preds_h[f"resid_std_train_h{h}"] = resid_std
        all_preds.append(preds_h)

        # Print one-line summary
        rmse = float(preds_h[f"rmse_h{h}"].iloc[0])
        mae = float(preds_h[f"mae_h{h}"].iloc[0])
        print(f"H{h}: RMSE={rmse:.3f}, MAE={mae:.3f}")

    # Merge predictions by date
    out = all_preds[0]
    for i in range(1, len(all_preds)):
        out = out.merge(all_preds[i].drop(columns=["close"]), on="date", how="inner")

    out_path = DATA_PROCESSED / "spy_predictions_h1_to_h7.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}")


if __name__ == "__main__":
    main()