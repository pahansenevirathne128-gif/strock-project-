import pandas as pd
import numpy as np

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: OHLCV dataframe with Date index.
    Output: dataframe with engineered features + targets:
      y_close_next (regression target)
      y_up_next    (classification target: 1 if next close > today close else 0)
    """
    out = df.copy()

    # Returns
    out["ret_1"] = out["Close"].pct_change(1 , fill_method=None)
    out["ret_5"] = out["Close"].pct_change(5,fill_method=None)
    out["ret_10"] = out["Close"].pct_change(10, fill_method=None)

    # Moving averages
    out["ma_5"] = out["Close"].rolling(5).mean()
    out["ma_20"] = out["Close"].rolling(20).mean()
    out["ma_50"] = out["Close"].rolling(50).mean()
    out["close_vs_ma20"] = out["Close"] / out["ma_20"] - 1.0
    out["close_vs_ma50"] = out["Close"] / out["ma_50"] - 1.0

    # Volatility
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # High-low range
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]

    # Volume change (safe against zeros -> prevents inf)
    if "Volume" in out.columns:
        vol_safe = out["Volume"].replace(0, np.nan)
        out["vol_chg_5"] = vol_safe.pct_change(5, fill_method = None)

    # Calendar features
    out["dow"] = out.index.dayofweek  # 0=Mon..4=Fri
    out["month"] = out.index.month

    # Targets (next day)
    out["y_close_next"] = out["Close"].shift(-1)
    out["y_up_next"] = (out["Close"].shift(-1) > out["Close"]).astype(int)

    # ✅ Critical: remove infinities produced by any division/pct_change
    out = out.replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaNs from rolling/shift/inf cleanup
    out = out.dropna()

    return out


def split_xy(feat_df: pd.DataFrame):
    feature_cols = [c for c in feat_df.columns if c not in ["y_close_next", "y_up_next"]]
    X = feat_df[feature_cols]
    y_reg = feat_df["y_close_next"]
    y_clf = feat_df["y_up_next"]
    return X, y_reg, y_clf, feature_cols