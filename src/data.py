import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Downloads daily OHLCV data using yfinance and normalizes column names so
    we always get: Open, High, Low, Close, Adj Close, Volume
    """
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # --- FIX: handle MultiIndex columns (common with yfinance) ---
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)

        # If ticker is in level 0, select it
        if ticker in set(lvl0):
            df = df.xs(ticker, axis=1, level=0, drop_level=True)
        # If ticker is in level 1, select it
        elif ticker in set(lvl1):
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            # fallback: flatten columns
            df.columns = [" ".join([str(a), str(b)]).strip() for a, b in df.columns]

    # --- Standardize column names (case/spacing) ---
    df.columns = [str(c).strip() for c in df.columns]

    # Case-insensitive renaming to the exact expected names
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}

    def map_col(want: str):
        key = want.lower()
        if key in lower_map and lower_map[key] != want:
            rename[lower_map[key]] = want

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        map_col(col)

    if rename:
        df = df.rename(columns=rename)

    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found. Columns are: {list(df.columns)}")

    df = df.dropna(subset=["Close"])
    return df