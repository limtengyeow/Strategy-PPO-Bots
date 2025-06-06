# 📦 Full Change Log: `generate_data.py`

## 🔖 Version: v1.1
- 📅 Date: 2025-06-05
- 🔀 Git Hash: [pending]
- 🌿 Branch: [pending]
- 📜 Description:
  Full refactor of `generate_data.py` to implement security, robustness, and performance recommendations.

---

## 🧠 Proposed Change

### What Will Change
- Load API key securely using `.env`
- Add session reuse with `requests.Session`
- Handle request errors with try/except
- Use `bars_per_day` from config
- Improve merging logic and timestamp handling

### Why It’s Needed
- To secure secrets, avoid hardcoded values, and improve stability

---

## 🔐 BEFORE: Full Code Snapshot

```python
"""
Data Manager: Fetches and processes financial data based on YAML configuration.
Handles:
- Fetching historical data from Polygon.io for multiple timeframes (intraday, daily, etc.)
- Generating synthetic/simulated data for '@SIM' tickers
- Applying buffer bars based on feature requirements (e.g., SMA/EMA windows)
- Merging multi-timeframe data into a single dataset aligned with the lowest timeframe
- Applying feature-specific normalization as per configuration
- Saving outputs in multiple formats (CSV, Parquet)
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

# === Load Config ===
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

API_KEY = cfg["env"]["POLYGON_API_KEY"]
DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DATA_DIR).mkdir(exist_ok=True)

TICKERS = cfg["data"]["tickers"]
START_DATE = cfg["data"]["start_date"]
END_DATE = cfg["data"]["end_date"]
OUTPUT_FORMATS = cfg["data"].get("output_formats", ["csv", "parquet"])


def compute_buffer_bars(feature_list):
    max_window = max([f.get("window", 0) for f in feature_list], default=0)
    return max_window + cfg["features"]["OBS_WINDOW"] + 20


def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    all_data = []

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching {ticker}: {response.text}")
            break
        results = response.json().get("results", [])
        if not results:
            break
        all_data.extend(results)
        if len(results) < params["limit"]:
            break
        last_ts = results[-1]["t"]
        next_from = datetime.utcfromtimestamp(last_ts / 1000.0).strftime("%Y-%m-%d")
        params["from"] = next_from
        time.sleep(0.5)

    if not all_data:
        print(f"No data found for {ticker}")
        return None

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def add_features(df, features, prefix):
    df = df.copy()
    feature_columns = []
    for f in features:
        col_name = f"{prefix}_{f['field']}_{f.get('window', '')}".rstrip("_")
        feature_columns.append(col_name)
        if f["field"] == "ema":
            df[col_name] = df[f["source"]].ewm(span=f["window"], adjust=False).mean()
        elif f["field"] == "sma":
            df[col_name] = df[f["source"]].rolling(window=f["window"]).mean()
        elif f["field"] == "vwap":
            df[col_name] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        elif f["type"] == "price":
            df[col_name] = df[f["field"]]
        if f.get("normalize"):
            method = f["method"]
            if method == "zscore":
                df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
            elif method == "rolling_zscore":
                df[col_name] = (df[col_name] - df[col_name].rolling(f["window"]).mean()) / df[col_name].rolling(f["window"]).std()
    return df[["timestamp"] + feature_columns]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg["training"]["SEED"])
    dates = pd.date_range(start=start_date, end=end_date, freq="5min", tz="America/New_York")
    prices = cfg["simulated_data"]["base_price"] + np.cumsum(np.random.normal(0, cfg["simulated_data"]["price_volatility"], size=len(dates)))
    volume = np.random.normal(cfg["simulated_data"]["volume_mean"], cfg["simulated_data"]["volume_std"], size=len(dates))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": volume,
    })
    return df


def merge_dataframes(base_df, higher_df, prefix):
    higher_df = higher_df.copy()
    higher_df["date"] = higher_df["timestamp"].dt.date
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df = higher_df.add_prefix(f"{prefix}_")
    merged = pd.merge(base_df, higher_df, left_on="date", right_on=f"{prefix}_date", how="left").drop(columns=[f"{prefix}_date", "date"])
    if f"{prefix}_timestamp" in merged.columns:
        merged = merged.drop(columns=[f"{prefix}_timestamp"])
    merged = merged.rename(
        columns={
            col: col.replace(f"{prefix}_daily_", f"{prefix}_")
            for col in merged.columns
            if col.startswith(f"{prefix}_daily_")
        }
    )
    feature_cols = [col for col in merged.columns if col != "timestamp"]
    return merged[["timestamp"] + feature_cols]


def save_dataframe(df, filename_base):
    for fmt in OUTPUT_FORMATS:
        if fmt == "parquet":
            df.to_parquet(os.path.join(DATA_DIR, f"{filename_base}.parquet"))
        elif fmt == "csv":
            df.to_csv(os.path.join(DATA_DIR, f"{filename_base}.csv"), index=False)
        else:
            print(f"⚠️ Unsupported output format: {fmt}")


for ticker in TICKERS:
    if ticker.startswith("@SIM"):
        for scenario in cfg["simulated_data"]["scenarios"]:
            df = generate_simulated_data(scenario, START_DATE, END_DATE, cfg)
            df = add_features(df, cfg["features"]["FEATURES_5MIN"], "intraday")
            save_dataframe(df, f"SIM_{scenario}_intraday")
            print(f"✅ Saved simulated data: SIM_{scenario}_intraday in {OUTPUT_FORMATS}")
    else:
        bars_per_day = 78
        buffer_intraday = compute_buffer_bars(cfg["features"]["FEATURES_5MIN"])
        buffer_days_intraday = int(np.ceil(buffer_intraday / bars_per_day))
        adj_start_intraday = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=buffer_days_intraday)).strftime("%Y-%m-%d")

        buffer_daily = compute_buffer_bars(cfg["features"]["FEATURES_DAILY"])
        approx_trading_density = 0.8
        buffer_days_daily = int(buffer_daily / approx_trading_density)
        adj_start_daily = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=buffer_days_daily)).strftime("%Y-%m-%d")

        df_intraday = fetch_polygon_ohlcv(ticker, 5, "minute", adj_start_intraday, END_DATE)
        if df_intraday is not None:
            df_intraday = add_features(df_intraday, cfg["features"]["FEATURES_5MIN"], "intraday")

        df_daily = fetch_polygon_ohlcv(ticker, 1, "day", adj_start_daily, END_DATE)
        if df_daily is not None:
            df_daily = add_features(df_daily, cfg["features"]["FEATURES_DAILY"], "daily")

        if df_intraday is not None and df_daily is not None:
            df_merged = merge_dataframes(df_intraday, df_daily, "daily")
            df_final = df_merged[df_merged["timestamp"] >= pd.to_datetime(START_DATE).tz_localize("America/New_York")]
            save_dataframe(df_final, f"{ticker}_intraday")
            print(f"✅ Saved merged data: {ticker}_intraday in {OUTPUT_FORMATS}")

```

---

## ✨ AFTER: Full Code Snapshot

```python
"""
Data Manager: Fetches and processes financial data based on YAML configuration.
Handles:
- Fetching historical data from Polygon.io for multiple timeframes (intraday, daily, etc.)
- Generating synthetic/simulated data for '@SIM' tickers
- Applying buffer bars based on feature requirements (e.g., SMA/EMA windows)
- Merging multi-timeframe data into a single dataset aligned with the lowest timeframe
- Applying feature-specific normalization as per configuration
- Saving outputs in multiple formats (CSV, Parquet)
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

# === Load .env ===
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# === Load Config ===
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY environment variable not set.")

DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DATA_DIR).mkdir(exist_ok=True)

TICKERS = cfg["data"]["tickers"]
START_DATE = cfg["data"]["start_date"]
END_DATE = cfg["data"]["end_date"]
OUTPUT_FORMATS = cfg["data"].get("output_formats", ["csv", "parquet"])
BARS_PER_DAY = cfg["data"].get("bars_per_day", 78)

session = requests.Session()


def compute_buffer_bars(feature_list):
    max_window = max([f.get("window", 0) for f in feature_list], default=0)
    return max_window + cfg["features"]["OBS_WINDOW"] + 20


def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    all_data = []

    while True:
        try:
            response = session.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            break
        if not results:
            break
        all_data.extend(results)
        if len(results) < params["limit"]:
            break
        last_ts = results[-1]["t"]
        next_from = datetime.utcfromtimestamp(last_ts / 1000.0).strftime("%Y-%m-%d")
        params["from"] = next_from
        time.sleep(0.5)

    if not all_data:
        print(f"No data found for {ticker}")
        return None

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def add_features(df, features, prefix):
    df = df.copy()
    feature_columns = []
    for f in features:
        col_name = f"{prefix}_{f['field']}_{f.get('window', '')}".rstrip("_")
        feature_columns.append(col_name)
        if f["field"] == "ema":
            df[col_name] = df[f["source"]].ewm(span=f["window"], adjust=False).mean()
        elif f["field"] == "sma":
            df[col_name] = df[f["source"]].rolling(window=f["window"]).mean()
        elif f["field"] == "vwap":
            df[col_name] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        elif f["type"] == "price":
            df[col_name] = df[f["field"]]
        if f.get("normalize"):
            method = f["method"]
            if method == "zscore":
                df[col_name] = (df[col_name] - df[col_name].mean()) / df[col_name].std()
            elif method == "rolling_zscore":
                df[col_name] = (df[col_name] - df[col_name].rolling(f["window"]).mean()) / df[col_name].rolling(f["window"]).std()
    return df[["timestamp"] + feature_columns]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg["training"]["SEED"])
    dates = pd.date_range(start=start_date, end=end_date, freq="5min", tz="America/New_York")
    prices = cfg["simulated_data"]["base_price"] + np.cumsum(np.random.normal(0, cfg["simulated_data"]["price_volatility"], size=len(dates)))
    volume = np.random.normal(cfg["simulated_data"]["volume_mean"], cfg["simulated_data"]["volume_std"], size=len(dates))
    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": prices + 0.5,
        "low": prices - 0.5,
        "close": prices,
        "volume": volume,
    })
    return df


def merge_dataframes(base_df, higher_df, prefix):
    base_df = base_df.copy()
    higher_df = higher_df.copy()
    higher_df["date"] = higher_df["timestamp"].dt.date
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df = higher_df.drop(columns=["timestamp"]).add_prefix(f"{prefix}_")
    merged = pd.merge(base_df, higher_df, left_on="date", right_on=f"{prefix}_date", how="left")
    return merged.drop(columns=["date", f"{prefix}_date"])


def save_dataframe(df, filename):
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(DATA_DIR, f"{filename}.{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path)
        else:
            print(f"⚠️ Unsupported output format: {fmt}")


for ticker in TICKERS:
    if ticker.startswith("@SIM"):
        for scenario in cfg["simulated_data"]["scenarios"]:
            df = generate_simulated_data(scenario, START_DATE, END_DATE, cfg)
            df = add_features(df, cfg["features"]["FEATURES_5MIN"], "intraday")
            save_dataframe(df, f"SIM_{scenario}_intraday")
            print(f"✅ Saved: SIM_{scenario}_intraday")
    else:
        buffer_intraday = compute_buffer_bars(cfg["features"]["FEATURES_5MIN"])
        start_intraday = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=int(np.ceil(buffer_intraday / BARS_PER_DAY)))).strftime("%Y-%m-%d")

        buffer_daily = compute_buffer_bars(cfg["features"]["FEATURES_DAILY"])
        start_daily = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=int(buffer_daily / 0.8))).strftime("%Y-%m-%d")

        df_intraday = fetch_polygon_ohlcv(ticker, 5, "minute", start_intraday, END_DATE)
        df_daily = fetch_polygon_ohlcv(ticker, 1, "day", start_daily, END_DATE)

        if df_intraday is not None and df_daily is not None:
            df_intraday = add_features(df_intraday, cfg["features"]["FEATURES_5MIN"], "intraday")
            df_daily = add_features(df_daily, cfg["features"]["FEATURES_DAILY"], "daily")
            merged = merge_dataframes(df_intraday, df_daily, "daily")
            start_ts = pd.to_datetime(START_DATE).tz_localize("America/New_York") if pd.to_datetime(START_DATE).tzinfo is None else pd.to_datetime(START_DATE)
            final = merged[merged["timestamp"] >= start_ts]
            save_dataframe(final, f"{ticker}_intraday")
            print(f"✅ Saved: {ticker}_intraday")

```

---
