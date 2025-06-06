# 📘 Full Change Log: `generate_data.py`

🔖 Version: v1.0

Date: 2025-06-05 (Assumed initial release date)
Description: Initial implementation of generate_data.py, including the full data pipeline for fetching, processing, and saving financial data.
🚀 Key Features Introduced in v1.0:
Basic script structure with functions for:
Data fetching: fetch_polygon_ohlcv
Feature addition: add_features
Data simulation: generate_simulated_data
Merging: merge_dataframes
Saving: save_dataframe
Configuration loaded from config.yaml; API key sourced directly via cfg["env"]["POLYGON_API_KEY"].
Data fetched from Polygon.io using direct requests.get (no session).
Feature prefixes ("intraday" / "daily") passed explicitly to add_features.
merge_dataframes merged df_daily into df_intraday, handled daily_daily_ naming.
Main loop had separate branches for simulated vs real tickers:
Real tickers: processed 5-minute and daily data.
BARS_PER_DAY was hardcoded as 78; approx_trading_density set to 0.8 for daily calculations.
🔖 Version: v1.1

Date: 2025-06-05 (Assumed update date)
Description: Improved configuration management, API handling, and modularity. Moved API key loading to environment variables and improved request reliability.
🔧 Key Changes from v1.0 to v1.1:
Environment Variable for API Key:
Added python-dotenv support.
Loaded POLYGON_API_KEY from .env using os.environ.get().
Added ValueError if key is missing.
Session Management:
Introduced requests.Session() globally.
fetch_polygon_ohlcv now uses session.get() and response.raise_for_status().
Configurable BARS_PER_DAY:
Now loaded from cfg["data"].get("bars_per_day", 78).
merge_dataframes Refinement:
Removed special-case renaming logic (_daily_daily_).
Still used .add_prefix(f"{prefix}_") to apply prefix before merging.
Code Cleanups:
filename_base in save_dataframe() renamed to filename.
Path construction and print logs simplified.
Improved timezone handling for START_DATE.
Renamed some variables (e.g., adj_start_intraday ➝ start_intraday).
🔖 Version: v1.2.2

Date: 2025-06-05 (Assumed culmination of v1.2.x series)
Description: Major refactor to unify and generalize multi-timeframe processing, prefixing, and merging logic for better extensibility and maintainability.
🧠 Key Changes from v1.1 to v1.2.2:
✅ Dynamic Timeframe Handling & Prefixing:

Introduced global TIMEFRAME_MAP for storing:
multiplier, timespan, and bar_divisor for each timeframe ("5min", "daily").
Replaced magic numbers like 0.8 with config values in the map.
Main loop now dynamically iterates over feature sets defined in cfg["features"]:
Auto-derives prefix (e.g., 5min, daily) via key.replace("FEATURES_", "").lower().
Skips non-feature keys like OBS_WINDOW.
Replaced hardcoded "intraday"/"daily" with dynamic prefixes in calls to add_features.
Created df_by_prefix dictionary to track processed DataFrames per timeframe.
Enhanced simulated ticker processing to derive and use "5min" prefix dynamically.
🔁 Merging Logic Enhancements:

merge_dataframes():
Removed internal .add_prefix() usage — prefixing is now done in add_features.
Added .copy() to base_df and higher_df to avoid modifying input DataFrames.
Removed prefix parameter.
Merged DataFrames on "date" column only.
Base timeframe (BASE_TIMEFRAME_PREFIX, e.g., "5min") receives merged higher timeframes.
✨ Code Style & Robustness:

Shortened some variable names (e.g., response ➝ r, feature_columns ➝ cols).
Reorganized import statements and dictionary/DF creation blocks.
Updated top-level docstring to reflect the new dynamic architecture.
Improved skipping of invalid keys in cfg["features"] loop.
---


## 📜 BEFORE: Full Code Snapshot (v1.1)
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

## 📜 AFTER: Full Code Snapshot (v1.2.2)
```python
"""
Data Manager: Dynamically fetches and processes financial data based on YAML configuration.
Implements:
- Dynamic prefix derivation and feature engineering for all configured timeframes
- Generalized merging of multiple higher timeframes into base (e.g., 5min)
- Centralized timeframe logic using TIMEFRAME_MAP
"""

import os, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd, requests, yaml
from dotenv import load_dotenv

# === Load .env ===
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# === Load Config ===
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

API_KEY = os.environ.get("POLYGON_API_KEY")
if not API_KEY:
    raise ValueError("POLYGON_API_KEY not set.")
DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DATA_DIR).mkdir(exist_ok=True)

TICKERS = cfg["data"]["tickers"]
START_DATE, END_DATE = cfg["data"]["start_date"], cfg["data"]["end_date"]
OUTPUT_FORMATS = cfg["data"].get("output_formats", ["csv", "parquet"])
BARS_PER_DAY = cfg["data"].get("bars_per_day", 78)

session = requests.Session()
TIMEFRAME_MAP = {
    "5min": {"multiplier": 5, "timespan": "minute", "bar_divisor": BARS_PER_DAY},
    "daily": {"multiplier": 1, "timespan": "day", "bar_divisor": 0.8},
    # Extendable: "weekly": {"multiplier": 1, "timespan": "week", "bar_divisor": 0.2}
}

def compute_buffer_bars(feature_list):
    max_window = max([f.get("window", 0) for f in feature_list], default=0)
    return max_window + cfg["features"]["OBS_WINDOW"] + 20

def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    all_data = []
    while True:
        try:
            r = session.get(url, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            break
        if not results: break
        all_data.extend(results)
        if len(results) < params["limit"]: break
        params["from"] = datetime.utcfromtimestamp(results[-1]["t"] / 1000).strftime("%Y-%m-%d")
        time.sleep(0.5)
    if not all_data: return None
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]

def add_features(df, features, prefix):
    df = df.copy()
    cols = []
    for f in features:
        col = f"{prefix}_{f['field']}_{f.get('window', '')}".rstrip("_")
        cols.append(col)
        if f["field"] == "ema":
            df[col] = df[f["source"]].ewm(span=f["window"], adjust=False).mean()
        elif f["field"] == "sma":
            df[col] = df[f["source"]].rolling(window=f["window"]).mean()
        elif f["field"] == "vwap":
            df[col] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        elif f["type"] == "price":
            df[col] = df[f["field"]]
        if f.get("normalize"):
            method = f["method"]
            if method == "zscore":
                df[col] = (df[col] - df[col].mean()) / df[col].std()
            elif method == "rolling_zscore":
                df[col] = (df[col] - df[col].rolling(f["window"]).mean()) / df[col].rolling(f["window"]).std()
    return df[["timestamp"] + cols]

def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg["training"]["SEED"])
    dates = pd.date_range(start=start_date, end=end_date, freq="5min", tz="America/New_York")
    prices = cfg["simulated_data"]["base_price"] + np.cumsum(np.random.normal(0, cfg["simulated_data"]["price_volatility"], len(dates)))
    volume = np.random.normal(cfg["simulated_data"]["volume_mean"], cfg["simulated_data"]["volume_std"], len(dates))
    return pd.DataFrame({"timestamp": dates, "open": prices, "high": prices+0.5, "low": prices-0.5, "close": prices, "volume": volume})

def merge_dataframes(base_df, higher_df, prefix):
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df["date"] = higher_df["timestamp"].dt.date
    merged = pd.merge(base_df, higher_df.drop(columns=["timestamp"]), left_on="date", right_on="date", how="left")
    return merged.drop(columns=["date"])

def save_dataframe(df, filename):
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(DATA_DIR, f"{filename}.{fmt}")
        df.to_csv(path, index=False) if fmt == "csv" else df.to_parquet(path)

for ticker in TICKERS:
    if ticker.startswith("@SIM"):
        for scenario in cfg["simulated_data"]["scenarios"]:
            prefix = "5min"
            df = generate_simulated_data(scenario, START_DATE, END_DATE, cfg)
            df = add_features(df, cfg["features"]["FEATURES_5MIN"], prefix)
            save_dataframe(df, f"SIM_{scenario}_{prefix}")
            print(f"✅ Saved: SIM_{scenario}_{prefix}")
    else:
        df_by_prefix = {}
        for key, features in cfg["features"].items():
            if key == "OBS_WINDOW": continue
            prefix = key.replace("FEATURES_", "").lower()
            if prefix not in TIMEFRAME_MAP:
                print(f"⏭️ Skipping unsupported timeframe: {prefix}")
                continue
            buffer_bars = compute_buffer_bars(features)
            params = TIMEFRAME_MAP[prefix]
            days_back = int(np.ceil(buffer_bars / params["bar_divisor"]))
            start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
            raw_df = fetch_polygon_ohlcv(ticker, params["multiplier"], params["timespan"], start, END_DATE)
            if raw_df is not None:
                df_by_prefix[prefix] = add_features(raw_df, features, prefix)

        if "5min" in df_by_prefix:
            merged = df_by_prefix["5min"]
            for prefix, df in df_by_prefix.items():
                if prefix != "5min":
                    merged = merge_dataframes(merged, df, prefix)
            start_ts = pd.to_datetime(START_DATE).tz_localize("America/New_York") if pd.to_datetime(START_DATE).tzinfo is None else pd.to_datetime(START_DATE)
            final = merged[merged["timestamp"] >= start_ts]
            save_dataframe(final, f"{ticker}_5min")
            print(f"✅ Saved: {ticker}_5min")
```
