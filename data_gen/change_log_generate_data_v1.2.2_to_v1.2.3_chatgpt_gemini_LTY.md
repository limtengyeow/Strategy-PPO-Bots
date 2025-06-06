# 📈 Change Log: `generate_data.py`

## 🔖 Version: v1.2.3 (Final)
- 🗓️ Date: 2025-06-05
- 🌿 Branch: [pending]
- 🔀 Git Hash: [pending]
- 🧩 Module: `data_gen/generate_data.py`
- 👤 Reviewer: Gemini
- ✍️ Author: ChatGPT
- 🧪 Status: ✅ Reviewed and Passed

---

## 📝 Description

**Patch v1.2.3 (Final)** removes the hardcoded `"5min"` timeframe and introduces dynamic base timeframe selection for merging. This version also ensures:
- ✅ All feature prefixes are respected
- ✅ The most granular timeframe is used as the merge base
- ✅ Output filenames and log messages reflect the correct timeframe dynamically

This improves flexibility and future-proofs the system to accommodate additional timeframes like `"weekly"` or `"monthly"`.

---

## 🔧 What Changed

- Removed:
```python
if "5min" in df_by_prefix:
    merged = df_by_prefix["5min"]
    for prefix, df in df_by_prefix.items():
        if prefix != "5min":
            merged = merge_dataframes(merged, df, prefix)
    ...
    save_dataframe(final, f"{ticker}_5min")
    print(f"✅ Saved: {ticker}_5min")
```

- Replaced with:
```python
if df_by_prefix:
    timeframe_order = {
        "1min": 0, "5min": 1, "15min": 2, "30min": 3, "hour": 4,
        "daily": 5, "weekly": 6, "monthly": 7
    }
    base_prefix = min(df_by_prefix.keys(), key=lambda x: timeframe_order.get(x, float('inf')))
    merged = df_by_prefix[base_prefix]
    for prefix, df in df_by_prefix.items():
        if prefix != base_prefix:
            merged = merge_dataframes(merged, df, prefix)
    ...
    save_dataframe(final, f"{ticker}_{base_prefix}")
    print(f"✅ Saved: {ticker}_{base_prefix}")
```

---

## ⏮️ BEFORE: Full Code Snapshot (v1.2.2)

```python
"""
Data Manager: Dynamically fetches and processes financial data based on YAML configuration.
Implements:
- Dynamic prefix derivation and feature engineering for all configured timeframes
- Generalized merging of multiple higher timeframes into base (e.g., 5min)
- Centralized timeframe logic using TIMEFRAME_MAP
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
        if not results:
            break
        all_data.extend(results)
        if len(results) < params["limit"]:
            break
        params["from"] = datetime.utcfromtimestamp(results[-1]["t"] / 1000).strftime(
            "%Y-%m-%d"
        )
        time.sleep(0.5)
    if not all_data:
        return None
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(
        "America/New_York"
    )
    return df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )[["timestamp", "open", "high", "low", "close", "volume"]]


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
                df[col] = (df[col] - df[col].rolling(f["window"]).mean()) / df[
                    col
                ].rolling(f["window"]).std()
    return df[["timestamp"] + cols]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg["training"]["SEED"])
    dates = pd.date_range(
        start=start_date, end=end_date, freq="5min", tz="America/New_York"
    )
    prices = cfg["simulated_data"]["base_price"] + np.cumsum(
        np.random.normal(0, cfg["simulated_data"]["price_volatility"], len(dates))
    )
    volume = np.random.normal(
        cfg["simulated_data"]["volume_mean"],
        cfg["simulated_data"]["volume_std"],
        len(dates),
    )
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": volume,
        }
    )


def merge_dataframes(base_df, higher_df, prefix):
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df["date"] = higher_df["timestamp"].dt.date
    merged = pd.merge(
        base_df,
        higher_df.drop(columns=["timestamp"]),
        left_on="date",
        right_on="date",
        how="left",
    )
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
            if key == "OBS_WINDOW":
                continue
            prefix = key.replace("FEATURES_", "").lower()
            if prefix not in TIMEFRAME_MAP:
                print(f"⏭️ Skipping unsupported timeframe: {prefix}")
                continue
            buffer_bars = compute_buffer_bars(features)
            params = TIMEFRAME_MAP[prefix]
            days_back = int(np.ceil(buffer_bars / params["bar_divisor"]))
            start = (
                datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=days_back)
            ).strftime("%Y-%m-%d")
            raw_df = fetch_polygon_ohlcv(
                ticker, params["multiplier"], params["timespan"], start, END_DATE
            )
            if raw_df is not None:
                df_by_prefix[prefix] = add_features(raw_df, features, prefix)

        if "5min" in df_by_prefix:
            merged = df_by_prefix["5min"]
            for prefix, df in df_by_prefix.items():
                if prefix != "5min":
                    merged = merge_dataframes(merged, df, prefix)
            start_ts = (
                pd.to_datetime(START_DATE).tz_localize("America/New_York")
                if pd.to_datetime(START_DATE).tzinfo is None
                else pd.to_datetime(START_DATE)
            )
            final = merged[merged["timestamp"] >= start_ts]
            save_dataframe(final, f"{ticker}_5min")
            print(f"✅ Saved: {ticker}_5min")

```

---

## ⏭️ AFTER: Full Code Snapshot (v1.2.3)

```python
"""
Data Manager: Dynamically fetches and processes financial data based on YAML configuration.
Implements:
- Dynamic prefix derivation and feature engineering for all configured timeframes
- Generalized merging of multiple higher timeframes into base (e.g., 5min)
- Centralized timeframe logic using TIMEFRAME_MAP
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
        if not results:
            break
        all_data.extend(results)
        if len(results) < params["limit"]:
            break
        params["from"] = datetime.utcfromtimestamp(results[-1]["t"] / 1000).strftime(
            "%Y-%m-%d"
        )
        time.sleep(0.5)
    if not all_data:
        return None
    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(
        "America/New_York"
    )
    return df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )[["timestamp", "open", "high", "low", "close", "volume"]]


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
                df[col] = (df[col] - df[col].rolling(f["window"]).mean()) / df[
                    col
                ].rolling(f["window"]).std()
    return df[["timestamp"] + cols]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg["training"]["SEED"])
    dates = pd.date_range(
        start=start_date, end=end_date, freq="5min", tz="America/New_York"
    )
    prices = cfg["simulated_data"]["base_price"] + np.cumsum(
        np.random.normal(0, cfg["simulated_data"]["price_volatility"], len(dates))
    )
    volume = np.random.normal(
        cfg["simulated_data"]["volume_mean"],
        cfg["simulated_data"]["volume_std"],
        len(dates),
    )
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
            "volume": volume,
        }
    )


def merge_dataframes(base_df, higher_df, prefix):
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df["date"] = higher_df["timestamp"].dt.date
    merged = pd.merge(
        base_df,
        higher_df.drop(columns=["timestamp"]),
        left_on="date",
        right_on="date",
        how="left",
    )
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
            if key == "OBS_WINDOW":
                continue
            prefix = key.replace("FEATURES_", "").lower()
            if prefix not in TIMEFRAME_MAP:
                print(f"⏭️ Skipping unsupported timeframe: {prefix}")
                continue
            buffer_bars = compute_buffer_bars(features)
            params = TIMEFRAME_MAP[prefix]
            days_back = int(np.ceil(buffer_bars / params["bar_divisor"]))
            start = (
                datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=days_back)
            ).strftime("%Y-%m-%d")
            raw_df = fetch_polygon_ohlcv(
                ticker, params["multiplier"], params["timespan"], start, END_DATE
            )
            if raw_df is not None:
                df_by_prefix[prefix] = add_features(raw_df, features, prefix)

        
        if df_by_prefix:
            # Dynamically determine the base timeframe (lowest granularity)
            timeframe_order = {
                "1min": 0, "5min": 1, "15min": 2, "30min": 3, "hour": 4,
                "daily": 5, "weekly": 6, "monthly": 7
            }
            prefixes = list(df_by_prefix.keys())
            base_prefix = min(prefixes, key=lambda x: timeframe_order.get(x, float('inf')))

            # Start merging with the base timeframe
            merged = df_by_prefix[base_prefix]

            # Merge all other higher timeframes
            for prefix, df in df_by_prefix.items():
                if prefix != base_prefix:
                    merged = merge_dataframes(merged, df, prefix)

            # Filter out buffer data to start from the configured START_DATE
            start_ts = (
                pd.to_datetime(START_DATE).tz_localize("America/New_York")
                if pd.to_datetime(START_DATE).tzinfo is None
                else pd.to_datetime(START_DATE)
            )
            final = merged[merged["timestamp"] >= start_ts].reset_index(drop=True)

            # Save and print using the dynamic base_prefix
            output_filename = f"{ticker}_{base_prefix}"
            save_dataframe(final, output_filename)
            print(f"✅ Saved: {output_filename}")


```
