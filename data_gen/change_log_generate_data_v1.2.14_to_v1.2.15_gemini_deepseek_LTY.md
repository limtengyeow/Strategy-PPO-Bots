📈 Change Log: generate_data.py
🔖 Version: v1.2.15 (Final)
🗓️ Date: 2025-06-06

🌿 Branch: [pending]

🔀 Git Hash: [pending]

🧩 Module: data_gen/generate_data.py

👤 Reviewer: Gemini

✍️ Author: ChatGPT & Gemini

🧪 Status: ✅ Reviewed and Passed

📝 Description
Version v1.2.15 simplifies the script's default behavior based on user feedback. It implements a "convention over configuration" approach for the final output.

By default, the script now automatically saves a clean dataset containing only the timestamp and the generated feature columns. The base OHLCV columns (open, high, low, close, volume) are no longer included unless explicitly requested.

To include the base OHLCV data or any other specific combination, the user can still leverage the optional output_columns key in config.yaml to override the new, cleaner default.

🔧 What Changed
1. Default Output Column Handling

Modified the final processing block to intelligently select output columns. If output_columns is not specified in the config.yaml, the script now defaults to saving only the timestamp and generated features, instead of all available columns.

⏮️ BEFORE: Full Code Snapshot (v1.2.14)
"""
Data Manager: Dynamically fetches and processes financial data based on YAML configuration.
Implements:
- Automated Two-Pass Architecture for handling different timeframe requirements.
- Robust buffer logic and request chunking to prevent API errors.
- Robust feature column naming to prevent duplicates.
- Optional filtering of final output columns based on config.
- Production-ready.
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

session = requests.Session()
TIMEFRAME_MAP = {
    "1min": {"multiplier": 1, "timespan": "minute", "bar_divisor": 390},
    "5min": {"multiplier": 5, "timespan": "minute", "bar_divisor": 78},
    "15min": {"multiplier": 15, "timespan": "minute", "bar_divisor": 26},
    "30min": {"multiplier": 30, "timespan": "minute", "bar_divisor": 13},
    "hour": {"multiplier": 60, "timespan": "minute", "bar_divisor": 6.5},
    "daily": {"multiplier": 1, "timespan": "day", "bar_divisor": 1},
    "weekly": {"multiplier": 1, "timespan": "week", "bar_divisor": 1/5},
    "monthly": {"multiplier": 1, "timespan": "month", "bar_divisor": 1/21},
}


def convert_trading_days_to_calendar_days(trading_days):
    return int(trading_days * 1.5)


def compute_buffer_bars(feature_list, apply_obs_window=True):
    if not feature_list:
        return 0
    max_window = max([f.get("window", 0) for f in feature_list], default=0)
    trading_days_needed = max_window + 20
    if apply_obs_window:
        trading_days_needed += cfg["features"].get("OBS_WINDOW", 50)
    return trading_days_needed


def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    all_data = []
    
    date_ranges = list(pd.date_range(start=from_date, end=to_date, freq='30D'))
    if pd.to_datetime(to_date).date() not in [d.date() for d in date_ranges]:
        date_ranges.append(pd.to_datetime(to_date))

    print(f"    Fetching {timespan} data in {len(date_ranges)-1} chunk(s)...")

    for i in range(len(date_ranges) - 1):
        chunk_from = date_ranges[i].strftime('%Y-%m-%d')
        chunk_to = date_ranges[i+1].strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{chunk_from}/{chunk_to}"
        params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}

        while True:
            try:
                r = session.get(url, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
            except Exception as e:
                print(f"    ❌ Error fetching chunk {chunk_from} to {chunk_to} for {ticker}: {e}")
                results = []
                break

            if not results:
                break
            
            all_data.extend(results)
            
            if "next_url" in data and data.get('results_count') == 50000:
                url = data["next_url"]
                params = {"apiKey": API_KEY}
                time.sleep(0.5)
            else:
                break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['t'])
    
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )[["timestamp", "open", "high", "low", "close", "volume"]].sort_values('timestamp')


def add_features(df, features, prefix):
    df = df.copy()
    cols_to_add = []
    for f in features:
        col_parts = [prefix]
        if f.get('source'):
            col_parts.append(f['source'])
        col_parts.append(f['field'])
        if f.get('window'):
            col_parts.append(str(f['window']))
        if f.get('normalize'):
            col_parts.append(f['method'])
        col = "_".join(col_parts)
        
        cols_to_add.append(col)
        
        source_col = f.get('source', f.get('field'))
        feature_series = pd.Series(index=df.index, dtype=float)

        if f["field"] == "ema":
            feature_series = df[source_col].ewm(span=f["window"], adjust=False).mean()
        elif f["field"] == "sma":
            feature_series = df[source_col].rolling(window=f["window"]).mean()
        elif f["field"] == "vwap":
            feature_series = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        elif f["type"] in ["price", "volume"]:
            feature_series = df[f["field"]]
        
        if f.get("normalize"):
            method = f["method"]
            if method == "zscore":
                df[col] = (feature_series - feature_series.mean()) / feature_series.std()
            elif method == "rolling_zscore":
                df[col] = (feature_series - feature_series.rolling(f["window"]).mean()) / feature_series.rolling(f["window"]).std()
        else:
            df[col] = feature_series

    unique_cols = list(dict.fromkeys(cols_to_add))
    return df[["timestamp"] + unique_cols]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg.get("training", {}).get("SEED", 42))
    dates = pd.date_range(start=start_date, end=end_date, freq="5min", tz="America/New_York")
    sim_cfg = cfg.get("simulated_data", {})
    prices = sim_cfg.get("base_price", 150) + np.cumsum(np.random.normal(0, sim_cfg.get("price_volatility", 0.5), len(dates)))
    volume = np.random.normal(sim_cfg.get("volume_mean", 500000), sim_cfg.get("volume_std", 100000), len(dates))
    return pd.DataFrame({"timestamp": dates, "open": prices, "high": prices + 0.5, "low": prices - 0.5, "close": prices, "volume": volume})


def merge_dataframes(base_df, higher_df):
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df["date"] = higher_df["timestamp"].dt.date
    merged = pd.merge(base_df, higher_df.drop(columns=["timestamp"]), on="date", how="left")
    return merged.drop(columns=["date"])


def save_dataframe(df, filename):
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(DATA_DIR, f"{filename}.{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        print(f"💾 Saved to {path}")


# --- Main Execution ---
for ticker in TICKERS:
    print(f"\n{'='*20}\nProcessing Ticker: {ticker}\n{'='*20}")

    if ticker.startswith("@SIM"):
        print("Recognized simulated ticker. Running simulation process...")
        try:
            for scenario in cfg.get("simulated_data", {}).get("scenarios", []):
                prefix = "5min"
                df = generate_simulated_data(scenario, START_DATE, END_DATE, cfg)
                df = add_features(df, cfg["features"]["FEATURES_5MIN"], prefix)
                save_dataframe(df, f"SIM_{scenario}_{prefix}")
                print(f"✅ Generated: SIM_{scenario}_{prefix}")
        except KeyError as e:
            print(f"❌ Error in simulation config: Missing key {e}. Skipping simulation.")
        continue

    all_configured_prefixes = [k.replace("FEATURES_", "").lower() for k in cfg.get("features", {}) if k.startswith("FEATURES_")]
    
    precompute_prefixes = []
    main_prefixes = []

    for prefix in all_configured_prefixes:
        if prefix not in TIMEFRAME_MAP:
            print(f"⏭️ Skipping unsupported timeframe '{prefix}' defined in config.")
            continue
        
        if TIMEFRAME_MAP[prefix].get("timespan") in ["day", "week", "month"]:
            precompute_prefixes.append(prefix)
        else:
            main_prefixes.append(prefix)
    
    print(f"Found {len(precompute_prefixes)} non-intraday timeframe(s) to pre-compute: {precompute_prefixes}")
    print(f"Found {len(main_prefixes)} intraday timeframe(s) for main processing: {main_prefixes}")

    precomputed_features_paths = {}
    
    if precompute_prefixes:
        print(f"\n--- PASS 1: Pre-computing features (OBS_WINDOW not applied) ---")
        for prefix in precompute_prefixes:
            feature_key = f"FEATURES_{prefix.upper()}"
            features = cfg["features"].get(feature_key)
            
            print(f"  ⚙️ Pre-computing for timeframe: {prefix}...")
            trading_days_buffer = compute_buffer_bars(features, apply_obs_window=False)
            calendar_days_back = convert_trading_days_to_calendar_days(trading_days_buffer)
            start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=calendar_days_back)).strftime("%Y-%m-%d")

            params = TIMEFRAME_MAP[prefix]
            raw_df = fetch_polygon_ohlcv(ticker, params["multiplier"], params["timespan"], start, END_DATE)

            if raw_df is not None and not raw_df.empty:
                feature_df = add_features(raw_df, features, prefix)
                filepath = os.path.join(DATA_DIR, f"{ticker}_{prefix}_precomputed.parquet")
                feature_df.to_parquet(filepath, index=False)
                precomputed_features_paths[prefix] = filepath
                print(f"  ✅ Saved pre-computed features to {filepath}")
            else:
                print(f"  ⚠️ No data for pre-computing {prefix}")

    print("\n--- PASS 2: Main processing (OBS_WINDOW applied) ---")
    
    main_features_config = {f"FEATURES_{p.upper()}": cfg["features"][f"FEATURES_{p.upper()}"] for p in main_prefixes}
    main_all_features = [feat for features in main_features_config.values() for feat in features]
    main_trading_days_buffer = compute_buffer_bars(main_all_features, apply_obs_window=True)
    main_calendar_days_back = convert_trading_days_to_calendar_days(main_trading_days_buffer)
    
    df_by_prefix = {}
    for prefix in main_prefixes:
        key = f"FEATURES_{prefix.upper()}"
        features = cfg["features"][key]
        
        print(f"  ⚙️ Processing timeframe: {prefix}...")
        start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=main_calendar_days_back)).strftime("%Y-%m-%d")
        
        params = TIMEFRAME_MAP[prefix]
        raw_df = fetch_polygon_ohlcv(ticker, params["multiplier"], params["timespan"], start, END_DATE)

        if raw_df is not None and not raw_df.empty:
            feature_df = add_features(raw_df.copy(), features, prefix)
            df_by_prefix[prefix] = pd.merge(raw_df, feature_df, on="timestamp", how="left")
        else:
            print(f"  ⚠️ No data returned for {ticker} timeframe {prefix}")

    if df_by_prefix:
        print("\n--- Final Merge ---")
        timeframe_order = {"1min": 0, "5min": 1, "15min": 2, "30min": 3, "hour": 4}
        base_prefix = min(main_prefixes, key=lambda x: timeframe_order.get(x, float("inf")))
        print(f"📉 Base timeframe determined as: {base_prefix}")

        merged = df_by_prefix[base_prefix]

        for prefix in main_prefixes:
            if prefix != base_prefix:
                print(f"  🖇️ Merging {prefix} features...")
                feature_cols = [c for c in df_by_prefix[prefix].columns if c.startswith(prefix)]
                
                left_df = merged.sort_values('timestamp')
                right_df = df_by_prefix[prefix][["timestamp"] + feature_cols].sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

                merged = pd.merge_asof(left_df, right_df, on='timestamp', direction='backward', tolerance=pd.Timedelta('15min'))
        
        for prefix, path in precomputed_features_paths.items():
            print(f"  🖇️ Merging pre-computed {prefix} features from {path}...")
            precomputed_df = pd.read_parquet(path)
            merged = merge_dataframes(merged, precomputed_df)

        start_ts = pd.to_datetime(START_DATE).tz_localize("America/New_York")
        final = merged[merged["timestamp"] >= start_ts].reset_index(drop=True)

        nan_summary = final.isna().sum()
        if nan_summary.any():
            print("\n🔴 Warning: NaNs detected in final dataset after trimming. Check feature definitions or data source.")
            print(nan_summary[nan_summary > 0])
        
        output_cols = cfg.get("data", {}).get("output_columns")
        if output_cols:
            print(f"🔬 Filtering final output to {len(output_cols)} specified column(s)...")
            if "timestamp" not in output_cols:
                output_cols.insert(0, "timestamp")
            
            existing_cols = [col for col in output_cols if col in final.columns]
            missing_cols = set(output_cols) - set(existing_cols)
            
            if missing_cols:
                print(f"  ⚠️ These requested output columns were not found and have been ignored: {list(missing_cols)}")
            
            final = final[existing_cols]

        output_filename = f"{ticker}_{base_prefix}"
        save_dataframe(final, output_filename)
        print(f"✅ Finished processing for {ticker}. Final file: {output_filename}")
    else:
        print(f"🤷 No main data processed for {ticker}. Skipping.")


⏭️ AFTER: Full Code Snapshot (v1.2.15)
"""
Data Manager: Dynamically fetches and processes financial data based on YAML configuration.
Implements:
- Automated Two-Pass Architecture for handling different timeframe requirements.
- Robust buffer logic and request chunking to prevent API errors.
- Optional filtering of final output columns based on config, with a cleaner default.
- Production-ready.
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

session = requests.Session()
TIMEFRAME_MAP = {
    "1min": {"multiplier": 1, "timespan": "minute", "bar_divisor": 390},
    "5min": {"multiplier": 5, "timespan": "minute", "bar_divisor": 78},
    "15min": {"multiplier": 15, "timespan": "minute", "bar_divisor": 26},
    "30min": {"multiplier": 30, "timespan": "minute", "bar_divisor": 13},
    "hour": {"multiplier": 60, "timespan": "minute", "bar_divisor": 6.5},
    "daily": {"multiplier": 1, "timespan": "day", "bar_divisor": 1},
    "weekly": {"multiplier": 1, "timespan": "week", "bar_divisor": 1/5},
    "monthly": {"multiplier": 1, "timespan": "month", "bar_divisor": 1/21},
}


def convert_trading_days_to_calendar_days(trading_days):
    return int(trading_days * 1.5)


def compute_buffer_bars(feature_list, apply_obs_window=True):
    if not feature_list:
        return 0
    max_window = max([f.get("window", 0) for f in feature_list], default=0)
    trading_days_needed = max_window + 20
    if apply_obs_window:
        trading_days_needed += cfg["features"].get("OBS_WINDOW", 50)
    return trading_days_needed


def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    all_data = []
    
    date_ranges = list(pd.date_range(start=from_date, end=to_date, freq='30D'))
    if pd.to_datetime(to_date).date() not in [d.date() for d in date_ranges]:
        date_ranges.append(pd.to_datetime(to_date))

    print(f"    Fetching {timespan} data in {len(date_ranges)-1} chunk(s)...")

    for i in range(len(date_ranges) - 1):
        chunk_from = date_ranges[i].strftime('%Y-%m-%d')
        chunk_to = date_ranges[i+1].strftime('%Y-%m-%d')
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{chunk_from}/{chunk_to}"
        params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}

        while True:
            try:
                r = session.get(url, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                results = data.get("results", [])
            except Exception as e:
                print(f"    ❌ Error fetching chunk {chunk_from} to {chunk_to} for {ticker}: {e}")
                results = []
                break

            if not results:
                break
            
            all_data.extend(results)
            
            if "next_url" in data and data.get('results_count') == 50000:
                url = data["next_url"]
                params = {"apiKey": API_KEY}
                time.sleep(0.5)
            else:
                break
    
    if not all_data:
        return None
    
    df = pd.DataFrame(all_data)
    df = df.drop_duplicates(subset=['t'])
    
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York")
    return df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )[["timestamp", "open", "high", "low", "close", "volume"]].sort_values('timestamp')


def add_features(df, features, prefix):
    df = df.copy()
    cols_to_add = []
    for f in features:
        col_parts = [prefix]
        if f.get('source'):
            col_parts.append(f['source'])
        col_parts.append(f['field'])
        if f.get('window'):
            col_parts.append(str(f['window']))
        if f.get('normalize'):
            col_parts.append(f['method'])
        col = "_".join(col_parts)
        
        cols_to_add.append(col)
        
        source_col = f.get('source', f.get('field'))
        feature_series = pd.Series(index=df.index, dtype=float)

        if f["field"] == "ema":
            feature_series = df[source_col].ewm(span=f["window"], adjust=False).mean()
        elif f["field"] == "sma":
            feature_series = df[source_col].rolling(window=f["window"]).mean()
        elif f["field"] == "vwap":
            feature_series = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        elif f["type"] in ["price", "volume"]:
            feature_series = df[f["field"]]
        
        if f.get("normalize"):
            method = f["method"]
            if method == "zscore":
                df[col] = (feature_series - feature_series.mean()) / feature_series.std()
            elif method == "rolling_zscore":
                df[col] = (feature_series - feature_series.rolling(f["window"]).mean()) / feature_series.rolling(f["window"]).std()
        else:
            df[col] = feature_series

    unique_cols = list(dict.fromkeys(cols_to_add))
    return df[["timestamp"] + unique_cols]


def generate_simulated_data(scenario, start_date, end_date, cfg):
    np.random.seed(cfg.get("training", {}).get("SEED", 42))
    dates = pd.date_range(start=start_date, end=end_date, freq="5min", tz="America/New_York")
    sim_cfg = cfg.get("simulated_data", {})
    prices = sim_cfg.get("base_price", 150) + np.cumsum(np.random.normal(0, sim_cfg.get("price_volatility", 0.5), len(dates)))
    volume = np.random.normal(sim_cfg.get("volume_mean", 500000), sim_cfg.get("volume_std", 100000), len(dates))
    return pd.DataFrame({"timestamp": dates, "open": prices, "high": prices + 0.5, "low": prices - 0.5, "close": prices, "volume": volume})


def merge_dataframes(base_df, higher_df):
    base_df["date"] = base_df["timestamp"].dt.date
    higher_df["date"] = higher_df["timestamp"].dt.date
    merged = pd.merge(base_df, higher_df.drop(columns=["timestamp"]), on="date", how="left")
    return merged.drop(columns=["date"])


def save_dataframe(df, filename):
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(DATA_DIR, f"{filename}.{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        print(f"💾 Saved to {path}")


# --- Main Execution ---
for ticker in TICKERS:
    print(f"\n{'='*20}\nProcessing Ticker: {ticker}\n{'='*20}")

    if ticker.startswith("@SIM"):
        print("Recognized simulated ticker. Running simulation process...")
        try:
            for scenario in cfg.get("simulated_data", {}).get("scenarios", []):
                prefix = "5min"
                df = generate_simulated_data(scenario, START_DATE, END_DATE, cfg)
                df = add_features(df, cfg["features"]["FEATURES_5MIN"], prefix)
                save_dataframe(df, f"SIM_{scenario}_{prefix}")
                print(f"✅ Generated: SIM_{scenario}_{prefix}")
        except KeyError as e:
            print(f"❌ Error in simulation config: Missing key {e}. Skipping simulation.")
        continue

    all_configured_prefixes = [k.replace("FEATURES_", "").lower() for k in cfg.get("features", {}) if k.startswith("FEATURES_")]
    
    precompute_prefixes = []
    main_prefixes = []

    for prefix in all_configured_prefixes:
        if prefix not in TIMEFRAME_MAP:
            print(f"⏭️ Skipping unsupported timeframe '{prefix}' defined in config.")
            continue
        
        if TIMEFRAME_MAP[prefix].get("timespan") in ["day", "week", "month"]:
            precompute_prefixes.append(prefix)
        else:
            main_prefixes.append(prefix)
    
    print(f"Found {len(precompute_prefixes)} non-intraday timeframe(s) to pre-compute: {precompute_prefixes}")
    print(f"Found {len(main_prefixes)} intraday timeframe(s) for main processing: {main_prefixes}")

    precomputed_features_paths = {}
    
    if precompute_prefixes:
        print(f"\n--- PASS 1: Pre-computing features (OBS_WINDOW not applied) ---")
        for prefix in precompute_prefixes:
            feature_key = f"FEATURES_{prefix.upper()}"
            features = cfg["features"].get(feature_key)
            
            print(f"  ⚙️ Pre-computing for timeframe: {prefix}...")
            trading_days_buffer = compute_buffer_bars(features, apply_obs_window=False)
            calendar_days_back = convert_trading_days_to_calendar_days(trading_days_buffer)
            start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=calendar_days_back)).strftime("%Y-%m-%d")

            params = TIMEFRAME_MAP[prefix]
            raw_df = fetch_polygon_ohlcv(ticker, params["multiplier"], params["timespan"], start, END_DATE)

            if raw_df is not None and not raw_df.empty:
                feature_df = add_features(raw_df, features, prefix)
                filepath = os.path.join(DATA_DIR, f"{ticker}_{prefix}_precomputed.parquet")
                feature_df.to_parquet(filepath, index=False)
                precomputed_features_paths[prefix] = filepath
                print(f"  ✅ Saved pre-computed features to {filepath}")
            else:
                print(f"  ⚠️ No data for pre-computing {prefix}")

    print("\n--- PASS 2: Main processing (OBS_WINDOW applied) ---")
    
    main_features_config = {f"FEATURES_{p.upper()}": cfg["features"][f"FEATURES_{p.upper()}"] for p in main_prefixes}
    main_all_features = [feat for features in main_features_config.values() for feat in features]
    main_trading_days_buffer = compute_buffer_bars(main_all_features, apply_obs_window=True)
    main_calendar_days_back = convert_trading_days_to_calendar_days(main_trading_days_buffer)
    
    df_by_prefix = {}
    for prefix in main_prefixes:
        key = f"FEATURES_{prefix.upper()}"
        features = cfg["features"][key]
        
        print(f"  ⚙️ Processing timeframe: {prefix}...")
        start = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=main_calendar_days_back)).strftime("%Y-%m-%d")
        
        params = TIMEFRAME_MAP[prefix]
        raw_df = fetch_polygon_ohlcv(ticker, params["multiplier"], params["timespan"], start, END_DATE)

        if raw_df is not None and not raw_df.empty:
            feature_df = add_features(raw_df.copy(), features, prefix)
            df_by_prefix[prefix] = pd.merge(raw_df, feature_df, on="timestamp", how="left")
        else:
            print(f"  ⚠️ No data returned for {ticker} timeframe {prefix}")

    if df_by_prefix:
        print("\n--- Final Merge ---")
        timeframe_order = {"1min": 0, "5min": 1, "15min": 2, "30min": 3, "hour": 4}
        base_prefix = min(main_prefixes, key=lambda x: timeframe_order.get(x, float("inf")))
        print(f"📉 Base timeframe determined as: {base_prefix}")

        merged = df_by_prefix[base_prefix]

        for prefix in main_prefixes:
            if prefix != base_prefix:
                print(f"  🖇️ Merging {prefix} features...")
                feature_cols = [c for c in df_by_prefix[prefix].columns if c.startswith(prefix)]
                
                left_df = merged.sort_values('timestamp')
                right_df = df_by_prefix[prefix][["timestamp"] + feature_cols].sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')

                merged = pd.merge_asof(left_df, right_df, on='timestamp', direction='backward', tolerance=pd.Timedelta('15min'))
        
        for prefix, path in precomputed_features_paths.items():
            print(f"  🖇️ Merging pre-computed {prefix} features from {path}...")
            precomputed_df = pd.read_parquet(path)
            merged = merge_dataframes(merged, precomputed_df)

        start_ts = pd.to_datetime(START_DATE).tz_localize("America/New_York")
        final = merged[merged["timestamp"] >= start_ts].reset_index(drop=True)

        nan_summary = final.isna().sum()
        if nan_summary.any():
            print("\n🔴 Warning: NaNs detected in final dataset after trimming. Check feature definitions or data source.")
            print(nan_summary[nan_summary > 0])
        
        # --- NEW: Cleaner Default Output Logic ---
        output_cols = cfg.get("data", {}).get("output_columns")
        
        if output_cols:
            print(f"🔬 Filtering final output to user-specified columns: {output_cols}")
            # Ensure timestamp is always included
            if "timestamp" not in output_cols:
                output_cols.insert(0, "timestamp")
            
            # Filter for existing columns to prevent errors and warn about missing ones
            existing_cols = [col for col in output_cols if col in final.columns]
            missing_cols = set(output_cols) - set(existing_cols)
            
            if missing_cols:
                print(f"  ⚠️ These requested output columns were not found and have been ignored: {list(missing_cols)}")
            
            final_to_save = final[existing_cols]
        else:
            # Default to saving only timestamp + generated features
            print("🔬 No output_columns specified. Defaulting to timestamp + generated features.")
            base_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in final.columns if col not in base_cols and col != 'timestamp']
            final_to_save = final[['timestamp'] + feature_cols]

        output_filename = f"{ticker}_{base_prefix}"
        save_dataframe(final_to_save, output_filename)
        print(f"✅ Finished processing for {ticker}. Final file: {output_filename}")
    else:
        print(f"🤷 No main data processed for {ticker}. Skipping.")

