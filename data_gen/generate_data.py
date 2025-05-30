import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# === Load config.json ===
with open("config.json") as f:
    cfg = json.load(f)

API_KEY = cfg["env"]["POLYGON_API_KEY"]
DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DATA_DIR).mkdir(exist_ok=True)

TICKERS = cfg["data"]["tickers"]
INTERVAL = cfg["data"]["default_interval"]
START_DATE = cfg["data"]["start_date"]
END_DATE = cfg["data"]["end_date"]
OUTPUT_FORMATS = cfg["data"].get("output_formats", ["csv"])

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
COARSER = [(1, "day"), (1, "week")]


def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date,
    )
    params = {"adjusted": "true", "limit": 50000, "apiKey": API_KEY}
    all_data = []

    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed for {ticker} ({multiplier} {timespan}): {response.text}")
            break
        data = response.json().get("results", [])
        if not data:
            break
        all_data.extend(data)
        if len(data) < params["limit"]:
            break
        last_ts = data[-1]["t"]
        next_from = datetime.utcfromtimestamp(last_ts / 1000.0).strftime("%Y-%m-%d")
        params["from"] = next_from
        time.sleep(1)

    df = pd.DataFrame(all_data)
    if df.empty:
        print(f"No data for {ticker} ({multiplier} {timespan}).")
        return None
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms")
    df.rename(
        columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"},
        inplace=True,
    )
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def save_data(df, ticker, interval):
    base_name = f"{ticker}_{interval}"
    for fmt in OUTPUT_FORMATS:
        path = os.path.join(DATA_DIR, f"{base_name}.{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
            print(f"Saved CSV: {path}")
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
            print(f"Saved Parquet: {path}")
        else:
            print(f"Unsupported format: {fmt}")


if __name__ == "__main__":
    # Parse interval
    interval = INTERVAL.lower()
    if interval.endswith("min"):
        mult = int(interval.replace("min", ""))
        span = "minute"
    elif interval.endswith("h") or interval.endswith("hour"):
        mult = int(interval.replace("h", "").replace("hour", ""))
        span = "hour"
    elif interval in ["day", "daily"]:
        mult = 1
        span = "day"
    elif interval in ["week", "weekly"]:
        mult = 1
        span = "week"
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    # Determine intervals to fetch
    to_fetch = [(mult, span)]
    if span in ["minute", "hour"]:
        to_fetch.extend(COARSER)
    elif span == "day":
        to_fetch.append((1, "week"))

    # Fetch and save for each ticker
    for ticker in TICKERS:
        for m, s in to_fetch:
            df = fetch_polygon_ohlcv(ticker, m, s, START_DATE, END_DATE)
            if df is not None:
                save_data(df, ticker, f"{m}{s}")
