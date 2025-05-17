#
import os
import json
import requests
import pandas as pd
import time
import argparse
from datetime import datetime
from pathlib import Path

# === Load config.json ===
with open("config.json") as f:
    cfg = json.load(f)

API_KEY = cfg["env"]["POLYGON_API_KEY"]
DEFAULT_DATA_DIR = cfg["env"].get("DATA_FOLDER", "data")
Path(DEFAULT_DATA_DIR).mkdir(exist_ok=True)

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"

# Determine additional coarser intervals based on requested interval
COARSER = [
    (1, 'day'),
    (1, 'week')
]

def fetch_polygon_ohlcv(ticker, multiplier, timespan, from_date, to_date, save_path):
    url = BASE_URL.format(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_date=from_date,
        to_date=to_date
    )
    params = {
        "adjusted": "true",
        "limit": 50000,
        "apiKey": API_KEY
    }

    all_data = []
    print(f"Fetching {ticker} from {from_date} to {to_date} ({multiplier} {timespan})...")
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Failed for {ticker} ({multiplier} {timespan}): {response.text}")
            break

        data = response.json().get("results", [])
        if not data:
            break

        all_data.extend(data)
        if len(data) < params['limit']:
            break

        last_ts = data[-1]['t']
        next_from = datetime.utcfromtimestamp(last_ts/1000.0).strftime('%Y-%m-%d')
        params['from'] = next_from
        time.sleep(1)

    df = pd.DataFrame(all_data)
    if df.empty:
        print(f"No data for {ticker} ({multiplier} {timespan}).")
        return

    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
    df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"}, inplace=True)
    df.set_index('timestamp', inplace=True)
    df = df[['open','high','low','close','volume']]

    df.to_csv(save_path)
    print(f"Saved {ticker}_{multiplier}{timespan} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download OHLCV data from Polygon.io at multiple intervals")
    parser.add_argument("--ticker", help="Single stock ticker (e.g., NVDA)")
    parser.add_argument("--tickers-file", help="Path to file with one ticker per line")
    parser.add_argument("--interval", required=True, help="Interval (e.g., 5min, 1h, daily, weekly)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    args = parser.parse_args()

    interval = args.interval.lower()
    # Parse requested interval
    if interval.endswith("min"):
        mult = int(interval.replace("min", ""))
        span = 'minute'
    elif interval.endswith("h") or interval.endswith("hour"):
        num = interval.replace("h", "").replace("hour", "")
        mult = int(num)
        span = 'hour'
    elif interval in ['day','daily']:
        mult = 1
        span = 'day'
    elif interval in ['week','weekly']:
        mult = 1
        span = 'week'
    else:
        raise ValueError(f"Unsupported interval: {interval}")

    # Build list of intervals to fetch
    to_fetch = []
    # always include requested
    to_fetch.append((mult, span))
    # for any intraday or hourly, also fetch daily and weekly
    if span in ['minute', 'hour']:
        to_fetch.extend(COARSER)
    # if daily, add weekly
    elif span == 'day':
        to_fetch.append((1, 'week'))
    # if weekly, nothing extra

    # Fetch for each ticker and each interval
    tickers = []
    if args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [l.strip() for l in f if l.strip()]
    elif args.ticker:
        tickers = [args.ticker]
    else:
        parser.error('Provide --ticker or --tickers-file')

    for ticker in tickers:
        for m, s in to_fetch:
            out_name = f"{ticker}_{m}{s}.csv"
            out_path = os.path.join(DEFAULT_DATA_DIR, out_name)
            fetch_polygon_ohlcv(ticker, m, s, args.start, args.end, out_path)# Placeholder
