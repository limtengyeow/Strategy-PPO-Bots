from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Set up the log file path relative to the current working directory
LOG_FILE = Path.cwd() / "logs" / "feature_debug.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_feature_debug(message, debug):
    if debug:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a") as f:
            f.write(f"[{timestamp}] {message}\n")


def init_feature_space(df, features_cfg, debug=False):
    features = features_cfg.get("FEATURES", [])
    daily_features = features_cfg.get("DAILY_FEATURES", [])
    selected_cols = []

    log_feature_debug("=== Feature Engineering Started ===", debug)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    raw_price_fields = [
        feat.get("field") for feat in features if feat.get("type") == "price"
    ]
    price_cols_backup = (
        df[raw_price_fields].copy() if raw_price_fields else pd.DataFrame()
    )

    for feat in features:
        field = feat.get("field")
        if feat.get("type") == "price":
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")

            if field in df.columns:
                series = df[field]
                if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
                    if normalize:
                        try:
                            if method == "zscore":
                                df[field] = (series - series.mean()) / (
                                    series.std() + 1e-8
                                )
                            elif method == "rolling_zscore":
                                window = feat.get("window", 20)
                                roll_mean = series.rolling(window, min_periods=1).mean()
                                roll_std = series.rolling(window, min_periods=1).std()
                                df[field] = (series - roll_mean) / (roll_std + 1e-8)
                            elif method == "minmax":
                                df[field] = (series - series.min()) / (
                                    series.max() - series.min() + 1e-8
                                )
                            elif method == "log_return":
                                df[field] = np.log(series / series.shift(1)).fillna(0)
                            elif method == "percent_change":
                                df[field] = series.pct_change().fillna(0)
                            else:
                                raise ValueError(
                                    f"Unsupported normalization method: {method}"
                                )
                        except Exception as e:
                            log_feature_debug(
                                f"[ERROR] {field} normalization failed: {e}", debug
                            )
                    selected_cols.append(field)

        elif feat.get("type") == "indicator":
            source = feat.get("source", "close")
            window = feat.get("window", 14)
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")

            try:
                if field == "vwap":
                    if "vwap" not in df.columns:
                        tp = (df["high"] + df["low"] + df["close"]) / 3
                        df["vwap"] = (tp * df["volume"]).cumsum() / df[
                            "volume"
                        ].cumsum()
                    name = "vwap"
                elif field == "ema":
                    name = f"ema_{window}_{source}"
                    df[name] = df[source].ewm(span=window, adjust=False).mean()
                elif field == "rsi":
                    delta = df[source].diff()
                    gain = delta.where(delta > 0, 0).rolling(window).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window).mean()
                    rs = gain / (loss + 1e-8)
                    df[name := f"rsi_{window}_{source}"] = 100 - (100 / (1 + rs))
                elif field == "atr":
                    tr = pd.concat(
                        [
                            df["high"] - df["low"],
                            abs(df["high"] - df["close"].shift()),
                            abs(df["low"] - df["close"].shift()),
                        ],
                        axis=1,
                    ).max(axis=1)
                    df[name := f"atr_{window}"] = tr.rolling(window).mean()
                else:
                    raise ValueError(f"Unsupported indicator: {field}")

                if normalize:
                    series = df[name].ffill().fillna(0)
                    if method == "zscore":
                        df[name] = (series - series.mean()) / (series.std() + 1e-8)
                    elif method == "rolling_zscore":
                        roll_mean = series.rolling(window, min_periods=1).mean()
                        roll_std = series.rolling(window, min_periods=1).std()
                        df[name] = (series - roll_mean) / (roll_std + 1e-8)
                    elif method == "minmax":
                        df[name] = (series - series.min()) / (
                            series.max() - series.min() + 1e-8
                        )
                    else:
                        raise ValueError(f"Unsupported normalization method: {method}")

                selected_cols.append(name)

            except Exception as e:
                log_feature_debug(
                    f"[ERROR] {field} feature creation failed: {e}", debug
                )

    df_daily = (
        df.resample("1D", on="timestamp")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    for dfeat in daily_features:
        field = dfeat.get("field")
        source = dfeat.get("source", "close")
        window = dfeat.get("window", 14)

        try:
            if field == "ema":
                df_daily[f"daily_ema_{window}"] = (
                    df_daily[source].ewm(span=window, adjust=False).mean().ffill()
                )
            elif field == "sma":
                df_daily[f"daily_sma_{window}"] = (
                    df_daily[source].rolling(window).mean().ffill()
                )
            else:
                raise ValueError(f"Unsupported daily indicator: {field}")
            log_feature_debug(f"[DAILY] Added {field}({window}) on {source}", debug)
        except Exception as e:
            log_feature_debug(
                f"[ERROR] {field} daily feature creation failed: {e}", debug
            )

    df = pd.merge_asof(
        df.sort_values("timestamp"),
        df_daily.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    for field in raw_price_fields:
        if field not in df.columns:
            df[field] = price_cols_backup[field]
            log_feature_debug(
                f"[FIX] Restored raw price field '{field}' after merge", debug
            )
        df[field] = df[field].ffill().bfill().fillna(0)
        log_feature_debug(
            f"[FIX] Filled NaNs in raw price field '{field}' after restore", debug
        )

    df = df.ffill().bfill().fillna(0)
    log_feature_debug("[FINAL] Filled any remaining NaNs globally in df", debug)

    selected_cols += [col for col in df.columns if col.startswith("daily_")]

    for field in raw_price_fields:
        if field not in selected_cols and field in df.columns:
            selected_cols.append(field)
            log_feature_debug(
                f"[FIX] Added raw price field '{field}' to selected_cols", debug
            )

    if not selected_cols:
        log_feature_debug("[ERROR] No valid numeric features found in config.", debug)
        raise ValueError("No valid numeric features found in config.")

    log_feature_debug(f"Features generated: {selected_cols}", debug)
    log_feature_debug("=== Feature Engineering Completed ===\n", debug)

    return df, selected_cols
