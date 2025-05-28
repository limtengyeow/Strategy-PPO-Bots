from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Set up the log file path relative to the current working directory (works in both Jupyter and terminal)
LOG_FILE = Path.cwd() / "logs" / "feature_debug.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_feature_debug(message, debug):
    """
    Helper function to write debug messages to the log file if debug mode is enabled.
    """
    if debug:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with LOG_FILE.open("a") as f:
            f.write(f"[{timestamp}] {message}\n")


def init_feature_space(df, features_cfg, debug=False):
    """
    Processes the input DataFrame to generate features for the trading model.
    Supports price features and technical indicators like SMA, EMA, RSI, ATR, and VWAP.

    Args:
        df (pd.DataFrame): Input DataFrame with raw OHLCV data.
        features_cfg (dict): Configuration dictionary with feature specifications (from config.json).
        debug (bool): If True, logs detailed feature creation steps to logs/feature_debug.log.

    Returns:
        pd.DataFrame: The DataFrame with new features added.
        list: List of feature column names used in the model.
    """
    features = features_cfg.get("FEATURES", [])
    selected_cols = []  # List of feature column names for model input

    log_feature_debug("=== Feature Engineering Started ===", debug)

    # Process each feature configuration
    for feat in features:
        if feat.get("type") == "price":
            # Handle raw price data (e.g., close, volume)
            field = feat.get("field")
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")

            if field in df.columns:
                series = df[field]
                if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
                    if normalize:
                        try:
                            if method == "zscore":
                                mean = series.mean()
                                std = series.std()
                                df[field] = (series - mean) / (std + 1e-8)
                                log_feature_debug(
                                    f"[NORMALIZE:zscore] {field} mean={mean:.4f}, std={std:.4f}",
                                    debug,
                                )
                            elif method == "rolling_zscore":
                                window = feat.get("window", 20)
                                roll_mean = series.rolling(
                                    window=window, min_periods=1
                                ).mean()
                                roll_std = series.rolling(
                                    window=window, min_periods=1
                                ).std()
                                zscore = (series - roll_mean) / (roll_std + 1e-8)
                                zscore.iloc[: window - 1] = 0
                                if zscore.iloc[window - 1 :].isnull().any():
                                    raise ValueError(
                                        f"{field} has NaNs after initial padding."
                                    )
                                df[field] = zscore
                                log_feature_debug(
                                    f"[NORMALIZE:rolling_zscore] {field} window={window}",
                                    debug,
                                )
                            elif method == "minmax":
                                min_val = series.min()
                                max_val = series.max()
                                df[field] = (series - min_val) / (
                                    max_val - min_val + 1e-8
                                )
                                log_feature_debug(
                                    f"[NORMALIZE:minmax] {field} min={min_val:.4f}, max={max_val:.4f}",
                                    debug,
                                )
                            elif method == "log_return":
                                df[field] = np.log(series / series.shift(1)).fillna(0)
                                log_feature_debug(
                                    f"[NORMALIZE:log_return] {field}", debug
                                )
                            elif method == "percent_change":
                                df[field] = series.pct_change().fillna(0)
                                log_feature_debug(
                                    f"[NORMALIZE:percent_change] {field}", debug
                                )
                            else:
                                raise ValueError(
                                    f"Unsupported normalization method: {method}"
                                )
                        except Exception as e:
                            log_feature_debug(
                                f"[ERROR] {field} normalization failed: {e}", debug
                            )
                    selected_cols.append(field)
                else:
                    log_feature_debug(
                        f"[WARN] Skipping {field}: not numeric or all NaN", debug
                    )
            else:
                log_feature_debug(
                    f"[WARN] Skipping {field}: not found in DataFrame", debug
                )

        elif feat.get("type") == "indicator":
            # Handle technical indicators like SMA, EMA, RSI, ATR, VWAP
            field = feat.get("field")
            source = feat.get("source", "close")
            window = feat.get("window", 14)
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")

            try:
                # Compute the requested indicator
                if field == "vwap":
                    if "vwap" not in df.columns:
                        tp = (df["high"] + df["low"] + df["close"]) / 3
                        df["vwap"] = (tp * df["volume"]).cumsum() / df[
                            "volume"
                        ].cumsum()
                    name = "vwap"
                elif field == "sma":
                    name = f"sma_{window}_{source}"
                    df[name] = df[source].rolling(window).mean()
                elif field == "ema":
                    name = f"ema_{window}_{source}"
                    df[name] = df[source].ewm(span=window, adjust=False).mean()
                elif field == "rsi":
                    delta = df[source].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                    rs = gain / (loss + 1e-8)
                    df[f"rsi_{window}_{source}"] = 100 - (100 / (1 + rs))
                    name = f"rsi_{window}_{source}"
                elif field == "atr":
                    high_low = df["high"] - df["low"]
                    high_close = np.abs(df["high"] - df["close"].shift(1))
                    low_close = np.abs(df["low"] - df["close"].shift(1))
                    tr = (
                        high_low.to_frame("hl")
                        .join(high_close.to_frame("hc"))
                        .join(low_close.to_frame("lc"))
                        .max(axis=1)
                    )
                    df[f"atr_{window}"] = tr.rolling(window).mean()
                    name = f"atr_{window}"
                else:
                    raise ValueError(f"Unsupported indicator: {field}")

                # Apply normalization to indicator if configured
                if normalize:
                    series = df[name].fillna(method="ffill").fillna(0)
                    if method == "zscore":
                        mean = series.mean()
                        std = series.std()
                        df[name] = (series - mean) / (std + 1e-8)
                    elif method == "rolling_zscore":
                        roll_mean = series.rolling(window=window, min_periods=1).mean()
                        roll_std = series.rolling(window=window, min_periods=1).std()
                        zscore = (series - roll_mean) / (roll_std + 1e-8)
                        zscore.iloc[: window - 1] = 0
                        if zscore.iloc[window - 1 :].isnull().any():
                            raise ValueError(f"{name} has NaNs after padding.")
                        df[name] = zscore
                    elif method == "minmax":
                        min_val = series.min()
                        max_val = series.max()
                        df[name] = (series - min_val) / (max_val - min_val + 1e-8)
                    else:
                        raise ValueError(f"Unsupported normalization method: {method}")

                selected_cols.append(name)
                log_feature_debug(
                    f"[INDICATOR] {name} from {source}, normalized={normalize} method={method}",
                    debug,
                )

            except Exception as e:
                log_feature_debug(
                    f"[ERROR] {field} feature creation failed: {e}", debug
                )

    # Final check: make sure we have features generated
    if not selected_cols:
        log_feature_debug("[ERROR] No valid numeric features found in config.", debug)
        raise ValueError("No valid numeric features found in config.")

    log_feature_debug(f"Features generated: {selected_cols}", debug)
    log_feature_debug("=== Feature Engineering Completed ===\n", debug)

    return df, selected_cols
