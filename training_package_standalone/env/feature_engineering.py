import pandas as pd
import numpy as np

def init_feature_space(df, features_cfg, debug=False):
    features = features_cfg.get("FEATURES", [])
    selected_cols = []

    for feat in features:
        if feat.get("type") == "price":
            field = feat.get("field")
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")
            if field in df.columns:
                series = df[field]
                if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
                    if normalize:
                        if method == "zscore":
                            mean = series.mean()
                            std = series.std()
                            df[field] = (series - mean) / (std + 1e-8)
                            if debug:
                                print(f"[NORMALIZE:zscore] {field} mean={mean:.4f}, std={std:.4f}")

                        elif method == "rolling_zscore":
                            window = feat.get("window", 20)
                            roll_mean = series.rolling(window=window, min_periods=1).mean()
                            roll_std = series.rolling(window=window, min_periods=1).std()
                            zscore = (series - roll_mean) / (roll_std + 1e-8)

                            # Fill first window-1 values with 0 (acceptable)
                            zscore.iloc[:window-1] = 0

                            # After that, raise error if NaNs exist
                            if zscore.iloc[window-1:].isnull().any():
                                raise ValueError(f"[ERROR] {field} has NaNs after initial padding. Check your data or method.")

                            df[field] = zscore

                            #  Debug print
                            if debug:
                                print(f"[NORMALIZE:rolling_zscore] {field} with window={window}")
                                print(f"[DEBUG:rolling_zscore] {field} - first 30 rows:")
                                print(zscore.head(30).to_string(index=True))
                                print(f"[DEBUG:rolling_std] {field} - first 30 rows:")
                                print(roll_std.head(30).to_string(index=True))

                        elif method == "minmax":
                            min_val = series.min()
                            max_val = series.max()
                            df[field] = (series - min_val) / (max_val - min_val + 1e-8)
                            if debug:
                                print(f"[NORMALIZE:minmax] {field} min={min_val:.4f}, max={max_val:.4f}")
                        elif method == "log_return":
                            df[field] = np.log(series / series.shift(1)).fillna(0)
                            if debug:
                                print(f"[NORMALIZE:log_return] {field}")
                        elif method == "percent_change":
                            df[field] = series.pct_change().fillna(0)
                            if debug:
                                print(f"[NORMALIZE:percent_change] {field}")
                        else:
                            raise ValueError(f"Unsupported normalization method: {method}")
                    selected_cols.append(field)
                elif debug:
                    print(f"[WARN] Skipping {field}: not numeric or all NaN")
            elif debug:
                print(f"[WARN] Skipping {field}: not found in DataFrame")

        elif feat.get("type") == "indicator":
            field = feat.get("field")         # "vwap", "sma", "ema"
            source = feat.get("source", "close")
            window = feat.get("window", 14)
            normalize = feat.get("normalize", False)
            method = feat.get("method", "zscore")

            if field == "vwap":
                if "vwap" not in df.columns:
                    tp = (df["high"] + df["low"] + df["close"]) / 3
                    df["vwap"] = (tp * df["volume"]).cumsum() / df["volume"].cumsum()
                name = "vwap"

            elif field == "sma":
                name = f"sma_{window}_{source}"
                df[name] = df[source].rolling(window).mean()

            elif field == "ema":
                name = f"ema_{window}_{source}"
                df[name] = df[source].ewm(span=window, adjust=False).mean()

            else:
                raise ValueError(f"Unsupported indicator: {field}")

            # Optional normalization for indicators
            if normalize:
                # Fill NaNs before normalization to prevent downstream errors
                series = df[name].fillna(method="ffill").fillna(0)

                if method == "zscore":
                    mean = series.mean()
                    std = series.std()
                    df[name] = (series - mean) / (std + 1e-8)

                elif method == "rolling_zscore":
                    roll_mean = series.rolling(window=window, min_periods=1).mean()
                    roll_std = series.rolling(window=window, min_periods=1).std()
                    zscore = (series - roll_mean) / (roll_std + 1e-8)
                    zscore.iloc[:window - 1] = 0

                    if zscore.iloc[window - 1:].isnull().any():
                        raise ValueError(f"[ERROR] {name} has NaNs after padding.")
                    df[name] = zscore

                elif method == "minmax":
                    min_val = series.min()
                    max_val = series.max()
                    df[name] = (series - min_val) / (max_val - min_val + 1e-8)

                else:
                    raise ValueError(f"Unsupported normalization method: {method}")


            selected_cols.append(name)

            if debug:
                print(f"[INDICATOR] {name} from {source}, normalized={normalize} method={method}")

    if not selected_cols:
        raise ValueError("No valid numeric features found in config.")

    return df, selected_cols
