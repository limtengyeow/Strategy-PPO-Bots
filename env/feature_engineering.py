
import pandas as pd

def init_feature_space(df, features_cfg, debug=False):
    features = features_cfg.get("FEATURES", [])
    selected_cols = []

    for feat in features:
        if feat.get("type") == "price":
            field = feat.get("field")
            if field in df.columns:
                series = df[field]
                if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
                    selected_cols.append(field)
                elif debug:
                    print(f"[WARN] Skipping {field}: not numeric or all NaN")
            elif debug:
                print(f"[WARN] Skipping {field}: not found in DataFrame")

    if not selected_cols:
        raise ValueError("No valid numeric features found in config.")

    return df, selected_cols
