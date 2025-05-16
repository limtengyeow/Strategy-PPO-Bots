import os
import sys
import json
import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from env.trading_env import TradingEnv
from callbacks.performance_metrics_callback import PerformanceMetricsCallback

# === Load config.json ===
def load_config(config_path="config.json", test_mode=False):
    with open(config_path) as f:
        cfg = json.load(f)

    if test_mode:
        cfg["training"]["TOTAL_TIMESTEPS"] = 2048
        cfg["training"]["EVAL_FREQ"] = 0
        cfg["training"]["SAVE_FREQ"] = 2048
        cfg["logging"]["use_mlflow"] = False

    return cfg

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run in test mode")
args, _ = parser.parse_known_args()
cfg = load_config(test_mode=args.test)

# === Parameters ===
data_dir = cfg["env"]["DATA_FOLDER"]
total_timesteps = cfg["training"]["TOTAL_TIMESTEPS"]
ppo_params = cfg["training"]["PPO_PARAMS"]

# Handle activation_fn string
activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}
if "policy_kwargs" in ppo_params:
    act_str = ppo_params["policy_kwargs"].get("activation_fn", "relu")
    ppo_params["policy_kwargs"]["activation_fn"] = activation_map.get(act_str, nn.ReLU)

# === Helper: Get all ticker-timeframe files ===
ticker_files = {}
for f in os.listdir(data_dir):
    if f.endswith(".csv") and "_" in f:
        ticker, rest = f.split("_", 1)
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(f)

# === Timeframe sort order (lowest to highest granularity) ===
timeframe_order = ["1minute", "5minute", "15minute", "1hour", "1day", "1week"]

# === Loop over tickers with lowest timeframe ===
for ticker, files in ticker_files.items():
    files_sorted = sorted(files, key=lambda x: timeframe_order.index(x.split("_")[1].replace(".csv", "")) if x.split("_")[1].replace(".csv", "") in timeframe_order else len(timeframe_order))
    selected_file = files_sorted[0]
    file_path = os.path.join(data_dir, selected_file)

    if not os.path.exists(file_path):
        print(f"[Skip] File not found for {ticker}: {file_path}")
        continue

    print(f"[Train] Using {selected_file} for {ticker}")
    df = pd.read_csv(file_path)

    # ✅ DEBUG BLOCK: Check config and CSV contents
    print("\n=== DEBUG START ===")
    print("[DEBUG] CONFIG keys:", cfg.keys())
    print("[DEBUG] FEATURES in config:", cfg.get("training", {}).get("FEATURES"))
    print("[DEBUG] DataFrame columns:", df.columns.tolist())
    print("[DEBUG] DataFrame dtypes:\n", df.dtypes)
    print("[DEBUG] First few rows:\n", df.head())
    print("=== DEBUG END ===\n")

    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df=df, config=cfg)])

    # Create model
    model = PPO("MlpPolicy", env, **ppo_params, verbose=1)

    # Setup callbacks
    callback = CallbackList([
        PerformanceMetricsCallback(config_path="config.json", verbose=1)
    ])

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    save_path = os.path.join(cfg["env"]["MODEL_DIR"], f"ppo_{ticker}")
    model.save(save_path)
    print(f"[Done] Model saved for {ticker} at {save_path}")
