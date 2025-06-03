import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks.performance_metrics_callback import PerformanceMetricsCallback
from env.trading_env import TradingEnv

# === Setup logging ===
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "train_debug.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# === Load config.json ===
def load_config(config_path="config.json"):
    with open(config_path) as f:
        return json.load(f)


cfg = load_config()

# === Set seed for reproducibility ===
assert "SEED" in cfg["training"], (
    "[Config] 'SEED' must be specified in training config."
)
SEED = cfg["training"]["SEED"]
print(f"[Seed] Using seed: {SEED}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Parameters ===
data_dir = cfg["env"]["DATA_FOLDER"]
total_timesteps = cfg["training"]["TOTAL_TIMESTEPS"]
ppo_params = cfg["training"]["PPO_PARAMS"]

# Handle activation_fn string
activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
if "policy_kwargs" in ppo_params:
    act_str = ppo_params["policy_kwargs"].get("activation_fn", "relu")
    ppo_params["policy_kwargs"]["activation_fn"] = activation_map.get(
        act_str.lower(), nn.ReLU
    )

# === Helper: Get all ticker files ===
ticker_files = {}
for f in os.listdir(data_dir):
    if f.endswith(".parquet") and "_" in f:
        ticker = f.split("_")[0]
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(f)

# === Training Loop ===
for ticker, files in ticker_files.items():
    selected_file = files[0]  # Assumes only 1 timeframe per ticker (5min)
    file_path = os.path.join(data_dir, selected_file)

    if not os.path.exists(file_path):
        print(f"[Skip] File not found for {ticker}: {file_path}")
        continue

    print(f"[Train] Using {selected_file} for {ticker}")
    df = pd.read_parquet(file_path)

    # === Data Debug Block ===
    if cfg.get("DEBUG"):
        logging.info(f"\n=== DEBUG: Data Summary for {ticker} ===")
        logging.info(f"[Shape] Rows: {len(df)}, Columns: {df.shape[1]}")
        logging.info(f"[Columns] {df.columns.tolist()}")
        logging.info(f"[NaN counts per column]\n{df.isnull().sum().to_string()}")
        logging.info(f"[First 5 rows]\n{df.head().to_string()}")

    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df=df, config=cfg)])

    # Create model
    model = PPO("MlpPolicy", env, seed=SEED, **ppo_params, verbose=1)

    # Setup callbacks
    callback = CallbackList(
        [PerformanceMetricsCallback(config_path="config.json", verbose=1)]
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    save_path = os.path.join(cfg["env"]["MODEL_DIR"], f"ppo_{ticker}")
    model.save(save_path)
    print(f"[Done] Model saved for {ticker} at {save_path}")
