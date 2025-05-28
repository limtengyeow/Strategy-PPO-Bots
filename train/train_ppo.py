import copy
import json
import os
import random
from datetime import datetime
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

# === Setup train debug log ===
LOG_FILE = Path.cwd() / "logs" / "train_debug.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_train_debug(message, debug):
    if debug:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        with LOG_FILE.open("a") as f:
            f.write(log_line + "\n")


# === Load Config ===
def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)


# === Parse command-line arguments (config file path) ===
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, default="config.json", help="Path to config file"
)
args, _ = parser.parse_known_args()
cfg = load_config(args.config)

# === Set Seed ===
SEED = cfg["training"]["SEED"]
DEBUG = cfg.get("DEBUG", False)
log_train_debug(f"Training script started with SEED={SEED}", DEBUG)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Map Activation Function Strings ===
activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
if "policy_kwargs" in cfg["training"]["PPO_PARAMS"]:
    act_str = cfg["training"]["PPO_PARAMS"]["policy_kwargs"].get(
        "activation_fn", "relu"
    )
    cfg["training"]["PPO_PARAMS"]["policy_kwargs"]["activation_fn"] = (
        activation_map.get(act_str, nn.ReLU)
    )

# === Log Config Once (deepcopy, stringified activation_fn) ===
config_for_log = copy.deepcopy(cfg)
if "policy_kwargs" in config_for_log["training"]["PPO_PARAMS"]:
    config_for_log["training"]["PPO_PARAMS"]["policy_kwargs"]["activation_fn"] = str(
        config_for_log["training"]["PPO_PARAMS"]["policy_kwargs"]["activation_fn"]
    )
log_train_debug(f"Training config: {json.dumps(config_for_log, indent=2)}", DEBUG)

# === TensorBoard Log Path ===
tensorboard_log_path = cfg["logging"].get("tensorboard_log", "tb_logs/PPO_Trading/")

# === Discover Ticker Files ===
data_dir = cfg["env"]["DATA_FOLDER"]
timeframe_order = ["1minute", "5minute", "15minute", "1hour", "1day", "1week"]

ticker_files = {}
for f in os.listdir(data_dir):
    if f.endswith(".csv") and "_" in f:
        ticker, rest = f.split("_", 1)
        ticker_files.setdefault(ticker, []).append(f)

# === Main Training Loop ===
for ticker, files in ticker_files.items():
    files_sorted = sorted(
        files,
        key=lambda x: timeframe_order.index(x.split("_")[1].replace(".csv", ""))
        if x.split("_")[1].replace(".csv", "") in timeframe_order
        else len(timeframe_order),
    )
    selected_file = files_sorted[0]
    file_path = os.path.join(data_dir, selected_file)

    if not os.path.exists(file_path):
        log_train_debug(f"[Skip] File not found for {ticker}: {file_path}", DEBUG)
        continue

    log_train_debug(
        f"[Train] Starting training for {ticker} using {selected_file}", DEBUG
    )
    df = pd.read_csv(file_path)

    # === Debug Block ===
    if DEBUG:
        log_train_debug(f"Data Summary for {ticker}:", DEBUG)
        log_train_debug(f"[Shape] Rows: {len(df)}, Columns: {df.shape[1]}", DEBUG)
        log_train_debug(f"[Columns] {df.columns.tolist()}", DEBUG)
        log_train_debug(f"[NaN counts per column]\n{df.isnull().sum()}", DEBUG)
        feature_cols = ["close", "open", "high", "low", "volume"]
        available = [col for col in feature_cols if col in df.columns]
        if available:
            log_train_debug(
                f"[Stats for key features]\n{df[available].describe()}", DEBUG
            )

    # === Create Environment ===
    env = DummyVecEnv([lambda: TradingEnv(df=df, config=cfg)])

    # === Create PPO Model ===
    ppo_params = cfg["training"]["PPO_PARAMS"]
    model = PPO(
        "MlpPolicy",
        env,
        seed=SEED,
        tensorboard_log=tensorboard_log_path,
        **ppo_params,
        verbose=1,
    )

    # === Setup Callbacks ===
    callbacks = CallbackList(
        [PerformanceMetricsCallback(config_path=args.config, verbose=1)]
    )

    # === Train Model ===
    total_timesteps = cfg["training"]["TOTAL_TIMESTEPS"]
    log_train_debug(
        f"Training for {ticker} started, total timesteps: {total_timesteps}", DEBUG
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    log_train_debug(f"Training for {ticker} completed", DEBUG)

    # === Save Model ===
    save_path = os.path.join(cfg["env"]["MODEL_DIR"], f"ppo_{ticker}")
    model.save(save_path)
    log_train_debug(f"Model saved for {ticker} at {save_path}", DEBUG)
