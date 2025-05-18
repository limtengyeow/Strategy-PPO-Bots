import os
import json
import torch
import pandas as pd
import multiprocessing
import random
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CallbackList
from env.trading_env import TradingEnv
from callbacks.performance_metrics_callback import PerformanceMetricsCallback
from train.train_ppo import load_config  # assumes this exists and loads config.json

def make_env(df, cfg, seed):
    def _init():
        env = TradingEnv(df=df.copy(), config=cfg)
        env.seed(seed)
        return env
    return _init

def main():
    multiprocessing.set_start_method("spawn", force=True)

   # Support dynamic config file override via ENV variable
    config_path = os.getenv("CONFIG_PATH", "config.json")
    print(f"[Config] Using configuration file: {config_path}")
    cfg = load_config(config_path)

   # === Set seed for reproducibility ===
    assert "SEED" in cfg["training"], "[Config] 'SEED' must be specified in training config."
    SEED = cfg["training"]["SEED"]
    print(f"[Seed] Using seed: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


    requested_gpu = cfg["training"].get("USE_GPU", False)
    device = torch.device("cuda" if requested_gpu and torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("[Perf] GPU optimization flags enabled.")
    else:
        print("[Perf] On CPU, overriding PPO config")
        cfg["training"]["PPO_PARAMS"]["n_steps"] = 2048
        cfg["training"]["PPO_PARAMS"]["batch_size"] = 64

    data_dir = cfg["env"]["DATA_FOLDER"]
    model_dir = cfg["env"]["MODEL_DIR"]
    os.makedirs(model_dir, exist_ok=True)

    timeframe_order = ["1minute", "5minute", "15minute", "1hour", "1day", "1week"]
    ticker_files = {}
    for f in os.listdir(data_dir):
        if f.endswith(".csv") and "_" in f:
            ticker, rest = f.split("_", 1)
            ticker_files.setdefault(ticker, []).append(f)

    for ticker, files in ticker_files.items():
        files_sorted = sorted(
            files,
            key=lambda x: timeframe_order.index(x.split("_")[1].replace(".csv", ""))
            if x.split("_")[1].replace(".csv", "") in timeframe_order else len(timeframe_order)
        )
        file_path = os.path.join(data_dir, files_sorted[0])
        print(f"[Train] Using {files_sorted[0]} for {ticker}")

        df = pd.read_csv(file_path)
        NUM_ENVS = 4
        env = SubprocVecEnv([make_env(df, cfg, SEED + i) for i in range(NUM_ENVS)])

        ppo_params = cfg["training"]["PPO_PARAMS"]
        model = PPO("MlpPolicy", env, seed=SEED, **ppo_params, verbose=1)

        model.policy.to(device)

        callback = CallbackList([
            PerformanceMetricsCallback(config_path="config.json", verbose=1)
        ])

        model.learn(total_timesteps=cfg["training"]["TOTAL_TIMESTEPS"], callback=callback)

        save_path = os.path.join(model_dir, f"ppo_{ticker}")
        model.save(save_path)
        print(f"[Done] Model saved for {ticker} at {save_path}")

if __name__ == "__main__":
    main()
