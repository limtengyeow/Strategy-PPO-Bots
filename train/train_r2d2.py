import json
import os
import random

import numpy as np
import pandas as pd
import torch

from agents.r2d2_agent import R2D2Agent
from env.trading_env import TradingEnv
from replay.replay_buffer import ReplayBuffer


# === Load config.json ===
def load_config(config_path="config.json"):
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg


cfg = load_config()

# === Set seed for reproducibility ===
assert "SEED" in cfg["training"], (
    "[Config] 'SEED' must be specified in training config."
)
SEED = cfg["training"]["SEED"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and cfg["training"]["USE_GPU"] else "cpu"
)

# === Hyperparameters ===
R2D2_CFG = cfg["training"]["R2D2_PARAMS"]
ACTION_SIZE = cfg["model"]["action_size"]
HIDDEN_SIZE = cfg["model"]["hidden_size"]
SEQ_LEN = R2D2_CFG["seq_len"]
BUFFER_SIZE = R2D2_CFG["buffer_size"]
BATCH_SIZE = R2D2_CFG["batch_size"]
EPISODES = R2D2_CFG["episodes"]
GAMMA = R2D2_CFG["gamma"]
EPSILON = R2D2_CFG["epsilon"]
TARGET_UPDATE_FREQ = R2D2_CFG["target_update_freq"]
MODEL_SAVE_PATH = R2D2_CFG["model_save_path"]

# === Load Data ===
data_dir = cfg["env"]["DATA_FOLDER"]
timeframe_order = ["1minute", "5minute", "15minute", "1hour", "1day", "1week"]

ticker_files = {}
for f in os.listdir(data_dir):
    if f.endswith(".csv") and "_" in f:
        ticker, rest = f.split("_", 1)
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(f)

if not ticker_files:
    raise ValueError("No CSV files found in data folder.")

ticker = list(ticker_files.keys())[0]
files_sorted = sorted(
    ticker_files[ticker],
    key=lambda x: timeframe_order.index(x.split("_")[1].replace(".csv", ""))
    if x.split("_")[1].replace(".csv", "") in timeframe_order
    else len(timeframe_order),
)
selected_file = files_sorted[0]
file_path = os.path.join(data_dir, selected_file)

print(f"[Train R2D2] Using {selected_file} for {ticker}")
df = pd.read_csv(file_path)

# === Initialize Components ===
env = TradingEnv(cfg, df)
INPUT_SIZE = len(env.feature_columns)  # Number of features per timestep

agent = R2D2Agent(INPUT_SIZE, ACTION_SIZE, HIDDEN_SIZE, gamma=GAMMA).to(DEVICE)
replay_buffer = ReplayBuffer(BUFFER_SIZE, SEQ_LEN)

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
total_steps = 0

# === Training Loop ===
for episode in range(1, EPISODES + 1):
    state = env.reset()  # shape: (obs_window, feature_dim)
    state = np.array(state, dtype=np.float32)
    hidden = (
        torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
        torch.zeros(1, 1, HIDDEN_SIZE).to(DEVICE),
    )
    episode_buffer = []
    done = False
    ep_reward = 0.0

    while not done:
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )  # (1, obs_window, feature_dim)

        action, hidden = agent.act(state_tensor, hidden, epsilon=EPSILON)
        next_state, reward, done, info = env.step(action)
        next_state = np.array(next_state, dtype=np.float32)

        next_state_tensor = (
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        )

        episode_buffer.append(
            (state_tensor, action, reward, next_state_tensor, done, hidden)
        )
        state = next_state
        ep_reward += reward
        total_steps += 1

    replay_buffer.push(episode_buffer)

    # Train
    if len(replay_buffer) >= BATCH_SIZE:
        batch = replay_buffer.sample(BATCH_SIZE)
        loss = agent.train_step(batch)
    else:
        loss = 0.0

    # Target Network Update
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # Logging
    if episode % 10 == 0:
        print(
            f"[Episode {episode}] Reward: {ep_reward:.4f} | Loss: {loss:.4f} | Buffer: {len(replay_buffer)}"
        )

    # Save model
    if episode % 50 == 0:
        torch.save(
            agent.state_dict(), os.path.join(MODEL_SAVE_PATH, f"r2d2_ep{episode}.pt")
        )

print("Training complete! 🚀")
