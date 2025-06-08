# Version 2.3.2 - Unified CLI & Jupyter Support
# train_ppo.py - The Unified Training Engine (v2.3.2)
# This script trains a PPO agent based on a specified configuration.
# It incorporates best practices for reproducibility, stability, and experiment management.
"""
Expected YAML structure:

training:
  SEED: 42
  USE_GPU: true
  NUM_ENVS: 8
  TOTAL_TIMESTEPS: 1000000
  BOT_NAME: "ppo_breakout_v1"
  PPO_PARAMS: { ... } # PPO hyperparameters
env:
  DATA_PATH: "data/processed_data.parquet" # Optional, if not specified DATA_FOLDER is used
  DATA_FOLDER: "data/" # Used for auto-discovery if DATA_PATH is not set
  MODEL_DIR: "models/"
  LOG_DIR: "logs/"
callbacks:
  CheckpointCallback:
    save_freq: 100000
"""

import argparse
import logging
import multiprocessing
import os
import random
import sys  # Import sys to check for interactive mode
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Import DummyVecEnv for Jupyter compatibility
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Assuming your custom environment is in this location
from env.trading_env import TradingEnv

# --- Setup professional logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Helper to detect if running in an interactive (e.g., Jupyter/IPython) environment
def is_interactive():
    return (
        hasattr(sys, "ps1")
        or "IPython.core.interactiveshell.InteractiveShell" in sys.modules
    )


def load_config(path: str) -> dict:
    """Loads and validates the configuration file, performing data path auto-discovery."""
    logger.info(f"Loading configuration from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if "training" not in config:
        raise ValueError("Missing required section 'training' in config file.")
    # Ensure 'env' section exists (previously 'environment')
    if "env" not in config:
        raise ValueError("Missing required section 'env' in config file.")

    env_cfg = config["env"]  # Consistently use 'env'

    # Auto-discovery logic for DATA_PATH
    data_path = env_cfg.get("DATA_PATH")
    if not data_path:
        data_folder = env_cfg.get("DATA_FOLDER", "data/")
        # Ensure data_folder exists before listing its contents
        if not os.path.exists(data_folder):
            raise FileNotFoundError(f"Data folder not found: {data_folder}")

        parquet_files = [f for f in os.listdir(data_folder) if f.endswith(".parquet")]
        if not parquet_files:
            raise FileNotFoundError(
                f"❌ No .parquet files found in data folder: {data_folder}. Please run generate_data.py first."
            )
        data_path = os.path.join(data_folder, parquet_files[0])
        logger.warning(
            f"⚠️ No DATA_PATH specified. Using first available file: {data_path}"
        )
        # Crucial: Update the master config with the resolved path for downstream use
        config["env"]["DATA_PATH"] = data_path

    # Final check for the resolved data_path's existence
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ Data file not found: {data_path}. Please run generate_data.py first."
        )

    return config


def make_env(df: pd.DataFrame, config: dict, seed: int):
    """
    Utility function for multiprocessing to create a new environment instance.
    This function is designed to be picklable for SubprocVecEnv.
    """

    def _init():
        # Use deepcopy for absolute process safety of the config
        env_config = deepcopy(config)
        # Treat df as read-only to save memory (assuming df is passed by reference and is immutable)
        env = TradingEnv(config=env_config, df=df)
        env.seed(seed)
        return env

    return _init


def run_training_session(config: dict):
    """
    The core logic for a training session.
    Takes a configuration dictionary as input.
    """
    try:
        # --- 1. Setup & Configuration ---
        training_cfg = config.get("training", {})
        env_cfg = config.get("env", {})  # Consistently using 'env'

        seed = training_cfg.get("SEED", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        use_gpu = training_cfg.get("USE_GPU", False)
        device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )

        if device.type == "cuda":
            torch.cuda.manual_seed(seed)

        logger.info(f"Using seed: {seed}")
        logger.info(f"Using device: {device}")

        model_dir = env_cfg.get("MODEL_DIR", "models/")
        log_dir = env_cfg.get("LOG_DIR", "logs/")  # Ensure consistent default paths
        tensorboard_log_dir = env_cfg.get(
            "TENSORBOARD_LOG", "tb_logs/"
        )  # Use TENSORBOARD_LOG from config

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(
            tensorboard_log_dir, exist_ok=True
        )  # Ensure tensorboard log directory exists

        # --- 2. Data Loading ---
        data_path = env_cfg.get("DATA_PATH")

        # Add a critical check here, just before using data_path, to catch configuration errors early
        if data_path is None:
            logger.critical(
                "FATAL: DATA_PATH is None inside run_training_session. Configuration error."
            )
            raise ValueError(
                "DATA_PATH was not properly resolved or passed to run_training_session."
            )

        logger.info(f"Loading data from: {data_path}")
        df = pd.read_parquet(data_path)

        # --- 3. Environment Creation ---
        num_envs = training_cfg.get("NUM_ENVS", 4)

        # Unified environment creation logic based on interactive mode
        if is_interactive():
            logger.warning(
                "⚠️ Running in interactive environment (Jupyter/IPython). Using DummyVecEnv (single process)."
            )
            # DummyVecEnv does not use multiprocessing, suitable for interactive sessions
            # For DummyVecEnv, we typically only need one environment
            env = DummyVecEnv([make_env(df, config, seed)])
        else:
            logger.info(
                f"Launching {num_envs} parallel environments using SubprocVecEnv."
            )
            env = SubprocVecEnv(
                [make_env(df, config, seed + i) for i in range(num_envs)]
            )

        # --- 4. Model Training ---
        ppo_params = training_cfg.get("PPO_PARAMS", {})
        total_timesteps = training_cfg.get("TOTAL_TIMESTEPS", 100000)
        bot_name = training_cfg.get("BOT_NAME", "ppo_model")

        # Handle activation_fn from string to callable
        if (
            "policy_kwargs" in ppo_params
            and "activation_fn_str" in ppo_params["policy_kwargs"]
        ):
            activation_str = ppo_params["policy_kwargs"].pop(
                "activation_fn_str"
            )  # Get and remove string key
            if activation_str.lower() == "relu":
                ppo_params["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
            elif activation_str.lower() == "tanh":
                ppo_params["policy_kwargs"]["activation_fn"] = torch.nn.Tanh
            # Add more mappings as needed (e.g., LeakyReLU, Sigmoid)
            else:
                logger.warning(
                    f"Unknown activation function string '{activation_str}'. Defaulting to ReLU."
                )
                ppo_params["policy_kwargs"]["activation_fn"] = torch.nn.ReLU
        # This else-if handles legacy 'activation_fn' if it's still a string
        elif (
            "policy_kwargs" in ppo_params
            and "activation_fn" in ppo_params["policy_kwargs"]
        ):
            if isinstance(ppo_params["policy_kwargs"]["activation_fn"], str):
                logger.error(
                    "Legacy 'activation_fn' in PPO_PARAMS is a string. Please use 'activation_fn_str' in config.yaml."
                )
                raise ValueError(
                    "Invalid activation_fn configuration. Must be a callable or specified by 'activation_fn_str'."
                )

        logger.info(f"Initializing PPO model: {bot_name}")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log_dir,  # Use the correct tensorboard log directory
            **ppo_params,
        )

        # --- Checkpointing Callback ---
        # Use SAVE_FREQ from training_cfg, if not provided, default to total_timesteps // 10
        save_freq = training_cfg.get("SAVE_FREQ", max(total_timesteps // 10, 1000))
        checkpoint_cb = CheckpointCallback(
            save_freq=save_freq,
            save_path=model_dir,
            name_prefix=f"{bot_name}_checkpoint",
        )

        # --- Robust Training Loop ---
        try:
            logger.info(f"Starting training for {total_timesteps} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=[
                    checkpoint_cb
                ],  # Callbacks must be in a list for Stable Baselines3
                tb_log_name=bot_name,
                progress_bar=True,  # Added for interactive sessions
            )
        finally:
            logger.info("Closing environments.")
            env.close()

        # --- 5. Save Final Model ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(model_dir, f"{bot_name}_{timestamp}.zip")
        model.save(save_path)
        logger.info(
            f"✅ [SUCCESS] Training complete. Final model saved to: {save_path}"
        )

    except Exception:
        logger.exception("A critical error occurred during the training session.")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Engine")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()

    # Only set multiprocessing start method if NOT in interactive mode
    # This prevents the `__spec__` error in Jupyter and ensures CLI robustness.
    if (
        not is_interactive() and os.name != "nt"
    ):  # Check if not Windows and not interactive
        try:
            # Force 'spawn' method for robustness across different environments
            multiprocessing.set_start_method("spawn", force=True)
            logger.info("Multiprocessing start method set to 'spawn'.")
        except RuntimeError as e:
            # Log a warning if it can't be set (e.g., already set by another library)
            logger.warning(
                f"Could not set multiprocessing start method (may already be set): {e}"
            )

    try:
        master_config = load_config(args.config)

        # The redundant check was removed as load_config now handles auto-discovery and validation.
        # data_file_path = master_config.get("env", {}).get("DATA_PATH")
        # if not data_file_path or not os.path.exists(data_file_path):
        #     raise FileNotFoundError(
        #         f"Data file missing: {data_file_path}. Please run generate_data.py first."
        #     )

        run_training_session(master_config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(
            f"❌ Failed to start training due to configuration or data error: {e}"
        )
