import os
import random
from collections import deque

import gym
import numpy as np
from gym import spaces

from .action_handler import take_action
from .feature_engineering import init_feature_space
from .observation_builder import build_observation
from .reward_calculator import calculate_reward


class TradingEnv(gym.Env):
    def __init__(self, config=None, df=None):
        super().__init__()
        if df is None:
            raise ValueError("DataFrame 'df' must be provided to TradingEnv.")
        self.cfg = config or {}
        self.df = df.copy()
        self.current_step = 0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0

        # Configurations for features, actions, and rewards
        self.features_cfg = self.cfg.get("features", {})
        self.actions_cfg = self.cfg.get("actions", {})
        self.rewards_cfg = self.cfg.get("rewards", {})

        # Read debug flags from config
        self.env_debug = self.cfg.get(
            "DEBUG", False
        )  # Root-level DEBUG flag controls env logs
        self.feature_debug = self.features_cfg.get(
            "DEBUG_FEATURES", False
        )  # Features debug flag

        self.allow_long = self.actions_cfg.get("ALLOW_LONG", True)
        self.allow_short = self.actions_cfg.get("ALLOW_SHORT", False)
        self.obs_window = self.features_cfg.get("OBS_WINDOW", 5)
        self.price_field = self.features_cfg.get("price_field", "close")

        self.obs_buffer = deque(maxlen=self.obs_window)
        self.trade_log = []  # Placeholder, not used yet
        self.position_duration = 0

        os.makedirs("logs", exist_ok=True)
        self._init_spaces()

    # === Set seed for reproducibility ===
    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def _init_spaces(self):
        # Initialize features and observation space
        self.df, self.feature_columns = init_feature_space(
            self.df, self.features_cfg, self.feature_debug
        )
        feature_dim = len(self.feature_columns)

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim * self.obs_window,),
            dtype=np.float32,
        )

        if self.env_debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(
                    f"[INIT] Features selected: {self.feature_columns}, Feature count: {feature_dim}, Obs window: {self.obs_window}\n"
                )

    def reset(self):
        # Reset environment state
        if self.cfg.get("training", {}).get("RANDOM_START", False):
            max_start = len(self.df) - self.obs_window - 1
            self.current_step = np.random.randint(0, max_start)
        else:
            self.current_step = 0

        self.position = 0
        self.entry_price = 0.0
        self.obs_buffer.clear()
        self.trade_log.clear()
        self.position_duration = 0
        obs = self._get_features()
        for _ in range(self.obs_window):
            self.obs_buffer.append(obs)

        if self.env_debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(f"[RESET] Environment reset. Start step: {self.current_step}\n")

        return self._get_observation()

    def step(self, action):
        prev_price = self._get_price()
        take_action(self, action, prev_price)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = calculate_reward(self)
        obs = self._get_features()
        self.obs_buffer.append(obs)

        if self.position != 0:
            self.position_duration += 1

        info = {
            "step": self.current_step,
            "reward": reward,
            "position": self.position,
            "entry_price": self.entry_price,
            "price": self._get_price(),
            "duration": self.position_duration,
            "trade_log": list(self.trade_log),  # Placeholder
        }

        if self.env_debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(
                    f"[STEP] step={self.current_step}, action={action}, reward={reward:.5f}, pos={self.position}, entry_price={self.entry_price:.2f}, price={self._get_price():.2f}, duration={self.position_duration}\n"
                )

        return self._get_observation(), reward, done, info

    def _get_price(self):
        return self.df.iloc[self.current_step][self.price_field]

    def _get_features(self):
        return self.df.loc[self.current_step, self.feature_columns].values.astype(
            np.float32
        )

    def _get_observation(self):
        obs = build_observation(self.obs_buffer)
        if np.any(np.isnan(obs)):
            raise ValueError(
                "[NaN DETECTED] Observation contains NaN values. Check feature preprocessing."
            )
        return obs
