import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque
import os
from .feature_engineering import init_feature_space
from .action_handler import take_action
from .reward_calculator import calculate_reward
from .observation_builder import build_observation

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

        self.features_cfg = self.cfg.get("features", {})
        self.actions_cfg = self.cfg.get("actions", {})
        self.rewards_cfg = self.cfg.get("rewards", {})

        self.allow_long = self.actions_cfg.get("ALLOW_LONG", True)
        self.allow_short = self.actions_cfg.get("ALLOW_SHORT", False)
        self.obs_window = self.features_cfg.get("OBS_WINDOW", 5)
        self.price_field = self.features_cfg.get("price_field", "close")

        self.obs_buffer = deque(maxlen=self.obs_window)
        self.debug = self.cfg.get("DEBUG", False)
        self.trade_log = []
        self.position_duration = 0

        os.makedirs("logs", exist_ok=True)
        self._init_spaces()

    def _init_spaces(self):
        self.df, self.feature_columns = init_feature_space(self.df, self.features_cfg, self.debug)
        feature_dim = len(self.feature_columns)

        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim * self.obs_window,),
            dtype=np.float32
        )

        if self.debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(f"[INIT] Features: {self.feature_columns}\n")

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.obs_buffer.clear()
        self.trade_log.clear()
        self.position_duration = 0
        obs = self._get_features()
        for _ in range(self.obs_window):
            self.obs_buffer.append(obs)
        if self.debug:
            with open("logs/env_debug.log", "a") as f:
                f.write("[RESET] Environment reset.\n")
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
            "trade_log": list(self.trade_log)
        }

        if self.debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(
                    f"[STEP] step={self.current_step}, action={action}, reward={reward:.5f}, pos={self.position}, price={self._get_price():.2f}, duration={self.position_duration}\n"
                )

        return self._get_observation(), reward, done, info

    def _get_price(self):
        return self.df.iloc[self.current_step][self.price_field]

    def _get_features(self):
        return self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)

    def _get_observation(self):
        return build_observation(self.obs_buffer)
