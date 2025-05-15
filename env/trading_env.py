import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, config=None, df=None):
        super().__init__()
        if df is None:
            raise ValueError("DataFrame 'df' must be provided to TradingEnv.")
        self.cfg = config or {}
        self.df = df
        self.current_step = 0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0
        self.obs_window = self.cfg.get("OBS_WINDOW", 5)
        self.obs_buffer = deque(maxlen=self.obs_window)

        self._init_spaces()

    def _init_spaces(self):
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        feature_dim = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(feature_dim * self.obs_window,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.obs_buffer.clear()
        obs = self._get_features()
        for _ in range(self.obs_window):
            self.obs_buffer.append(obs)
        return self._get_observation()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = self._calculate_reward()
        obs = self._get_features()
        self.obs_buffer.append(obs)

        info = {
            "step": self.current_step,
            "reward": reward,
            "position": self.position,
            "entry_price": self.entry_price,
            "price": self._get_price()
        }

        return self._get_observation(), reward, done, info

    def _take_action(self, action):
        price = self._get_price()
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:
            self.position = 0
            self.entry_price = 0.0

    def _calculate_reward(self):
        if self.position == 0:
            return 0.0
        current_price = self._get_price()
        return (current_price - self.entry_price) / self.entry_price

    def _get_price(self):
        return self.df.iloc[self.current_step]["close"]

    def _get_features(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def _get_observation(self):
        return np.concatenate(list(self.obs_buffer)).astype(np.float32)
