import gym
import random
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

        self.episode_duration_days = self.cfg["env"].get("EPISODE_DURATION_DAYS", 10)
        self.bars_per_day = self.cfg["env"].get("BARS_PER_DAY", 78)
        self.max_episode_steps = self.episode_duration_days * self.bars_per_day
        self.episode_steps = 0

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

        # Check for NaNs in features after preprocessing
        if self.df[self.feature_columns].isnull().values.any():
            raise ValueError("[INIT] Feature DataFrame contains NaNs. Check feature engineering.")



   # === Set seed for reproducibility ===
    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]


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
            print(f"[OBS SPACE] Features selected: {self.feature_columns}")
            print(f"[OBS SPACE] Feature count: {feature_dim}")
            print(f"[OBS SPACE] Observation window: {self.obs_window}")
            print(f"[OBS SPACE] Final shape: {self.observation_space.shape}")
 

    def reset(self):

 
        if self.cfg.get("training", {}).get("RANDOM_START", False):
            max_start = len(self.df) - self.obs_window - self.max_episode_steps - 1
            if max_start <= 0:
                raise ValueError(f"[RESET] Dataset too small for RANDOM_START. max_start={max_start}")
            self.current_step = np.random.randint(0, max_start)


        else:
                self.current_step = 0
                self.episode_steps = 0
                self.cumulative_pnl = 0.0  # If you're using it


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
                f.write(f"[RESET] Environment reset. Start step: {self.current_step}\n")

        return self._get_observation()

    def step(self, action):
        prev_price = self._get_price()
        take_action(self, action, prev_price)
        self.current_step += 1
        self.episode_steps += 1

        # --- End episode after fixed number of steps or end of data ---
        done = self.episode_steps >= self.max_episode_steps or self.current_step >= len(self.df) - 1

        # === Trade logging logic (before calculating reward) ===
        if self.position != 0:
            closing_action = (self.position == 1 and action == 2) or (self.position == -1 and action == 1)
            if closing_action:
                current_price = self._get_price()
                if self.entry_price != 0:
                    if self.position == 1:
                        pnl = (current_price - self.entry_price) / self.entry_price
                    else:
                        pnl = (self.entry_price - current_price) / self.entry_price
                else:
                    pnl = 0.0

                self.trade_log.append({
                    "pnl": pnl,
                    "duration": self.position_duration
                })
                self.entry_price = 0.0
                self.position_duration = 0

        # === Reward ===
        reward = calculate_reward(self)

        # === Force-close open position at episode end ===
        if done and self.position != 0:
            if self.entry_price != 0:
                final_pnl = (self._get_price() - self.entry_price) / self.entry_price
                if self.position == -1:
                    final_pnl *= -1
                reward += final_pnl * self.rewards_cfg.get("REWARD_COMPONENTS", [{}])[0].get("scale", 100)
            else:
                if self.debug:
                    with open("logs/env_debug.log", "a") as f:
                        f.write(f"[WARNING] Entry price is zero at episode end. Skipping final reward calc.\n")


            self.position = 0
            self.entry_price = 0.0

        # === Observation and bookkeeping ===
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

        if done:
            print(f"[EPISODE END] Steps: {self.episode_steps}, Reward: {reward:.2f}")

        return self._get_observation(), reward, done, info

    
    def _get_price(self):
        return self.df.iloc[self.current_step][self.price_field]

    def _get_features(self):
        return self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)

    def _get_observation(self):
        obs = build_observation(self.obs_buffer)
        if np.any(np.isnan(obs)):
            raise ValueError("[NaN DETECTED] Observation contains NaN values. Check feature preprocessing.")
        return obs
