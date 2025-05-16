import gym
import numpy as np
import pandas as pd
from gym import spaces
from collections import deque
import os

class TradingEnv(gym.Env):
    def __init__(self, config=None, df=None):
        super().__init__()
        if df is None:
            raise ValueError("DataFrame 'df' must be provided to TradingEnv.")
        self.cfg = config or {}
        training_cfg = self.cfg.get("training", {})
        self.cfg = config or {}
        self.df = df
        self.current_step = 0
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.entry_price = 0.0
        training_cfg = self.cfg.get("training", {})
        self.allow_long = training_cfg.get("ALLOW_LONG", True)
        self.allow_short = training_cfg.get("ALLOW_SHORT", False)
        self.obs_window = training_cfg.get("OBS_WINDOW", 5)

        self.obs_buffer = deque(maxlen=self.obs_window)
        self.debug = self.cfg.get("DEBUG", False)
        self.trade_log = []
        self.position_duration = 0

        os.makedirs("logs", exist_ok=True)
        self._init_spaces()



#    def _init_spaces(self):
#        features_cfg = self.cfg.get("training", {}).get("FEATURES", [])

#        selected_cols = []
#        for feat in features_cfg:
#            if feat.get("type") == "price":
#                field = feat.get("field")
#                if field in self.df.columns and pd.api.types.is_numeric_dtype(self.df[field]):
#                    selected_cols.append(field)
#        if not selected_cols:
#            raise ValueError("No valid numeric features selected in config under 'FEATURES'.")
#        self.feature_columns = selected_cols
#        feature_dim = len(self.feature_columns)
#        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
#        self.observation_space = spaces.Box(
#            low=-np.inf, high=np.inf, shape=(feature_dim * self.obs_window,), dtype=np.float32
#        )
#        if self.debug:
#            with open("logs/env_debug.log", "a") as f:
#                f.write(f"[INIT] Features: {self.feature_columns}\n")

    def _init_spaces(self):
        features_cfg = self.cfg.get("training", {}).get("FEATURES", [])
        selected_cols = []

        for feat in features_cfg:
            if feat.get("type") == "price":
                field = feat.get("field")
                if field in self.df.columns:
                    series = self.df[field]
                    if pd.api.types.is_numeric_dtype(series) and not series.isnull().all():
                        selected_cols.append(field)
                    else:
                        if self.debug:
                            print(f"[WARN] Skipping '{field}': not numeric or all values are NaN")
                else:
                    if self.debug:
                        print(f"[WARN] Skipping '{field}': column not found in DataFrame")

        if not selected_cols:
            raise ValueError("No valid numeric features selected in config under 'FEATURES'.")

        self.feature_columns = selected_cols
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
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = self._calculate_reward()
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

    def _take_action(self, action):
        price = self._get_price()
        if action == 1:
            if self.allow_long and self.position == 0:
                self.position = 1
                self.entry_price = price
                self.position_duration = 0
                self.trade_log.append({"action": "long", "entry": price, "step": self.current_step})
            elif self.allow_short and self.position == 1:
                pnl = price - self.entry_price
                self.trade_log.append({"action": "exit_long", "exit": price, "pnl": pnl, "step": self.current_step})
                self.position = -1
                self.entry_price = price
                self.position_duration = 0
                self.trade_log.append({"action": "short", "entry": price, "step": self.current_step})
        elif action == 2:
            if self.allow_short and self.position == 0:
                self.position = -1
                self.entry_price = price
                self.position_duration = 0
                self.trade_log.append({"action": "short", "entry": price, "step": self.current_step})
            elif self.allow_long and self.position == -1:
                pnl = self.entry_price - price
                self.trade_log.append({"action": "exit_short", "exit": price, "pnl": pnl, "step": self.current_step})
                self.position = 1
                self.entry_price = price
                self.position_duration = 0
                self.trade_log.append({"action": "long", "entry": price, "step": self.current_step})
            elif self.position != 0:
                pnl = (price - self.entry_price) if self.position == 1 else (self.entry_price - price)
                self.trade_log.append({"action": "exit", "exit": price, "pnl": pnl, "step": self.current_step})
                self.position = 0
                self.entry_price = 0.0
                self.position_duration = 0

    def _calculate_reward(self):
        if self.position == 0:
            return 0.0
        current_price = self._get_price()
        if self.position == 1:
            return (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            return (self.entry_price - current_price) / self.entry_price
        return 0.0

    def _get_price(self):
        return self.df.iloc[self.current_step]["close"]

    def _get_features(self):
        return self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)

    def _get_observation(self):
        return np.concatenate(list(self.obs_buffer)).astype(np.float32)
