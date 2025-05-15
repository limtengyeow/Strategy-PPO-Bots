
import gym
import numpy as np
import pandas as pd
from collections import deque

class TradingEnv(gym.Env):
    def __init__(self, config=None, df=None):
        super(TradingEnv, self).__init__()
        self.cfg = config or {}
        self.data = df
        self.current_step = 0
        self.obs_window = self.cfg.get("OBS_WINDOW", 20)
        self.obs_buffer = deque(maxlen=self.obs_window)

        self.allow_long = self.cfg.get("ALLOW_LONG", True)
        self.allow_short = self.cfg.get("ALLOW_SHORT", True)
        self.reward_components = self.cfg.get("REWARD_COMPONENTS", [])

        self.norm_obs = self.cfg.get("NORM_OBS", False)
        self.norm_type = self.cfg.get("NORM_TYPE", "returns")
        self.norm_volume = self.cfg.get("NORM_VOLUME", False)

        if self.norm_obs and self.data is not None and self.norm_type == "returns":
            self.data = self._compute_return_features(self.data)

        self.position = 0
        self.entry_price = 0.0
        self.highest_price_since_entry = 0.0
        self.took_profit = False
        self.last_reward = 0.0
        self.ema_window = 20
        self.price_history = []
        self.open_step = None
        self.trade_log = []

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._get_feature_dim() * self.obs_window,),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        if self.data is None:
            self._load_data()
        self.position = 0
        self.entry_price = 0.0
        self.highest_price_since_entry = 0.0
        self.took_profit = False
        self.price_history = []
        self.open_step = None
        self.trade_log = []
        self.last_reward = 0.0

        initial_obs = self._get_features()
        for _ in range(self.obs_window):
            self.obs_buffer.append(initial_obs)
        return self._get_sliding_obs()

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        reward = self._get_reward()
        self.last_reward = reward

        obs = self._get_features()
        self.obs_buffer.append(obs)

        info = {
            "step": self.current_step,
            "reward": reward,
            "trades": self.trade_log.copy()
        }
        self.trade_log.clear()
        return self._get_sliding_obs(), reward, done, info

    def _load_data(self):
        self.data = pd.DataFrame(np.random.randn(1000, 5), columns=["timestamp", "close", "volume", "sma", "donchian"])

    def _compute_return_features(self, df):
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])

        for col in df.columns:
            if col.lower() == "volume" and self.norm_volume:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)
            else:
                df[col] = df[col].pct_change().fillna(0)
        return df

    def _get_features(self):
        row = self.data.iloc[self.current_step]
        if "timestamp" in row.index:
            row = row.drop("timestamp")
        return row.values.astype(np.float32)

    def _get_price(self):
        return self.data.iloc[self.current_step]["close"]

    def _get_ema(self, window):
        prices = pd.Series(self.price_history[-window:])
        return prices.ewm(span=window, adjust=False).mean().iloc[-1] if len(prices) >= window else prices.mean()

    def _get_unrealized_return(self):
        if self.position == 0:
            return 0.0
        current_price = self._get_price()
        if self.position == 1:
            return (current_price - self.entry_price) / self.entry_price
        elif self.position == -1:
            return (self.entry_price - current_price) / self.entry_price
        return 0.0

    def _close_position(self):
        current_price = self._get_price()
        exit_step = self.current_step
        if self.open_step is not None:
            trade = {
                "pnl": self._get_unrealized_return(),
                "duration": exit_step - self.open_step,
                "risk": 0.01,
                "equity": current_price
            }
            self.trade_log.append(trade)
        self.position = 0
        self.entry_price = 0.0
        self.highest_price_since_entry = 0.0
        self.took_profit = False
        self.open_step = None

    def _partial_exit(self, fraction):
        self.took_profit = True

    def _get_reward(self):
        debug_rewards = self.cfg.get("DEBUG_REWARDS", False)
        reward_details = []
        reward = 0.0
        current_price = self._get_price()
        base_scale = next((comp.get("scale", 1000.0) for comp in self.reward_components if comp["type"] == "pnl"), 1000.0)
        self.price_history.append(current_price)
        unrealized = self._get_unrealized_return()

        if self.position == 1:
            self.highest_price_since_entry = max(self.highest_price_since_entry, current_price)

        if self.position == 1 and self.open_step and (self.current_step - self.open_step) >= 20:
            self._close_position()

        for comp in self.reward_components:
            if comp["type"] == "pnl":
                comp_reward = unrealized * comp.get("scale", 1.0)
                reward += comp_reward
                if debug_rewards:
                    reward_details.append(("pnl", comp_reward))
            elif comp["type"] == "cut_loss" and unrealized <= comp["threshold"]:
                penalty = -abs(unrealized * base_scale * comp.get("penalty_ratio", 1.0))
                reward += penalty
                if debug_rewards:
                    reward_details.append(("cut_loss", penalty))
                self._close_position()
            elif comp["type"] == "trend_hold_bonus":
                ema = self._get_ema(comp.get("ema_window", 20))
                if self.position == 1 and current_price > ema:
                    trend_bonus = abs(unrealized * base_scale * comp.get("bonus_ratio", 0.05))
                    reward += trend_bonus
                    if debug_rewards:
                        reward_details.append(("trend_hold_bonus", trend_bonus))
            elif comp["type"] == "overtrade_penalty" and self.open_step is not None:
                duration = self.current_step - self.open_step
                if duration < comp.get("min_duration", 5):
                    penalty = -abs(unrealized * base_scale * comp.get("penalty_ratio", 1.0))
                    reward += penalty
                    if debug_rewards:
                        reward_details.append(("overtrade_penalty", penalty))

        if debug_rewards and reward_details:
            print(f"[REWARD DEBUG] Step {self.current_step}:")
            for name, val in reward_details:
                print(f"  {name}: {val:.4f}")

        return reward

    def _take_action(self, action):
        current_price = self._get_price()
        if action == 1:
            if self.position == -1:
                self._close_position()
            elif self.allow_long and self.position == 0:
                self.position = 1
                self.entry_price = current_price
                self.highest_price_since_entry = current_price
                self.open_step = self.current_step
        elif action == 2:
            if self.position == 1:
                self._close_position()
            elif self.allow_short and self.position == 0:
                self.position = -1
                self.entry_price = current_price
                self.open_step = self.current_step

    def _get_sliding_obs(self):
        return np.array(self.obs_buffer).flatten()

    def _get_feature_dim(self):
        if self.data is not None:
            cols = [c for c in self.data.columns if c != "timestamp"]
            return len(cols)
        return 4
