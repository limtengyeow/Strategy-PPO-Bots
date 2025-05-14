import gym
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df, config):
        super(TradingEnv, self).__init__()
        self.df = df
        self.config = config
        self.current_step = 0
        self.done = False
        self.position = 0  # 0 = flat, 1 = long, -1 = short
        self.cash = 100000
        self.shares_held = 0
        self.net_worth = self.cash

        # Observation space: all columns except timestamp
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns) - 1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

    def reset(self):
        self.current_step = 0
        self.done = False
        self.position = 0
        self.cash = 100000
        self.shares_held = 0
        self.net_worth = self.cash
        return self._get_obs()

    def _get_obs(self):
        obs = self.df.iloc[self.current_step].drop(["timestamp"], errors='ignore').values
        return obs.astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0, self.done, {}

        row = self.df.iloc[self.current_step]
        price = row["close"]

        reward = 0
        if action == 1:  # buy
            if self.cash > 0:
                self.shares_held = self.cash / price
                self.cash = 0
                self.position = 1
        elif action == 2:  # sell
            if self.shares_held > 0:
                self.cash = self.shares_held * price
                self.shares_held = 0
                self.position = -1

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self.done = True

        self.net_worth = self.cash + self.shares_held * price
        reward = self.net_worth - 100000  # profit since start
        return self._get_obs(), reward, self.done, {"net_worth": self.net_worth}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}")
