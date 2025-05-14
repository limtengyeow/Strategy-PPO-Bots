# import unittest
import pandas as pd
from env.trading_env import TradingEnv

class TestTradingEnv(unittest.TestCase):
    def test_env_initialization(self):
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        env = TradingEnv(data, features=[], reward_components=[])
        obs = env.reset()
        self.assertEqual(env.action_space.n, 3)
        self.assertEqual(len(obs), env.observation_space.shape[0])
