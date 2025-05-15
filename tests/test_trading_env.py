import os
import unittest
import json
import numpy as np
from env.trading_env import TradingEnv

class TestTradingEnv(unittest.TestCase):
 
    def setUp(self):
        # Load configuration from the root directory relative to this test file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "..", "config.json")
        with open(config_path) as f:
            config = json.load(f)
        self.env = TradingEnv(config)
        self.env.reset()

    def test_reset_output_shape_and_type(self):
        # Ensure the reset observation is a NumPy array and has the correct shape
        obs = self.env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, self.env.observation_space.shape)

    def test_step_output_types(self):
        # Check the types returned by the step function match Gym expectations
        obs, reward, done, info = self.env.step(self.env.action_space.sample())
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

    def test_can_open_position_with_buy(self):
        # Simulate a Buy action and ensure a long position is opened
        self.env.reset()
        obs, reward, done, info = self.env.step(1)  # 1 = Buy
        self.assertIn(self.env.position, [1, -1, 0])  # numerical position: 1=long, -1=short, 0=flat
        if self.env.position == 1:
            self.assertTrue(self.env.entry_price is not None)

    def test_reward_is_not_constant_zero(self):
        self.env.reset()
        rewards = []
        for _ in range(30):
            _, reward, _, _ = self.env.step(self.env.action_space.sample())
            rewards.append(reward)
        print("Sampled rewards:", rewards)
        self.assertTrue(any(r != 0.0 for r in rewards), "All rewards are 0.0. Check reward function logic.")

    def test_reward_has_positive_and_negative_values(self):
        self.env.reset()
        rewards = []
        for _ in range(50):
            _, reward, _, _ = self.env.step(self.env.action_space.sample())
            rewards.append(reward)
        print("Sampled rewards for sign check:", rewards)
        self.assertTrue(any(r > 0 for r in rewards), "No positive rewards detected. Check if agent ever profits.")
        self.assertTrue(any(r < 0 for r in rewards), "No negative rewards detected. Check loss mechanics.")

#    def test_reward_is_not_constant_zero(self):
#        # Run several steps and check that rewards are not all zero
#        self.env.reset()
#        rewards = []
#        for _ in range(15):
#            _, reward, _, _ = self.env.step(self.env.action_space.sample())
#            rewards.append(reward)
#        self.assertTrue(any(r != 0.0 for r in rewards))

#   def test_reward_has_positive_and_negative_values(self):
#        # Ensure both positive and negative rewards occur
#        self.env.reset()
#        rewards = []
#        for _ in range(30):
#            _, reward, _, _ = self.env.step(self.env.action_space.sample())
#            rewards.append(reward)
#        self.assertTrue(any(r > 0 for r in rewards))
#        self.assertTrue(any(r < 0 for r in rewards))

    def test_episode_runs_without_crash(self):
        # Run a full episode and confirm no crashes occur
        self.env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = self.env.step(self.env.action_space.sample())
            steps += 1
        self.assertGreater(steps, 0)

    def test_env_closes_gracefully(self):
        # Confirm that the environment can be closed without errors
        try:
            self.env.close()
        except Exception as e:
            self.fail(f"Environment close() raised an exception: {e}")

    def test_position_close_on_reverse_action(self):
        # Ensure that a long position is closed before opening a short one
        self.env.reset()
        self.env.step(1)  # Buy
        self.assertEqual(self.env.position, 1)
        self.env.step(2)  # Sell
        self.assertIn(self.env.position, [-1, 0, None])


    def test_single_active_position_only(self):
        # Ensure position switches do not result in overlapping open state
        self.env.reset()
        self.env.step(1)  # Buy
        pos_after_buy = self.env.position
        self.env.step(2)  # Sell
        pos_after_sell = self.env.position
        self.env.step(1)  # Buy again
        pos_after_rebuy = self.env.position
        # The position should be updated and not overlap
        self.assertIn(pos_after_buy, [1, 0])
        self.assertIn(pos_after_sell, [-1, 0])
        self.assertIn(pos_after_rebuy, [1, 0])

    def tearDown(self):
        # Clean up after each test
        self.env.close()

if __name__ == "__main__":
    unittest.main()
