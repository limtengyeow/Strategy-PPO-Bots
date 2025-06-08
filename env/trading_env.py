import logging
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Assuming these are in the same directory or accessible via Python path
from .action_handler import take_action
from .reward_calculator import calculate_reward

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    A custom Gym environment for trading, designed to work with pre-processed
    dataframes containing both raw price data and engineered features.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config=None, df=None):
        super().__init__()
        if df is None:
            raise ValueError("DataFrame 'df' must be provided to TradingEnv.")

        self.cfg = config or {}
        self.df = df.copy()

        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0

        self.features_cfg = self.cfg.get("features", {})
        self.actions_cfg = self.cfg.get("actions", {})
        self.rewards_cfg = self.cfg.get("rewards", {})
        self.training_cfg = self.cfg.get("training", {})

        self.allow_long = self.actions_cfg.get("ALLOW_LONG", True)
        self.allow_short = self.actions_cfg.get("ALLOW_SHORT", False)

        self.obs_window = self.features_cfg.get("OBS_WINDOW", 20)

        self.debug = self.cfg.get("DEBUG", False)
        self.trade_log = []
        self.position_duration = 0

        # --- NEW: Determine the lowest timeframe from config ---
        self.lowest_timeframe = self._determine_lowest_timeframe()

        self._identify_data_columns()  # This will now use self.lowest_timeframe

        self.obs_buffer = deque(maxlen=self.obs_window)

        os.makedirs("logs", exist_ok=True)

        self._init_spaces()

        self.balance = 10000.0
        self.initial_balance = self.balance
        self.shares = 0
        self.total_profit = 0.0
        self.rewards_history = []

    def seed(self, seed=None):
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def _determine_lowest_timeframe(self) -> str:
        """
        Infers the lowest timeframe from the feature group keys in config.yaml.
        Assumes keys like 'FEATURES_1MIN', 'FEATURES_5MIN', 'FEATURES_DAILY'.
        """
        timeframe_map = {
            "MIN": 0,  # Shortest duration, highest priority
            "HOUR": 1,
            "DAY": 2,
            "WEEK": 3,
            "MONTH": 4,
            # Add more granularities as needed
        }

        configured_timeframes = []
        for key in self.features_cfg.keys():
            if key.startswith("FEATURES_") and key != "FEATURES_RAW":
                # Extract '1MIN', '5MIN', 'DAILY'
                timeframe_str = key.replace("FEATURES_", "")
                configured_timeframes.append(timeframe_str)

        if not configured_timeframes:
            logger.warning(
                "No time-based feature groups found (e.g., FEATURES_1MIN). Cannot infer lowest timeframe."
            )
            return (
                ""  # Return empty string or raise error, depending on desired behavior
            )

        lowest_priority = float("inf")
        lowest_tf_name = ""

        # Find the timeframe with the highest priority (lowest number in timeframe_map)
        for tf_str in configured_timeframes:
            # Attempt to parse numerical part and unit (e.g., '1MIN' -> 1, 'MIN')
            for unit, priority in timeframe_map.items():
                if unit in tf_str:
                    try:
                        num = int(tf_str.replace(unit, ""))
                        # Assign a combined priority (lower num + lower unit priority is better)
                        current_priority = (
                            num + priority * 1000
                        )  # Scale unit priority to make it dominant
                        if current_priority < lowest_priority:
                            lowest_priority = current_priority
                            lowest_tf_name = tf_str.lower()  # e.g., '1min'
                        break  # Found a match for this tf_str, move to next configured_timeframe
                    except ValueError:
                        # Case like 'DAILY' where no number is present, handle by unit priority
                        current_priority = priority * 1000  # Only unit priority
                        if current_priority < lowest_priority:
                            lowest_priority = current_priority
                            lowest_tf_name = tf_str.lower()
                        break

        if not lowest_tf_name:
            logger.warning(
                "Could not determine lowest timeframe from config feature groups. Defaulting to '1min'."
            )
            return "1min"  # Fallback if parsing fails

        logger.info(f"Inferred lowest timeframe: {lowest_tf_name}")
        return lowest_tf_name

    def _identify_data_columns(self):
        """
        Dynamically identifies the raw price column (for PnL) and the list of
        observation features from the DataFrame based on config.yaml.
        """
        all_df_columns = self.df.columns.tolist()

        # 1. Identify the raw 'close' price field for PnL calculation
        self.raw_close_price_field = None

        # Construct the expected raw close name using the inferred lowest timeframe
        expected_raw_close_name = (
            f"{self.lowest_timeframe}_close" if self.lowest_timeframe else "close"
        )

        if expected_raw_close_name in all_df_columns:
            self.raw_close_price_field = expected_raw_close_name
            logger.info(
                f"Identified raw close price field (inferred): {self.raw_close_price_field}"
            )
        elif "close" in all_df_columns:  # Fallback to generic 'close'
            self.raw_close_price_field = "close"
            logger.info(f"Inferred raw close price field: {self.raw_close_price_field}")
        elif "Close" in all_df_columns:  # Fallback to 'Close' (capital C)
            self.raw_close_price_field = "Close"
            logger.info(
                f"Inferred raw close price field (capital C): {self.raw_close_price_field}"
            )

        if self.raw_close_price_field is None:
            logger.critical(
                "FATAL: Raw 'close' price column not found in DataFrame for PnL calculation."
            )
            logger.critical(f"Available columns: {all_df_columns}")
            raise ValueError(
                "Raw 'close' price column missing from data. "
                "Ensure config.yaml's features section includes 'close' with normalize: false "
                f"for the lowest timeframe ({self.lowest_timeframe}), "
                "and that generate_data.py correctly saves it."
            )
        else:
            logger.info(
                f"Identified raw close price field: {self.raw_close_price_field}"
            )

        # 2. Identify observation features based on config.yaml's feature definitions
        temp_observation_features = []
        seen_columns = set()

        # Iterate through all feature groups defined in config, including FEATURES_RAW if they are meant for observation
        feature_group_keys = [
            "FEATURES_RAW",
            "FEATURES_1MIN",
            "FEATURES_5MIN",
            "FEATURES_DAILY",
        ]

        for group_key in feature_group_keys:
            if group_key in self.features_cfg:
                timeframe_prefix = group_key.replace("FEATURES_", "").lower()
                # If it's a raw feature group, just use the field name, no timeframe prefix
                if group_key == "FEATURES_RAW":
                    timeframe_prefix = ""

                for feature_def in self.features_cfg[group_key]:
                    field_name = feature_def["field"]
                    feature_type = feature_def.get("type")
                    normalize = feature_def.get("normalize", False)
                    method = feature_def.get("method", "")  # Handle missing 'method'
                    source_name = feature_def.get("source", "")
                    window = feature_def.get("window", "")  # Handle missing 'window'
                    # indicator_name = feature_def.get('field') # field for indicators, already 'field_name'

                    final_col_name = ""

                    if feature_type == "price":
                        if normalize:
                            # e.g., 1min_close_zscore
                            final_col_name = (
                                f"{timeframe_prefix}_{field_name}_{method}"
                                if timeframe_prefix
                                else f"{field_name}_{method}"
                            )
                        else:
                            # Raw prices: e.g., 'open', 'high', 'low', 'close', 'volume' (no prefix if FEATURES_RAW, or if explicitly configured not to have one)
                            final_col_name = (
                                f"{timeframe_prefix}_{field_name}"
                                if timeframe_prefix
                                else field_name
                            )
                    elif feature_type == "indicator":
                        # e.g., 5min_close_ema_10_zscore, 5min_vwap_zscore
                        base_col_name_parts = (
                            [timeframe_prefix] if timeframe_prefix else []
                        )
                        if source_name:
                            base_col_name_parts.append(
                                source_name
                            )  # source like 'close'
                        base_col_name_parts.append(
                            field_name
                        )  # indicator name like 'ema' or 'vwap'

                        if window:
                            base_col_name_parts.append(str(window))

                        base_col_name = "_".join(base_col_name_parts)

                        if (
                            normalize and method
                        ):  # Indicators are usually normalized with a method
                            final_col_name = f"{base_col_name}_{method}"
                        else:  # Fallback for non-normalized indicators (less common but possible)
                            final_col_name = base_col_name

                    # Add to temporary list if not a duplicate and exists in DataFrame
                    if (
                        final_col_name
                        and final_col_name not in seen_columns
                        and final_col_name in all_df_columns
                    ):
                        temp_observation_features.append(final_col_name)
                        seen_columns.add(final_col_name)

        # Final cleanup: Exclude 'timestamp' and 'date' from observation features
        self.observation_features = [
            col for col in temp_observation_features if col not in ["timestamp", "date"]
        ]

        if not self.observation_features:
            logger.critical(
                "FATAL: No valid observation features identified based on config and DataFrame columns."
            )
            raise ValueError(
                "No valid observation features found. Check config.yaml 'features' and generate_data.py output naming."
            )

        # Final check to ensure all identified observation features actually exist in the DataFrame
        missing_obs_features_in_df = [
            f for f in self.observation_features if f not in all_df_columns
        ]
        if missing_obs_features_in_df:
            logger.critical(
                f"FATAL: Identified observation features are missing from DataFrame: {missing_obs_features_in_df}"
            )
            logger.critical(f"Available columns in DF: {all_df_columns}")
            raise KeyError(
                f"DataFrame is missing expected observation feature columns: {missing_obs_features_in_df}. "
                "Ensure generate_data.py is creating these features correctly based on config."
            )

        logger.info(
            f"[OBS SPACE] Features identified from data: {self.observation_features}"
        )
        logger.info(f"[OBS SPACE] Feature count: {len(self.observation_features)}")
        logger.info(f"[OBS SPACE] Observation window: {self.obs_window}")
        # Added sample feature logging for verification
        logger.info(f"[OBS SPACE] Sample features: {self.observation_features[:5]}")

        self.feature_count = len(self.observation_features)
        self.final_obs_shape = (self.obs_window * self.feature_count,)

    def _init_spaces(self):
        """Initializes the observation and action spaces based on identified features."""
        feature_dim = len(self.observation_features)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_dim * self.obs_window,),
            dtype=np.float32,
        )

        if self.debug:
            logger.info(f"[OBS SPACE] Features selected: {self.observation_features}")
            logger.info(f"[OBS SPACE] Feature count: {feature_dim}")
            logger.info(f"[OBS SPACE] Observation window: {self.obs_window}")
            logger.info(f"[OBS SPACE] Final shape: {self.observation_space.shape}")
            logger.info(
                f"Final observation shape: ({self.obs_window}, {self.feature_count})"
            )

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)

        max_start_step = len(self.df) - 1
        if max_start_step < self.obs_window:
            raise ValueError(
                f"[ERROR] Not enough data to support obs_window={self.obs_window}. "
                f"Data rows: {len(self.df)}. Required minimum rows: {self.obs_window + 1}"
            )

        if self.training_cfg.get("RANDOM_START", False):
            self.current_step = np.random.randint(self.obs_window - 1, max_start_step)
        else:
            self.current_step = self.obs_window - 1

        self.position = 0
        self.entry_price = 0.0
        self.balance = self.initial_balance
        self.shares = 0
        self.total_profit = 0.0
        self.rewards_history = []
        self.trade_log.clear()
        self.position_duration = 0

        self.obs_buffer.clear()
        for i in range(self.obs_window):
            step_to_read = self.current_step - self.obs_window + 1 + i
            if step_to_read < 0 or step_to_read >= len(self.df):
                logger.warning(
                    f"Attempted to read out-of-bounds step {step_to_read} for buffer initialization. Padding with zeros."
                )
                self.obs_buffer.append(np.zeros(self.feature_count, dtype=np.float32))
            else:
                self.obs_buffer.append(
                    self.df.loc[step_to_read, self.observation_features].values.astype(
                        np.float32
                    )
                )

        if self.debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(
                    f"[RESET] Environment reset. Start step: {self.current_step}. Initial obs_buffer size: {len(self.obs_buffer)}\n"
                )

        observation = self._get_observation()
        info = {"current_step": self.current_step, "price": self._get_price()}
        return observation, info

    def step(self, action: int):
        """
        Performs one step in the environment based on the agent's action.
        """
        prev_raw_price = self._get_price()

        take_action(self, action, prev_raw_price)

        self.current_step += 1

        terminated = self.current_step >= len(self.df)
        truncated = False

        reward = calculate_reward(self)

        if self.current_step < len(self.df):
            obs_features_current_step = self.df.loc[
                self.current_step, self.observation_features
            ].values.astype(np.float32)
            self.obs_buffer.append(obs_features_current_step)
        else:
            self.obs_buffer.append(np.zeros(self.feature_count, dtype=np.float32))

        if self.position != 0:
            self.position_duration += 1

        info = {
            "step": self.current_step,
            "reward_this_step": reward,
            "position": self.position,
            "entry_price": self.entry_price,
            "current_raw_price": self._get_price(),
            "position_duration": self.position_duration,
            "trade_log": list(self.trade_log),
        }

        if self.debug:
            with open("logs/env_debug.log", "a") as f:
                f.write(
                    f"[STEP] step={self.current_step}, action={action}, reward={reward:.5f}, pos={self.position}, price={self._get_price():.2f}, duration={self.position_duration}\n"
                )

        next_observation = self._get_observation()

        return next_observation, reward, terminated, truncated, info

    def _get_price(self):
        """Returns the raw 'close' price for the current step. Used for PnL calculations."""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1][self.raw_close_price_field]
        return self.df.iloc[self.current_step][self.raw_close_price_field]

    def _get_observation(self):
        """
        Returns the flattened observation array from the observation buffer.
        This is the actual input to the RL agent.
        """
        obs = np.array(list(self.obs_buffer), dtype=np.float32).flatten()
        if np.any(np.isnan(obs)):
            logger.error(
                "[NaN DETECTED] Observation contains NaN values. Check feature preprocessing or data loading."
            )
            raise ValueError("Observation contains NaN values.")
        return obs

    def render(self, mode="human"):
        """Renders the environment."""
        pass

    def close(self):
        """Cleans up resources."""
        pass
