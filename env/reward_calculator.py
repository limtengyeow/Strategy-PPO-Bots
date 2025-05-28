from pathlib import Path


def calculate_reward(env):
    rewards_cfg = env.rewards_cfg
    components = rewards_cfg.get("REWARD_COMPONENTS", [])
    reward = 0.0

    current_price = env._get_price()

    # Calculate PnL reward
    if any(c.get("type") == "pnl" for c in components) and env.position != 0:
        pnl_component = next(c for c in components if c.get("type") == "pnl")
        pnl = (
            (current_price - env.entry_price) / env.entry_price
            if env.position == 1
            else (env.entry_price - current_price) / env.entry_price
        )
        reward += pnl * pnl_component.get("scale", 1.0)

    # Cut Loss Penalty
    for c in components:
        if c.get("type") == "cut_loss" and env.position != 0:
            threshold = c.get("threshold", -0.01)
            penalty_ratio = c.get("penalty_ratio", 1.0)
            unrealized_pnl = (
                (current_price - env.entry_price) / env.entry_price
                if env.position == 1
                else (env.entry_price - current_price) / env.entry_price
            )
            if unrealized_pnl < threshold:
                reward += threshold * penalty_ratio

    # Overtrade Penalty
    for c in components:
        if c.get("type") == "overtrade_penalty" and env.position != 0:
            min_duration = c.get("min_duration", 5)
            penalty_ratio = c.get("penalty_ratio", 1.0)
            if env.position_duration < min_duration:
                reward -= penalty_ratio * 0.001  # Small penalty per step

    # Trend Bonus
    for c in components:
        if c.get("type") == "trend_bonus" and env.position != 0:
            bonus = c.get("bonus_per_step", 0.001)
            reward += bonus

    # Breakout Reward
    for c in components:
        if c.get("type") == "breakout_reward" and env.position != 0:
            window = c.get("window", 50)
            bonus = c.get("bonus", 0.005)
            start = max(env.current_step - window, 0)
            recent_prices = env.df.iloc[start : env.current_step][env.price_field]
            if env.position == 1 and current_price > recent_prices.max():
                reward += bonus
            elif env.position == -1 and current_price < recent_prices.min():
                reward += bonus

    # Volatility Penalty
    for c in components:
        if c.get("type") == "volatility_penalty":
            window = c.get("window", 20)
            threshold = c.get("threshold", 0.03)
            penalty = c.get("penalty", 0.002)
            start = max(env.current_step - window, 0)
            recent_prices = env.df.iloc[start : env.current_step][env.price_field]
            if not recent_prices.empty:
                vol = recent_prices.pct_change().std()
                if vol > threshold:
                    reward -= penalty

    # Profit Target Bonus
    for c in components:
        if c.get("type") == "profit_target_bonus" and env.position != 0:
            target_pct = c.get("target_pct", 0.05)
            bonus = c.get("bonus", 0.01)
            unrealized_pnl = (
                (current_price - env.entry_price) / env.entry_price
                if env.position == 1
                else (env.entry_price - current_price) / env.entry_price
            )
            if unrealized_pnl >= target_pct:
                reward += bonus

    # Time Penalty
    for c in components:
        if c.get("type") == "time_penalty" and env.position != 0:
            penalty_per_step = c.get("penalty_per_step", 0.0005)
            reward -= penalty_per_step

    # Apply Reward Scaling
    reward *= rewards_cfg.get("REWARD_SCALE", 1.0)

    # Apply Clipping
    clip = rewards_cfg.get("CLIP_REWARD", None)
    if clip is not None:
        reward = max(min(reward, clip), -clip)

    # Normalize Reward
    if rewards_cfg.get("NORM_REWARD", False):
        reward = (reward - env.df[env.price_field].mean()) / env.df[
            env.price_field
        ].std()

    # Debug Logging (Jupyter-compatible)
    if rewards_cfg.get("DEBUG_REWARDS", False):
        log_dir = Path.cwd() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "reward_debug.log"
        log_entry = (
            f"Step: {env.current_step}, "
            f"Position: {env.position}, "
            f"Price: {current_price:.2f}, "
            f"Entry Price: {env.entry_price:.2f}, "
            f"Reward: {reward:.5f}\n"
        )
        with open(log_file, "a") as f:
            f.write(log_entry)

    return reward
