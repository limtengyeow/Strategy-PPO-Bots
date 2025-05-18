def calculate_reward(env):
    if env.position == 0:
        return 0.0

    cfg = env.rewards_cfg
    pnl = 0.0
    reward = 0.0

    current_price = env._get_price()
    entry_price = env.entry_price
    position = env.position
    duration = env.position_duration

    # === PnL component
    for component in cfg.get("REWARD_COMPONENTS", []):
        ctype = component.get("type")

        if ctype == "pnl":
            scale = component.get("scale", 1.0)
            if position == 1:
                pnl = (current_price - entry_price) / entry_price
            elif position == -1:
                pnl = (entry_price - current_price) / entry_price
            reward += pnl * scale

        elif ctype == "cut_loss":
            threshold = component.get("threshold", -0.01)
            ratio = component.get("penalty_ratio", 1.0)
            if pnl < threshold:
                reward -= abs(pnl) * ratio

        elif ctype == "overtrade_penalty":
            min_duration = component.get("min_duration", 5)
            ratio = component.get("penalty_ratio", 1.0)
            if duration < min_duration:
                reward -= ratio

    # === Final adjustments
    if cfg.get("CLIP_REWARD") is not None:
        clip = cfg["CLIP_REWARD"]
        reward = max(min(reward, clip), -clip)

    if cfg.get("NORM_REWARD", False):
        reward = np.tanh(reward)

    if cfg.get("DEBUG_REWARDS", False):
        print(f"[REWARD DEBUG] PnL={pnl:.5f}, duration={duration}, total_reward={reward:.5f}")

    return reward
