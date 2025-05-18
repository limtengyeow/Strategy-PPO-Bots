def calculate_reward(env):
    if env.position == 0:
        return 0.0

    cfg = env.rewards_cfg
    reward = 0.0

    current_price = env._get_price()
    entry_price = env.entry_price
    position = env.position
    duration = env.position_duration

    # === Safe PnL Calculation ===
    if entry_price == 0:
        pnl = 0.0
    elif position == 1:
        pnl = (current_price - entry_price) / entry_price
    elif position == -1:
        pnl = (entry_price - current_price) / entry_price
    else:
        pnl = 0.0

    # === Reward Components ===
    for component in cfg.get("REWARD_COMPONENTS", []):
        ctype = component.get("type")

        if ctype == "pnl":
            scale = component.get("scale", 100.0)
            reward += pnl * scale

        elif ctype == "cut_loss":
            threshold = component.get("threshold", -0.01)
            ratio = component.get("penalty_ratio", 100.0)
            if pnl < threshold:
                reward -= abs(pnl) * ratio

        elif ctype == "overtrade_penalty":
            min_duration = component.get("min_duration", 5)
            ratio = component.get("penalty_ratio", 20.0)
            if position != 0 and duration < min_duration:
                reward -= ratio

        elif ctype == "hold_bonus":
            bonus = component.get("reward_per_step", 0.1)
            if position != 0:
                reward += bonus

        else:
            if cfg.get("DEBUG_REWARDS", False):
                print(f"[REWARD DEBUG] Unknown reward type: {ctype}")

    # === Final Adjustments ===
    if cfg.get("CLIP_REWARD") is not None:
        clip = cfg["CLIP_REWARD"]
        reward = max(min(reward, clip), -clip)

    if cfg.get("NORM_REWARD", False):
        reward = np.tanh(reward)

    if cfg.get("DEBUG_REWARDS", False):
        print(f"[REWARD DEBUG] PnL={pnl:.5f}, duration={duration}, total_reward={reward:.5f}")

    return reward
