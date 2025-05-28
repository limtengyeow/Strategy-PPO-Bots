def calculate_reward(env):
    if env.position == 0:
        return 0.0

    current_price = env._get_price()

    # Long position reward: percentage gain/loss from entry
    if env.position == 1:
        reward = (current_price - env.entry_price) / env.entry_price

    # Short position reward: reversed gain/loss
    elif env.position == -1:
        reward = (env.entry_price - current_price) / env.entry_price

    else:
        reward = 0.0

    return reward
