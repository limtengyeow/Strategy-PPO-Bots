
def calculate_reward(env):
    if env.position == 0:
        return 0.0
    current_price = env._get_price()
    if env.position == 1:
        return (current_price - env.entry_price) / env.entry_price
    elif env.position == -1:
        return (env.entry_price - current_price) / env.entry_price
    return 0.0
