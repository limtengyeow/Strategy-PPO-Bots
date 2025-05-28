
def take_action(env, action, price):
    if action == 1:  # Buy
        if env.allow_long and env.position == 0:
            env.position = 1
            env.entry_price = price
            env.position_duration = 0
            env.trade_log.append({"action": "long", "entry": price, "step": env.current_step})
        elif env.allow_short and env.position == 1:
            pnl = price - env.entry_price
            env.trade_log.append({"action": "exit_long", "exit": price, "pnl": pnl, "step": env.current_step})
            env.position = -1
            env.entry_price = price
            env.position_duration = 0
            env.trade_log.append({"action": "short", "entry": price, "step": env.current_step})
    elif action == 2:  # Sell
        if env.allow_short and env.position == 0:
            env.position = -1
            env.entry_price = price
            env.position_duration = 0
            env.trade_log.append({"action": "short", "entry": price, "step": env.current_step})
        elif env.allow_long and env.position == -1:
            pnl = env.entry_price - price
            env.trade_log.append({"action": "exit_short", "exit": price, "pnl": pnl, "step": env.current_step})
            env.position = 1
            env.entry_price = price
            env.position_duration = 0
            env.trade_log.append({"action": "long", "entry": price, "step": env.current_step})
        elif env.position != 0:
            pnl = (price - env.entry_price) if env.position == 1 else (env.entry_price - price)
            env.trade_log.append({"action": "exit", "exit": price, "pnl": pnl, "step": env.current_step})
            env.position = 0
            env.entry_price = 0.0
            env.position_duration = 0
