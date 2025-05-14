from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import mlflow
import json

class PerformanceMetricsCallback(BaseCallback):
    def __init__(self, config_path="config/config.json", verbose=0):
        super().__init__(verbose)
        with open(config_path) as f:
            cfg = json.load(f)
        self.metrics_to_track = set(cfg.get("metrics", {}).get("track", []))
        self.episode_rewards = []
        self.episode_trades = []

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "trades" in info:
            self.episode_trades.extend(info["trades"])
        if self.locals["dones"][0]:
            reward = self.locals["rewards"][0]
            self.episode_rewards.append(reward)
            metrics = self.compute_metrics(pd.DataFrame(self.episode_trades))
            self.log_metrics(metrics)
            self.episode_trades = []
        return True

    def compute_metrics(self, trades_df):
        if trades_df.empty:
            return {}
        results = {}
        if "win_rate" in self.metrics_to_track:
            wins = trades_df[trades_df.pnl > 0]
            results["win_rate"] = len(wins) / len(trades_df) * 100
        if "profit_factor" in self.metrics_to_track:
            wins = trades_df[trades_df.pnl > 0]
            losses = trades_df[trades_df.pnl <= 0]
            gross_profit = wins.pnl.sum()
            gross_loss = abs(losses.pnl.sum())
            results["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf
        if "expectancy" in self.metrics_to_track:
            results["expectancy"] = trades_df.pnl.mean()
        if "max_drawdown" in self.metrics_to_track and "equity" in trades_df.columns:
            results["max_drawdown"] = (trades_df.equity.cummax() - trades_df.equity).max()
        if "sharpe_ratio" in self.metrics_to_track:
            results["sharpe_ratio"] = trades_df.pnl.mean() / trades_df.pnl.std() if trades_df.pnl.std() != 0 else 0
        if "avg_trade_duration" in self.metrics_to_track:
            results["avg_trade_duration"] = trades_df.duration.mean()
        if "r_multiple" in self.metrics_to_track and "risk" in trades_df.columns:
            results["r_multiple"] = (trades_df.pnl / trades_df.risk).mean()
        return results

    def log_metrics(self, metrics):
        for k, v in metrics.items():
            self.logger.record(f"custom_metrics/{k}", v)
            mlflow.log_metric(k, v)
        if self.verbose > 0:
            print(f"[Eval] {metrics}")
