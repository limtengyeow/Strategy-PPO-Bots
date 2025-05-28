import json
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

# === Setup metrics log file ===
LOG_FILE = Path.cwd() / "logs" / "metrics_debug.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_metrics_debug(message, debug=True):
    if debug:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        with LOG_FILE.open("a") as f:
            f.write(log_line)
            f.flush()  # Ensure logs are written immediately for terminal and Jupyter


def format_metrics(metrics):
    if not metrics:
        return "No trades yet."
    return (
        f"- Total Trades: {metrics.get('number_of_trades', 0)}\n"
        f"- Win Rate: {metrics.get('win_rate', 0):.2f}%\n"
        f"- Avg Win: {metrics.get('avg_%_win', 0):+.2f}%\n"
        f"- Avg Loss: {metrics.get('avg_%_loss', 0):.2f}%"
    )


class PerformanceMetricsCallback(BaseCallback):
    def __init__(self, config_path="config.json", verbose=1):
        super().__init__(verbose)
        with open(config_path) as f:
            cfg = json.load(f)
        self.metrics_to_track = set(cfg.get("metrics", {}).get("track", []))
        self.use_mlflow = cfg.get("logging", {}).get("use_mlflow", False)
        self.debug = cfg.get("DEBUG", False)
        self.episode_trades = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[0]
        done = self.locals.get("dones", [False])[0]

        if "trades" in info:
            self.episode_trades.extend(info["trades"])

        if done:
            metrics = self.compute_metrics(pd.DataFrame(self.episode_trades))
            self.log_metrics(metrics)
            if self.verbose > 0:
                print(f"[Episode Summary]\n{format_metrics(metrics)}", flush=True)
            log_metrics_debug(
                f"[Episode Summary]\n{format_metrics(metrics)}", self.debug
            )
            self.episode_trades = []

        return True

    def _on_rollout_end(self):
        if self.episode_trades:
            metrics = self.compute_metrics(pd.DataFrame(self.episode_trades))
            if self.verbose > 0:
                print(f"[Rollout Summary]\n{format_metrics(metrics)}", flush=True)
            log_metrics_debug(
                f"[Rollout Summary]\n{format_metrics(metrics)}", self.debug
            )
        else:
            print("[Rollout Summary] No trades yet in this episode.", flush=True)
            log_metrics_debug(
                "[Rollout Summary] No trades yet in this episode.", self.debug
            )

    def compute_metrics(self, trades_df):
        if trades_df.empty:
            return {}

        results = {}
        num_trades = len(trades_df)
        wins = trades_df[trades_df.pnl > 0]
        losses = trades_df[trades_df.pnl <= 0]

        results["number_of_trades"] = num_trades
        results["win_rate"] = (len(wins) / num_trades * 100) if num_trades > 0 else 0

        if len(wins) > 0:
            results["avg_%_win"] = wins["pnl"].mean() * 100
        else:
            results["avg_%_win"] = 0

        if len(losses) > 0:
            results["avg_%_loss"] = losses["pnl"].mean() * 100
        else:
            results["avg_%_loss"] = 0

        return results

    def log_metrics(self, metrics):
        for k, v in metrics.items():
            self.logger.record(f"custom_metrics/{k}", v)
            if self.use_mlflow:
                k_clean = (
                    k.replace("%", "pct")
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                )
                mlflow.log_metric(k_clean, v)
