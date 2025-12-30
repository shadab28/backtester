"""
Intraday EMA-RSI Stock Backtester

A Python-based backtesting framework for intraday trading strategies
on NSE stocks using EMA crossovers and RSI indicators.
"""

__version__ = "1.0.0"
__author__ = "Shadab"
__email__ = "shadab@example.com"

from pathlib import Path

# Package root directory
PACKAGE_DIR = Path(__file__).parent

# Default configuration
DEFAULT_CONFIG = {
    "base_capital": 10_00_000,  # 10 Lac
    "risk_per_trade_pct": 0.005,  # 0.5%
    "stop_loss_pct": 0.005,  # 0.5%
    "profit_target_pct": 0.02,  # 2%
    "trail_start_pct": 0.005,  # 0.5%
    "trail_step_pct": 0.0075,  # 0.75%
}
