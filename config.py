"""
Configuration module for the backtester.

This module contains all configurable parameters for the trading strategy.
Modify these values to customize the backtesting behavior.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TradingConfig:
    """Trading strategy configuration parameters."""
    
    # Capital settings
    base_capital: float = 10_00_000  # 10 Lac (Indian numbering)
    
    # Risk management
    risk_per_trade_pct: float = 0.005  # 0.5% of current equity
    stop_loss_pct: float = 0.005  # 0.5% below entry
    profit_target_pct: float = 0.02  # 2% above entry
    
    # Trailing stop
    trail_start_pct: float = 0.005  # Start trailing after 0.5% profit
    trail_step_pct: float = 0.0075  # Trail by 0.75%
    
    # Indicator periods
    ema_fast_period: int = 3  # EMA(3) on 10-min
    ema_slow_period: int = 10  # EMA(10) on 10-min
    ema_trend_period: int = 50  # EMA(50) on 1-hour
    rsi_period: int = 14  # RSI(14) on 10-min
    
    # Entry thresholds
    rsi_long_threshold: float = 60  # RSI > 60 for LONG
    rsi_short_threshold: float = 30  # RSI < 30 for SHORT
    
    # Time settings
    market_open: str = "09:15"
    selection_time: str = "09:25"
    market_close: str = "15:29"
    
    # Stock selection
    top_n_stocks: int = 10  # Select top N by turnover
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert 0 < self.risk_per_trade_pct <= 0.1, "Risk per trade should be between 0% and 10%"
        assert 0 < self.stop_loss_pct <= 0.1, "Stop loss should be between 0% and 10%"
        assert 0 < self.profit_target_pct <= 0.5, "Profit target should be between 0% and 50%"
        assert self.base_capital > 0, "Base capital must be positive"


@dataclass  
class PathConfig:
    """File and directory path configuration."""
    
    base_dir: Path = None
    
    def __post_init__(self):
        if self.base_dir is None:
            self.base_dir = Path(__file__).parent
    
    @property
    def data_dir(self) -> Path:
        """Directory containing stock data CSV files."""
        return self.base_dir / "stock_data"
    
    @property
    def trade_log_dir_long(self) -> Path:
        """Directory for LONG only strategy trade logs."""
        return self.base_dir / "trade_logs_only_long"
    
    @property
    def trade_log_dir_full(self) -> Path:
        """Directory for full strategy trade logs."""
        return self.base_dir / "trade_logs_new_strategy"


# Default instances
DEFAULT_TRADING_CONFIG = TradingConfig()
DEFAULT_PATH_CONFIG = PathConfig()
