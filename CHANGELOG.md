# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-30

### Added
- Initial release of Intraday EMA-RSI Stock Backtester
- Full strategy with LONG and SHORT trades (`new_strategy.py`)
- LONG only strategy (`new_strategy_only_long.py`)
- Multi-timeframe indicator support (10-min EMA, 1-hour EMA, RSI)
- Top 10 stock selection by turnover at 9:25 AM
- Risk management with 0.5% stop loss and 2% target
- Trailing stop (0.75% after 0.5% profit)
- Position sizing based on risk per trade
- Daily and aggregate trade logging
- Performance metrics and summary reports
- Command-line interface with date range support
- Sample trade logs from August 2025 backtest

### Technical Details
- Python 3.9+ support
- Pandas-based data processing
- Multi-processing support for parallel backtesting
- Comprehensive documentation
