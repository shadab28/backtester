# Intraday EMA-RSI Stock Backtester

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready Python backtesting framework for intraday trading strategies on NSE (National Stock Exchange of India) stocks using EMA crossovers and RSI indicators.

## 🚀 Features

- **Multi-timeframe Analysis**: EMA on 10-minute and 1-hour timeframes
- **Smart Stock Selection**: Top 10 stocks by turnover at market open
- **Risk Management**: Position sizing based on risk per trade
- **Trailing Stops**: Lock in profits with dynamic trailing
- **Detailed Logging**: Trade-by-trade and daily summary reports
- **Command-line Interface**: Easy date range selection
- **Production Ready**: Type hints, docstrings, and clean code

## 📊 Strategy Overview

### Stock Selection
At 9:25 AM, the strategy selects the **top 10 stocks by turnover** (Volume × Close) calculated from the first 10 minutes of trading (09:15-09:24).

### Technical Indicators

| Indicator | Timeframe | Period | Purpose |
|-----------|-----------|--------|---------|
| EMA(3) | 10-minute | 3 | Fast signal |
| EMA(10) | 10-minute | 10 | Slow signal |
| EMA(50) | 1-hour | 50 | Trend filter |
| RSI(14) | 10-minute | 14 | Momentum |

### Entry Rules

#### 📈 LONG Entry
```
EMA(3) > EMA(10)  AND  RSI(14) > 60  AND  Close > EMA(50)
→ Buy at HIGH of entry candle
```

#### 📉 SHORT Entry (Full strategy only)
```
EMA(3) < EMA(10)  AND  RSI(14) < 30  AND  Close < EMA(50)
→ Sell at LOW of last 5 candles
```

### Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Capital | ₹10,00,000 | Starting capital (10 Lac) |
| Risk per Trade | 0.5% | Maximum loss per trade |
| Stop Loss | 0.5% | Distance from entry |
| Profit Target | 2% | Take profit level |
| Trailing Start | 0.5% | Activate trailing after this profit |
| Trailing Step | 0.75% | Trail distance from high |

### Exit Rules

1. **Stop Loss** - Exit when price breaches SL level
2. **Target** - Exit when price reaches 2% profit
3. **Trailing Stop** - Dynamic exit after 0.5% profit
4. **End of Day** - All positions closed at 15:29

## 🛠️ Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/shadab28/backtester.git
cd backtester

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using pip (development mode)

```bash
pip install -e .
```

## 📁 Data Format

Place your NSE minute data in the `stock_data/` folder with this naming convention:

```
stock_data/
├── dataNSE_20250801.csv
├── dataNSE_20250804.csv
├── dataNSE_20250805.csv
└── ...
```

### CSV Format

```csv
ticker,time,open,high,low,close,volume
RELIANCE,2025-08-01 09:15:00,2450.00,2455.50,2448.00,2453.25,125000
RELIANCE,2025-08-01 09:16:00,2453.25,2458.00,2452.00,2456.75,98500
...
```

**Required columns:**
- `ticker` - Stock symbol
- `time` - Datetime (YYYY-MM-DD HH:MM:SS)
- `open`, `high`, `low`, `close` - OHLC prices
- `volume` - Trading volume

## 💻 Usage

### Basic Usage

```bash
# Run LONG only strategy for all available dates
python new_strategy_only_long.py --all

# Run for specific date range
python new_strategy_only_long.py --start-date 2025-08-01 --end-date 2025-08-31

# Run full LONG+SHORT strategy
python new_strategy.py --all
```

### Command-line Options

| Option | Description |
|--------|-------------|
| `--start-date` | Start date (YYYY-MM-DD) |
| `--end-date` | End date (YYYY-MM-DD) |
| `--all` | Process all available data files |

## 📈 Output

### Directory Structure

```
trade_logs_only_long/
├── trade_log_2025-08-01.csv   # Daily trade details
├── trade_log_2025-08-04.csv
├── ...
├── all_trades.csv              # Combined trades
└── daily_summary.csv           # Performance summary
```

### Trade Log Format

```csv
time,symbol,side,entry_price,size,status,exit_time,exit_price,pnl,reason
2025-08-06 09:25:00,NCC,LONG,220.35,4224,CLOSED,2025-08-06 09:26:00,219.25,-4653.79,SL
2025-08-06 12:00:00,WAAREEENER,LONG,3216.00,289,CLOSED,2025-08-06 12:34:00,3222.95,2007.18,TRAIL
```

### Daily Summary

```csv
date,num_trades,total_pnl,return_pct,win_rate,eod_equity
2025-08-01,4,-15884.87,-1.59,0.00,984115.13
2025-08-06,7,-7403.69,-0.80,42.86,923493.42
```

## 📊 Sample Results (August 2025)

### LONG Only Strategy

| Metric | Value |
|--------|-------|
| Trading Days | 17 |
| Total Trades | 107 |
| Win Rate | 27.10% |
| Total P&L | -₹1,58,747 |
| Return | -15.87% |
| Final Equity | ₹8,41,253 |

### Full Strategy (LONG + SHORT)

| Metric | Value |
|--------|-------|
| Trading Days | 17 |
| Total Trades | 170 |
| Win Rate | 18.24% |
| Total P&L | -₹3,65,434 |
| Return | -36.54% |

> **Note:** Past performance does not guarantee future results. Always backtest on out-of-sample data.

## 🏗️ Project Structure

```
backtester/
├── new_strategy_only_long.py   # LONG only strategy
├── new_strategy.py             # Full LONG+SHORT strategy
├── config.py                   # Configuration parameters
├── intraday.py                 # Legacy backtester
├── final_stats.py              # Statistics utilities
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package configuration
├── README.md                   # This file
├── LICENSE                     # MIT License
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guide
├── stock_data/                 # Input data (not in repo)
├── trade_logs_only_long/       # LONG strategy output
└── trade_logs_new_strategy/    # Full strategy output
```

## ⚙️ Configuration

Edit `config.py` to customize strategy parameters:

```python
@dataclass
class TradingConfig:
    base_capital: float = 10_00_000  # Starting capital
    risk_per_trade_pct: float = 0.005  # 0.5%
    stop_loss_pct: float = 0.005  # 0.5%
    profit_target_pct: float = 0.02  # 2%
    trail_start_pct: float = 0.005  # 0.5%
    trail_step_pct: float = 0.0075  # 0.75%
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading in the stock market involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consult with a qualified financial advisor before making investment decisions.

## 📧 Contact

- **Author:** Shadab
- **GitHub:** [@shadab28](https://github.com/shadab28)
- **Repository:** [backtester](https://github.com/shadab28/backtester)

---

⭐ **Star this repo** if you find it helpful!
