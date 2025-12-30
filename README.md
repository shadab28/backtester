# Intraday EMA-RSI Stock Backtester

A Python-based backtesting framework for intraday trading strategies on NSE stocks using EMA crossovers and RSI indicators.

## Strategy Overview

### Stock Selection
- At 9:25 AM, select **top 10 stocks by turnover** (Volume × Close) from the first 10 minutes (09:15-09:24)

### Indicators
| Indicator | Timeframe | Period |
|-----------|-----------|--------|
| EMA(3) | 10-minute | 3 periods |
| EMA(10) | 10-minute | 10 periods |
| EMA(50) | 1-hour | 50 periods |
| RSI(14) | 10-minute | 14 periods |

### Entry Rules

**LONG Entry:**
- EMA(3) > EMA(10)
- RSI(14) > 60
- Close > EMA(50)
- Entry at **HIGH** of entry candle

**SHORT Entry:** (in full strategy only)
- EMA(3) < EMA(10)
- RSI(14) < 30
- Close < EMA(50)
- Entry at **LOW** of last 5 candles

### Risk Management
| Parameter | Value |
|-----------|-------|
| Base Capital | ₹10,00,000 (10 Lac) |
| Risk per Trade | 0.5% of current equity |
| Stop Loss | 0.5% |
| Profit Target | 2% |
| Trailing Start | After 0.5% profit |
| Trailing Step | 0.75% |

### Exit Rules
1. **Stop Loss** - Exit at SL price if LOW breaches SL
2. **Target** - Exit at target price if HIGH reaches target
3. **Trailing Stop** - Activated after 0.5% profit, trails by 0.75%
4. **End of Day** - All positions closed at 15:29

## Files

| File | Description |
|------|-------------|
| `new_strategy.py` | Full strategy with LONG + SHORT trades |
| `new_strategy_only_long.py` | LONG only strategy |
| `intraday.py` | Original/legacy backtester |
| `final_stats.py` | Statistics and analysis utilities |

## Data Format

Place NSE minute data in `stock_data/` folder with naming convention:
```
dataNSE_YYYYMMDD.csv
```

CSV columns: `ticker, time, open, high, low, close, volume`

## Usage

```bash
# Run LONG only strategy
python new_strategy_only_long.py

# Run full LONG+SHORT strategy  
python new_strategy.py
```

## Output

Trade logs are saved to:
- `trade_logs_only_long/` - For LONG only strategy
- `trade_logs_new_strategy/` - For full strategy

Each run generates:
- Daily trade logs (`trade_log_YYYY-MM-DD.csv`)
- Combined trades (`all_trades.csv`)
- Daily summary (`daily_summary.csv`)

## Requirements

```
pandas
numpy
tqdm
```

## License

MIT License
