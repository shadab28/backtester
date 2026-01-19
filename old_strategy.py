"""
Intraday EMA-RSI Stock Strategy Backtester (New Strategy)

Strategy Rules:
- Base Capital: 10 Lac (10,00,000)
- Stock Selection: At 9:25, select top 10 stocks by turnover (sum of volume*close for first 10 minutes 09:15-09:24)
- Indicators:
    - EMA(3) on 10-minute timeframe
    - EMA(10) on 10-minute timeframe
    - EMA(50) on 1-hour timeframe
    - RSI(14) on 10-minute timeframe

Entry Rules:
- LONG: EMA(3) > EMA(10) AND RSI(14) > 60 AND Close > EMA(50) -> Buy at high of entry candle
- SHORT: EMA(3) < EMA(10) AND RSI(14) < 30 AND Close < EMA(50) -> Sell at low of last 5 1-min candles

Risk Management:
- Stop Loss: 0.5%
- Profit Target: 2%
- Risk per trade: 0.5% of allocated capital per stock
- Trailing: Start continuous trailing by 0.75% after trade is in profit of 0.5%

Usage:
    python new_strategy.py --start-date 2025-08-01
    python new_strategy.py --start-date 2025-08-01 --end-date 2025-08-31
    python new_strategy.py --all
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import warnings
import time
try:
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except Exception:
    HAVE_MATPLOTLIB = False

import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
BASE_CAPITAL = 10_00_000  # 10 Lac (Indian numbering)
RISK_PER_TRADE_PCT = 0.005  # 0.5% of allocated capital
STOP_LOSS_PCT = 0.005  # 0.5%   
PROFIT_TARGET_PCT = 0.02  # 2%
TRAIL_START_PCT = 0.005  # 0.5% profit to start trailing
TRAIL_STEP_PCT = 0.0075  # 0.75% trailing step

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "stock_data"
TRADE_LOG_DIR = SCRIPT_DIR / "trade_logs_new_strategy/old"

# Allow entries during the first N minutes after trade start (inclusive)
# Increased to 30 minutes so indicators (10-min / 1-hour) have time to populate
ENTRY_WINDOW_MINUTES = 30

# In-memory cache for preloaded CSVs: { 'YYYYMMDD': DataFrame }
PRELOAD_CACHE = {}

# Optional Numba acceleration
HAVE_NUMBA = False
try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False

if HAVE_NUMBA:
    # Numba-accelerated EMA (simple loop, handles NaNs by forward-filling initial value)
    @njit
    def ema_numba(arr, span):
        n = arr.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.nan

        if n == 0:
            return out

        alpha = 2.0 / (span + 1.0)

        # find first non-nan
        first_idx = -1
        for i in range(n):
            if not np.isnan(arr[i]):
                first_idx = i
                break

        if first_idx == -1:
            return out

        last = arr[first_idx]
        out[first_idx] = last

        for j in range(first_idx + 1, n):
            if np.isnan(arr[j]):
                out[j] = np.nan
            else:
                last = alpha * arr[j] + (1.0 - alpha) * last
                out[j] = last

        return out

    # Numba-accelerated RSI using Wilder smoothing (alpha = 1/period)
    @njit
    def rsi_numba(arr, period):
        n = arr.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = np.nan

        if n <= period:
            return out

        # compute deltas
        deltas = np.empty(n, dtype=np.float64)
        deltas[0] = 0.0
        for i in range(1, n):
            if np.isnan(arr[i]) or np.isnan(arr[i - 1]):
                deltas[i] = 0.0
            else:
                deltas[i] = arr[i] - arr[i - 1]

        gains = np.zeros(n, dtype=np.float64)
        losses = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            d = deltas[i]
            if d > 0:
                gains[i] = d
                losses[i] = 0.0
            else:
                gains[i] = 0.0
                losses[i] = -d

        # initial average gains/losses (from 1..period)
        sum_gain = 0.0
        sum_loss = 0.0
        count = 0
        for i in range(1, period + 1):
            sum_gain += gains[i]
            sum_loss += losses[i]
            count += 1

        if count == 0:
            return out

        avg_gain = sum_gain / count
        avg_loss = sum_loss / count

        # first valid RSI at index 'period'
        if avg_loss == 0.0:
            out[period] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[period] = 100.0 - (100.0 / (1.0 + rs))

        for i in range(period + 1, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0.0:
                out[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                out[i] = 100.0 - (100.0 / (1.0 + rs))

        return out

# ========== INDICATOR FUNCTIONS ==========

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    # Use numba accelerated implementation when possible for large numpy arrays
    try:
        # If caller passed a pandas Series, convert to numpy for numba
        if HAVE_NUMBA and isinstance(series, (pd.Series, np.ndarray)):
            arr = np.asarray(series)
            ema_arr = ema_numba(arr, span)
            return pd.Series(ema_arr, index=getattr(series, 'index', None))
    except NameError:
        pass

    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    # Try to use numba-accelerated version for performance on large arrays
    try:
        if HAVE_NUMBA and isinstance(series, (pd.Series, np.ndarray)):
            arr = np.asarray(series)
            rsi_arr = rsi_numba(arr, period)
            return pd.Series(rsi_arr, index=getattr(series, 'index', None))
    except NameError:
        pass

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ========== DATA PROCESSING FUNCTIONS ==========

def load_data_for_date(date: datetime.date, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load all stock data for a given date from CSV file"""
    date_str = date.strftime("%Y%m%d")
    # Check preload cache first
    if PRELOAD_CACHE and date_str in PRELOAD_CACHE:
        return PRELOAD_CACHE[date_str]

    file_path = data_dir / f"dataNSE_{date_str}.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path, parse_dates=["time"])
    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
    return df


def preload_all_data(data_dir: Path = DATA_DIR) -> dict:
    """Preload all CSV files in data_dir into memory and return the cache dict.

    Cache key: YYYYMMDD (string), value: pandas DataFrame
    """
    files = sorted(data_dir.glob("dataNSE_*.csv"))
    cache = {}
    for f in tqdm(files, desc="Preloading CSVs", unit="file"):
        date_str = f.stem.replace("dataNSE_", "")
        try:
            df = pd.read_csv(f, parse_dates=["time"])
            df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
            cache[date_str] = df
        except Exception:
            continue

    # update global cache
    PRELOAD_CACHE.clear()
    PRELOAD_CACHE.update(cache)
    return PRELOAD_CACHE


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample minute data to specified timeframe (e.g., '10min', '60min')"""
    if df.empty:
        return df
    
    df_resampled = df.set_index("time").resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna().reset_index()
    
    return df_resampled


def calculate_turnover_top10(df: pd.DataFrame, date: datetime.date) -> list:
    """
    Select top 10 stocks by turnover at 9:25
    Turnover = sum of (volume * close) for first 10 minutes (09:15-09:24)
    """
    start_time = datetime.combine(date, datetime.strptime("09:15", "%H:%M").time())
    end_time = datetime.combine(date, datetime.strptime("09:24", "%H:%M").time())
    
    # Filter data for first 10 minutes
    mask = (df["time"] >= start_time) & (df["time"] <= end_time)
    first_10_min = df[mask].copy()
    
    if first_10_min.empty:
        return []
    
    # Calculate turnover per stock
    first_10_min["turnover"] = first_10_min["volume"] * first_10_min["close"]
    turnover_by_stock = first_10_min.groupby("ticker")["turnover"].sum().sort_values(ascending=False)
    
    # Return top 10 tickers
    top10 = turnover_by_stock.head(10).index.tolist()
    return top10


def prepare_stock_data_with_indicators(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare stock data with all required indicators:
    - EMA(3) on 10-min timeframe
    - EMA(10) on 10-min timeframe  
    - EMA(50) on 1-hour timeframe
    - RSI(14) on 10-min timeframe
    """
    if df_minute.empty:
        return df_minute
    
    # Resample to 10-minute and 1-hour timeframes
    df_10min = resample_to_timeframe(df_minute.copy(), "10min")
    df_1hour = resample_to_timeframe(df_minute.copy(), "60min")
    
    if df_10min.empty or df_1hour.empty:
        return pd.DataFrame()
    
    # Calculate indicators on respective timeframes
    df_10min["ema3"] = calculate_ema(df_10min["close"], span=3)
    df_10min["ema10"] = calculate_ema(df_10min["close"], span=10)
    df_10min["rsi14"] = calculate_rsi(df_10min["close"], period=14)
    df_1hour["ema50"] = calculate_ema(df_1hour["close"], span=50)
    
    # Map indicators back to minute-level data
    df_minute = df_minute.set_index("time")
    
    # Create indicator series aligned to minute data
    df_10min_indexed = df_10min.set_index("time")
    df_1hour_indexed = df_1hour.set_index("time")
    
    # Forward-fill indicators to minute resolution
    ema3_min = df_10min_indexed["ema3"].resample("1min").ffill()
    ema10_min = df_10min_indexed["ema10"].resample("1min").ffill()
    rsi14_min = df_10min_indexed["rsi14"].resample("1min").ffill()
    ema50_min = df_1hour_indexed["ema50"].resample("1min").ffill()
    
    # Assign to minute dataframe
    df_minute["ema3"] = ema3_min.reindex(df_minute.index).ffill()
    df_minute["ema10"] = ema10_min.reindex(df_minute.index).ffill()
    df_minute["rsi14"] = rsi14_min.reindex(df_minute.index).ffill()
    df_minute["ema50"] = ema50_min.reindex(df_minute.index).ffill()
    
    df_minute = df_minute.reset_index()
    return df_minute


# ========== TRADING SIMULATION ==========

class Order:
    """Represents a single order/trade with metadata and P&L tracking.

    Fields:
    - symbol, side, entry_time, entry_price, size
    - exit_time, exit_price, pnl, status, reason
    - stop_loss, target, trail_active, trail_stop
    - position_value: entry_price * size
    """
    def __init__(self, symbol: str, side: str, entry_time: datetime, entry_price: float, size: int):
        self.symbol = symbol
        self.side = side  # 'LONG' or 'SHORT'
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.size = size
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0.0
        self.status = "OPEN"
        self.reason = None
        self.stop_loss = None
        self.target = None
        self.trail_active = False
        self.trail_stop = None

    def position_value(self) -> float:
        try:
            return float(self.entry_price) * int(self.size)
        except Exception:
            return None

    def to_dict(self) -> dict:
        return {
            "time": self.entry_time,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "size": self.size,
            "position_value": round(self.position_value(), 2) if self.position_value() is not None else None,
            "status": self.status,
            "exit_time": self.exit_time,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "pnl": round(self.pnl, 2),
            "reason": self.reason,
            "stop_loss": round(self.stop_loss, 4) if self.stop_loss is not None else None,
            "target": round(self.target, 4) if self.target is not None else None,
            "trail_active": bool(self.trail_active),
            "trail_stop": round(self.trail_stop, 4) if self.trail_stop is not None else None
        }


def simulate_stock_trading(symbol: str, df: pd.DataFrame, date: datetime.date, 
                           start_equity: float) -> list:
    """
    Simulate trading for a single stock on a given date
    Returns list of completed trades
    """
    trades = []
    current_order = None
    entered_today = False
    
    # Risk amount per trade
    risk_amount = start_equity * RISK_PER_TRADE_PCT
    
    # Filter to target date only and after 09:25
    trade_start_time = datetime.combine(date, datetime.strptime("09:25", "%H:%M").time())
    eod_time = datetime.combine(date, datetime.strptime("15:25", "%H:%M").time())
    
    # Keep full day data for lookback (needed for SHORT entry - last 5 candles)
    df_full_day = df[df["time"].dt.date == date].copy().reset_index(drop=True)
    df_day = df_full_day[df_full_day["time"] >= trade_start_time].copy()
    df_day = df_day.reset_index(drop=True)
    
    if df_day.empty:
        return trades
    
    for i in range(len(df_day)):
        row = df_day.iloc[i]
        current_time = row["time"]
        
        # Check for exit conditions if we have an open position
        if current_order is not None:
            exit_price = None
            exit_reason = None

            if current_order.side == "LONG":
                # Check stop loss
                if row["low"] <= current_order.stop_loss:
                    exit_price = current_order.stop_loss
                    exit_reason = "SL"
                # Check target
                elif row["high"] >= current_order.target:
                    exit_price = current_order.target
                    exit_reason = "TG"
                else:
                    # Check trailing stop activation
                    if not current_order.trail_active:
                        if row["high"] >= current_order.entry_price * (1 + TRAIL_START_PCT):
                            current_order.trail_active = True
                            current_order.trail_stop = row["high"] * (1 - TRAIL_STEP_PCT)

                    # Update and check trailing stop
                    if current_order.trail_active:
                        new_trail = row["high"] * (1 - TRAIL_STEP_PCT)
                        current_order.trail_stop = max(current_order.trail_stop, new_trail)

                        if row["low"] <= current_order.trail_stop:
                            exit_price = current_order.trail_stop
                            exit_reason = "TRAIL"

                # Calculate PnL for long
                if exit_price:
                    current_order.pnl = (exit_price - current_order.entry_price) * current_order.size

            else:  # SHORT
                # Check stop loss
                if row["high"] >= current_order.stop_loss:
                    exit_price = current_order.stop_loss
                    exit_reason = "SL"
                # Check target
                elif row["low"] <= current_order.target:
                    exit_price = current_order.target
                    exit_reason = "TG"
                else:
                    # Check trailing stop activation
                    if not current_order.trail_active:
                        if row["low"] <= current_order.entry_price * (1 - TRAIL_START_PCT):
                            current_order.trail_active = True
                            current_order.trail_stop = row["low"] * (1 + TRAIL_STEP_PCT)

                    # Update and check trailing stop
                    if current_order.trail_active:
                        new_trail = row["low"] * (1 + TRAIL_STEP_PCT)
                        current_order.trail_stop = min(current_order.trail_stop, new_trail)

                        if row["high"] >= current_order.trail_stop:
                            exit_price = current_order.trail_stop
                            exit_reason = "TRAIL"

                # Calculate PnL for short
                if exit_price:
                    current_order.pnl = (current_order.entry_price - exit_price) * current_order.size

            # Close the order if exit condition met
            if exit_price:
                current_order.exit_time = current_time
                current_order.exit_price = exit_price
                current_order.status = "CLOSED"
                current_order.reason = exit_reason
                trades.append(current_order.to_dict())
                current_order = None
                continue
        
        # Check for EOD forced close
        if current_order is not None and current_time >= eod_time:
            exit_price = row["close"]
            if current_order.side == "LONG":
                current_order.pnl = (exit_price - current_order.entry_price) * current_order.size
            else:
                current_order.pnl = (current_order.entry_price - exit_price) * current_order.size

            current_order.exit_time = current_time
            current_order.exit_price = exit_price
            current_order.status = "CLOSED"
            current_order.reason = "EOD"
            trades.append(current_order.to_dict())
            current_order = None
            continue
    # Entry check (only allow entries on the first minute after trade_start_time)
        if current_order is None and not entered_today and i < ENTRY_WINDOW_MINUTES:
            # Ensure indicators are available
            if pd.isnull(row.get("ema3")) or pd.isnull(row.get("ema10")) or \
               pd.isnull(row.get("rsi14")) or pd.isnull(row.get("ema50")):
                # skip entry if indicators not ready
                pass
            else:
                # LONG Entry: EMA(3) > EMA(10) AND RSI(14) > 60 AND Close > EMA(50)
                if row["ema3"] > row["ema10"] and row["rsi14"] > 60 and row["close"] > row["ema50"]:
                    entry_price = row["high"]  # Buy at high of entry candle
                    size = int(risk_amount / (entry_price * STOP_LOSS_PCT))

                    if size > 0:
                        current_order = Order(symbol, "LONG", current_time, entry_price, size)
                        current_order.stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                        current_order.target = entry_price * (1 + PROFIT_TARGET_PCT)
                        entered_today = True
                
                # SHORT Entry: EMA(3) < EMA(10) AND RSI(14) < 30 AND Close < EMA(50)
                if current_order is None and row["ema3"] < row["ema10"] and row["rsi14"] < 30 and row["close"] < row["ema50"]:
                    # Entry at low of last 5 1-min candles (use full day data for proper lookback)
                    full_day_idx = df_full_day[df_full_day["time"] == current_time].index
                    if len(full_day_idx) > 0:
                        idx_in_full = full_day_idx[0]
                        start_idx = max(0, idx_in_full - 4)
                        recent_candles = df_full_day.iloc[start_idx:idx_in_full+1]
                        entry_price = recent_candles["low"].min()
                    else:
                        entry_price = row["low"]

                    size = int(risk_amount / (entry_price * STOP_LOSS_PCT))

                    if size > 0:
                        current_order = Order(symbol, "SHORT", current_time, entry_price, size)
                        current_order.stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                        current_order.target = entry_price * (1 - PROFIT_TARGET_PCT)
                        entered_today = True
    
    # Handle any remaining open position at end of data
    if current_order is not None:
        last_row = df_day.iloc[-1]
        exit_price = last_row["close"]
        if current_order.side == "LONG":
            current_order.pnl = (exit_price - current_order.entry_price) * current_order.size
        else:
            current_order.pnl = (current_order.entry_price - exit_price) * current_order.size

        current_order.exit_time = last_row["time"]
        current_order.exit_price = exit_price
        current_order.status = "CLOSED"
        current_order.reason = "EOD"
        trades.append(current_order.to_dict())
    
    return trades


def backtest_single_date(args: tuple) -> dict:
    """
    Run backtest for a single date (used for multiprocessing)
    Returns dict with date, trades, and metrics
    """
    date, start_equity, data_dir = args
    
    # Load data for the date
    df = load_data_for_date(date, data_dir)
    
    if df.empty:
        return {
            "date": str(date),
            "trades": [],
            "metrics": {
                "date": str(date),
                "num_trades": 0,
                "total_pnl": 0.0,
                "return_pct": 0.0,
                "win_rate": 0.0,
                "eod_equity": start_equity
            }
        }
    
    # Select top 10 stocks by turnover
    top10_stocks = calculate_turnover_top10(df, date)
    
    if not top10_stocks:
        return {
            "date": str(date),
            "trades": [],
            "metrics": {
                "date": str(date),
                "num_trades": 0,
                "total_pnl": 0.0,
                "return_pct": 0.0,
                "win_rate": 0.0,
                "eod_equity": start_equity
            }
        }
    
    all_trades = []
    # local timing
    local_indicator_time = 0.0
    local_simulation_time = 0.0
    
    # Simulate trading for each stock
    for symbol in top10_stocks:
        stock_df = df[df["ticker"] == symbol].copy()
        if stock_df.empty:
            continue
            
        # Prepare indicators
        t0 = time.time()
        stock_df_with_indicators = prepare_stock_data_with_indicators(stock_df)
        t1 = time.time()
        local_indicator_time += (t1 - t0)
        
        if stock_df_with_indicators.empty:
            continue
        
    # Run simulation
        t2 = time.time()
        stock_trades = simulate_stock_trading(symbol, stock_df_with_indicators, date, start_equity)
        t3 = time.time()
        local_simulation_time += (t3 - t2)
        all_trades.extend(stock_trades)
    
    # Calculate metrics
    total_pnl = sum(t["pnl"] for t in all_trades if t["pnl"] is not None)
    num_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    win_rate = (wins / num_trades * 100) if num_trades > 0 else 0.0
    return_pct = (total_pnl / start_equity * 100) if start_equity > 0 else 0.0
    eod_equity = start_equity + total_pnl
    
    metrics = {
        "date": str(date),
        "num_trades": num_trades,
        "total_pnl": round(total_pnl, 2),
        "return_pct": round(return_pct, 4),
        "win_rate": round(win_rate, 2),
        "eod_equity": round(eod_equity, 2)
    }
    
    return {
        "date": str(date),
        "trades": all_trades,
        "metrics": metrics,
        "timing": {
            "indicator_time_sec": round(local_indicator_time, 6),
            "simulation_time_sec": round(local_simulation_time, 6)
        }
    }



# ========== PERFORMANCE METRICS ==========

def calculate_performance_metrics(daily_results: list, base_capital: float) -> dict:
    """
    Calculate overall performance metrics from daily results
    """
    if not daily_results:
        return {}
    
    # Aggregate all trades
    all_trades = []
    for result in daily_results:
        all_trades.extend(result["trades"])
    
    if not all_trades:
        return {
            "total_trades": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "sharpe_ratio": None,
            "final_equity": base_capital
        }
    
    # Build equity curve
    equity = base_capital
    equity_curve = [base_capital]
    daily_returns = []
    
    for result in daily_results:
        day_pnl = sum(t["pnl"] for t in result["trades"] if t["pnl"] is not None)
        prev_equity = equity
        equity += day_pnl
        equity_curve.append(equity)
        
        if prev_equity > 0:
            daily_returns.append(day_pnl / prev_equity)
    
    # Calculate metrics
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    # Total return
    total_return_pct = ((equity - base_capital) / base_capital * 100)
    
    # Max drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min()) * 100
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe_ratio = None
    
    return {
        "total_trades": total_trades,
        "total_return_pct": round(total_return_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe_ratio, 2) if sharpe_ratio is not None else None,
        "final_equity": round(equity, 2),
        "net_pnl": round(equity - base_capital, 2)
    }


# ========== MAIN FUNCTIONS ==========

def get_available_dates(data_dir: Path = DATA_DIR) -> list:
    """Get all available dates from data files"""
    files = sorted(data_dir.glob("dataNSE_*.csv"))
    dates = []
    
    for f in files:
        # Extract date from filename: dataNSE_YYYYMMDD.csv
        date_str = f.stem.replace("dataNSE_", "")
        try:
            date = datetime.strptime(date_str, "%Y%m%d").date()
            dates.append(date)
        except ValueError:
            continue
    
    return sorted(dates)


def run_backtest(start_date: str, end_date: str = None, use_multiprocessing: bool = True, 
                 num_workers: int = None, verbose: bool = False, profile_enabled: bool = False) -> dict:
    """
    Run the backtest from start_date to end_date
    
    Args:
        start_date: Start date in format 'YYYY-MM-DD' or 'MM-DD-YYYY'
        end_date: End date (optional, defaults to all available dates after start)
        use_multiprocessing: Whether to use multiprocessing
        num_workers: Number of worker processes (default: CPU count)
        verbose: Print verbose output
        
    Returns:
        Dict containing all results and metrics
    """
    # Parse start date (support multiple formats)
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError:
        try:
            start_dt = datetime.strptime(start_date, "%m-%d-%Y").date()
        except ValueError:
            start_dt = datetime.strptime(start_date, "%d-%m-%Y").date()
    
    # Parse end date if provided
    end_dt = None
    if end_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            try:
                end_dt = datetime.strptime(end_date, "%m-%d-%Y").date()
            except ValueError:
                end_dt = datetime.strptime(end_date, "%d-%m-%Y").date()
    
    # Get available dates
    available_dates = get_available_dates(DATA_DIR)
    
    if not available_dates:
        print(f"No data files found in {DATA_DIR}")
        return {}
    
    # Filter dates
    dates_to_process = [d for d in available_dates if d >= start_dt]
    if end_dt:
        dates_to_process = [d for d in dates_to_process if d <= end_dt]
    
    if not dates_to_process:
        print(f"No dates found in range {start_dt} to {end_dt or 'end'}")
        return {}
    
    print(f"\n{'='*60}")
    print(f"INTRADAY EMA-RSI STRATEGY BACKTEST")
    print(f"{'='*60}")
    print(f"Base Capital: ₹{BASE_CAPITAL:,.2f}")
    print(f"Date Range: {dates_to_process[0]} to {dates_to_process[-1]}")
    print(f"Trading Days: {len(dates_to_process)}")
    print(f"{'='*60}\n")
    
    # Create output directory
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for each date
    # For proper equity carry-forward, we process dates sequentially
    # but can still parallelize within each day
    
    daily_results = []
    running_equity = BASE_CAPITAL
    
    # profiling accumulators
    total_indicator_time = 0.0
    total_simulation_time = 0.0

    if use_multiprocessing and len(dates_to_process) > 1:
        # For multiprocessing, we need to process in batches
        # Process each date sequentially to maintain equity carry-forward
        num_workers = num_workers or min(cpu_count(), 4)
        
        with tqdm(dates_to_process, desc="Processing dates", unit="day") as pbar:
            for date in pbar:
                pbar.set_postfix({"date": str(date), "equity": f"₹{running_equity:,.0f}"})
                
                result = backtest_single_date((date, running_equity, DATA_DIR))
                daily_results.append(result)
                if profile_enabled and result.get('timing'):
                    total_indicator_time += result['timing'].get('indicator_time_sec', 0.0)
                    total_simulation_time += result['timing'].get('simulation_time_sec', 0.0)

                # Update running equity
                running_equity = result["metrics"]["eod_equity"]
                
                # Save daily trade log
                if result["trades"]:
                    trade_df = pd.DataFrame(result["trades"])
                    trade_log_path = TRADE_LOG_DIR / f"trade_log_{date}.csv"
                    trade_df.to_csv(trade_log_path, index=False, float_format='%.2f')
    else:
        # Sequential processing
        for date in tqdm(dates_to_process, desc="Processing dates", unit="day"):
            result = backtest_single_date((date, running_equity, DATA_DIR))
            daily_results.append(result)
            if profile_enabled and result.get('timing'):
                total_indicator_time += result['timing'].get('indicator_time_sec', 0.0)
                total_simulation_time += result['timing'].get('simulation_time_sec', 0.0)

            # Update running equity
            running_equity = result["metrics"]["eod_equity"]
            
            # Save daily trade log
            if result["trades"]:
                trade_df = pd.DataFrame(result["trades"])
                trade_log_path = TRADE_LOG_DIR / f"trade_log_{date}.csv"
                trade_df.to_csv(trade_log_path, index=False, float_format='%.2f')
    
    # Calculate overall performance metrics
    overall_metrics = calculate_performance_metrics(daily_results, BASE_CAPITAL)
    
    # Save daily summary
    daily_summary = pd.DataFrame([r["metrics"] for r in daily_results])
    summary_path = TRADE_LOG_DIR / "daily_summary.csv"
    daily_summary.to_csv(summary_path, index=False, float_format='%.2f')
    
    # Save all trades combined
    all_trades = []
    for r in daily_results:
        all_trades.extend(r["trades"])
    
    if all_trades:
        all_trades_df = pd.DataFrame(all_trades)
        all_trades_path = TRADE_LOG_DIR / "all_trades.csv"
        all_trades_df.to_csv(all_trades_path, index=False, float_format='%.2f')
    
    # Print results
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Total Trades:       {overall_metrics.get('total_trades', 0)}")
    print(f"Win Rate:           {overall_metrics.get('win_rate_pct', 0):.2f}%")
    print(f"Total Return:       {overall_metrics.get('total_return_pct', 0):.2f}%")
    print(f"Max Drawdown:       {overall_metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Sharpe Ratio:       {overall_metrics.get('sharpe_ratio', 'N/A')}")
    print(f"Final Equity:       ₹{overall_metrics.get('final_equity', BASE_CAPITAL):,.2f}")
    print(f"Net P&L:            ₹{overall_metrics.get('net_pnl', 0):,.2f}")
    print(f"{'='*60}")
    
    print(f"\nTrade logs saved to: {TRADE_LOG_DIR}")
    print(f"Daily summary: {summary_path}")
    if all_trades:
        print(f"All trades: {all_trades_path}")
    
    # Write profile if enabled
    if profile_enabled:
        try:
            import json
            profile_path = TRADE_LOG_DIR / "profile_summary.json"
            profile = {
                "total_indicator_time_sec": round(total_indicator_time, 6),
                "total_simulation_time_sec": round(total_simulation_time, 6),
                "dates_processed": len(dates_to_process)
            }
            with open(profile_path, 'w') as fh:
                json.dump(profile, fh, indent=2)
            print(f"Profile summary written to: {profile_path}")
        except Exception:
            pass

    # Save equity curve and plot
    try:
        equity_curve = [BASE_CAPITAL]
        for r in daily_results:
            day_pnl = sum(t.get('pnl', 0) for t in r['trades'] if t.get('pnl') is not None)
            equity_curve.append(equity_curve[-1] + day_pnl)

        equity_df = pd.DataFrame({
            'day': list(range(len(equity_curve))),
            'equity': equity_curve
        })
        equity_csv_path = TRADE_LOG_DIR / 'equity_curve.csv'
        equity_df.to_csv(equity_csv_path, index=False, float_format='%.2f')

        if HAVE_MATPLOTLIB:
            plt.figure(figsize=(10, 5))
            plt.plot(equity_df['day'], equity_df['equity'], marker='o', linewidth=1)
            plt.xlabel('Day')
            plt.ylabel('Equity (INR)')
            plt.title('Equity Curve')
            plt.grid(True)
            equity_png = TRADE_LOG_DIR / 'equity_curve.png'
            plt.savefig(equity_png, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Equity curve saved to: {equity_png}")
        else:
            print(f"Equity CSV saved to: {equity_csv_path} (matplotlib not available for PNG)")
    except Exception as e:
        print(f"Failed to save equity curve: {e}")
    
    return {
        "daily_results": daily_results,
        "overall_metrics": overall_metrics,
        "daily_summary": daily_summary
    }


def main():
    parser = argparse.ArgumentParser(
        description="Intraday EMA-RSI Stock Strategy Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python new_strategy.py --start-date 2025-08-01
  python new_strategy.py --start-date 2025-08-01 --end-date 2025-08-31
  python new_strategy.py --all
  python new_strategy.py --start-date 08-01-2025  # MM-DD-YYYY format also supported
        """
    )
    
    parser.add_argument(
        "--start-date", "-s",
        type=str,
        help="Start date for backtest (YYYY-MM-DD or MM-DD-YYYY)"
    )
    
    parser.add_argument(
        "--end-date", "-e",
        type=str,
        default=None,
        help="End date for backtest (optional)"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run backtest for all available dates"
    )
    
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        help="Disable multiprocessing"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: ./stock_data)"
    )

    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt for dates; use defaults or provided args"
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record simple timing profile for indicator calc vs simulation"
    )

    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload all CSVs into memory before running (uses more RAM)"
    )
    
    args = parser.parse_args()
    
    # Update data directory if specified
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)

    # Preload data if requested
    if getattr(args, 'preload', False):
        print("Preloading all CSVs into memory...")
        preload_all_data(DATA_DIR)

    # Enforce fixed minimum start date
    MIN_START_DATE = datetime.strptime("2025-08-01", "%Y-%m-%d").date()
    
    # Determine date range
    available_dates = get_available_dates(DATA_DIR)
    if not available_dates:
        print(f"No data files found in {DATA_DIR}")
        sys.exit(1)

    if args.all:
        # Force start at MIN_START_DATE; ensure data exists from that date
        if not any(d >= MIN_START_DATE for d in available_dates):
            print(f"No available data on or after {MIN_START_DATE}. Cannot run.")
            sys.exit(1)
        start_date = str(MIN_START_DATE)
        end_date = str(available_dates[-1])
    elif args.start_date:
        # Ignore provided start date and enforce MIN_START_DATE
        start_date = str(MIN_START_DATE)
        end_date = args.end_date
        print(f"Start date fixed to {start_date} (user-provided start date ignored).")
    else:
        # Interactive prompt for start/end dates with defaults but enforce MIN_START_DATE
        if args.non_interactive:
            start_date = str(MIN_START_DATE)
            end_date = args.end_date or "2025-08-28"
        else:
            default_start = str(MIN_START_DATE)
            default_end = "2025-08-28"
            print("No date range provided. Press Enter to accept the default shown in brackets.")
            inp_start = input(f"Enter start date [default: {default_start}] (YYYY-MM-DD): ").strip()
            inp_end = input(f"Enter end date   [default: {default_end}] (YYYY-MM-DD): ").strip()
            # use defaults but enforce minimum
            chosen_start = inp_start if inp_start else default_start
            chosen_dt = datetime.strptime(chosen_start, "%Y-%m-%d").date()
            if chosen_dt < MIN_START_DATE:
                print(f"Provided start date {chosen_start} is before minimum allowed {MIN_START_DATE}. Using {MIN_START_DATE}.")
                start_date = str(MIN_START_DATE)
            else:
                start_date = chosen_start
            end_date = inp_end if inp_end else default_end
    
    # Run backtest
    results = run_backtest(
    start_date=start_date,
    end_date=end_date,
    use_multiprocessing=not args.no_multiprocessing,
    num_workers=args.workers,
    verbose=args.verbose,
    profile_enabled=getattr(args, 'profile', False)
    )
    
    return results


if __name__ == "__main__":
    main()
