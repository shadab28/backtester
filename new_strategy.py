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
TRADE_LOG_DIR = SCRIPT_DIR / "trade_logs_new_strategy"

# ========== INDICATOR FUNCTIONS ==========

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
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
    file_path = data_dir / f"dataNSE_{date_str}.csv"
    
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, parse_dates=["time"])
    df = df.sort_values(["ticker", "time"]).reset_index(drop=True)
    return df


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

class Trade:
    """Represents a single trade"""
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
        
    def to_dict(self) -> dict:
        return {
            "time": self.entry_time,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": round(self.entry_price, 2),
            "size": self.size,
            "status": self.status,
            "exit_time": self.exit_time,
            "exit_price": round(self.exit_price, 2) if self.exit_price else None,
            "pnl": round(self.pnl, 2),
            "reason": self.reason
        }


def simulate_stock_trading(symbol: str, df: pd.DataFrame, date: datetime.date, 
                           start_equity: float) -> list:
    """
    Simulate trading for a single stock on a given date
    Returns list of completed trades
    """
    trades = []
    current_trade = None
    entered_today = False
    
    # Risk amount per trade
    risk_amount = start_equity * RISK_PER_TRADE_PCT
    
    # Filter to target date only and after 09:25
    trade_start_time = datetime.combine(date, datetime.strptime("09:25", "%H:%M").time())
    eod_time = datetime.combine(date, datetime.strptime("15:29", "%H:%M").time())
    
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
        if current_trade is not None:
            exit_price = None
            exit_reason = None
            
            if current_trade.side == "LONG":
                # Check stop loss
                if row["low"] <= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss
                    exit_reason = "SL"
                # Check target
                elif row["high"] >= current_trade.target:
                    exit_price = current_trade.target
                    exit_reason = "TG"
                else:
                    # Check trailing stop activation
                    if not current_trade.trail_active:
                        if row["high"] >= current_trade.entry_price * (1 + TRAIL_START_PCT):
                            current_trade.trail_active = True
                            current_trade.trail_stop = row["high"] * (1 - TRAIL_STEP_PCT)
                    
                    # Update and check trailing stop
                    if current_trade.trail_active:
                        new_trail = row["high"] * (1 - TRAIL_STEP_PCT)
                        current_trade.trail_stop = max(current_trade.trail_stop, new_trail)
                        
                        if row["low"] <= current_trade.trail_stop:
                            exit_price = current_trade.trail_stop
                            exit_reason = "TRAIL"
                
                # Calculate PnL for long
                if exit_price:
                    current_trade.pnl = (exit_price - current_trade.entry_price) * current_trade.size
                    
            else:  # SHORT
                # Check stop loss
                if row["high"] >= current_trade.stop_loss:
                    exit_price = current_trade.stop_loss
                    exit_reason = "SL"
                # Check target
                elif row["low"] <= current_trade.target:
                    exit_price = current_trade.target
                    exit_reason = "TG"
                else:
                    # Check trailing stop activation
                    if not current_trade.trail_active:
                        if row["low"] <= current_trade.entry_price * (1 - TRAIL_START_PCT):
                            current_trade.trail_active = True
                            current_trade.trail_stop = row["low"] * (1 + TRAIL_STEP_PCT)
                    
                    # Update and check trailing stop
                    if current_trade.trail_active:
                        new_trail = row["low"] * (1 + TRAIL_STEP_PCT)
                        current_trade.trail_stop = min(current_trade.trail_stop, new_trail)
                        
                        if row["high"] >= current_trade.trail_stop:
                            exit_price = current_trade.trail_stop
                            exit_reason = "TRAIL"
                
                # Calculate PnL for short
                if exit_price:
                    current_trade.pnl = (current_trade.entry_price - exit_price) * current_trade.size
            
            # Close the trade if exit condition met
            if exit_price:
                current_trade.exit_time = current_time
                current_trade.exit_price = exit_price
                current_trade.status = "CLOSED"
                current_trade.reason = exit_reason
                trades.append(current_trade.to_dict())
                current_trade = None
                continue
        
        # Check for EOD forced close
        if current_trade is not None and current_time >= eod_time:
            exit_price = row["close"]
            if current_trade.side == "LONG":
                current_trade.pnl = (exit_price - current_trade.entry_price) * current_trade.size
            else:
                current_trade.pnl = (current_trade.entry_price - exit_price) * current_trade.size
            
            current_trade.exit_time = current_time
            current_trade.exit_price = exit_price
            current_trade.status = "CLOSED"
            current_trade.reason = "EOD"
            trades.append(current_trade.to_dict())
            current_trade = None
            continue
        
        # Check for entry conditions (only if no position and not already entered today)
        if current_trade is None and not entered_today:
            # Ensure indicators are available
            if pd.isnull(row.get("ema3")) or pd.isnull(row.get("ema10")) or \
               pd.isnull(row.get("rsi14")) or pd.isnull(row.get("ema50")):
                continue
            
            # LONG Entry: EMA(3) > EMA(10) AND RSI(14) > 60 AND Close > EMA(50)
            if row["ema3"] > row["ema10"] and row["rsi14"] > 60 and row["close"] > row["ema50"]:
                entry_price = row["high"]  # Buy at high of entry candle
                size = int(risk_amount / (entry_price * STOP_LOSS_PCT))
                
                if size > 0:
                    current_trade = Trade(symbol, "LONG", current_time, entry_price, size)
                    current_trade.stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    current_trade.target = entry_price * (1 + PROFIT_TARGET_PCT)
                    entered_today = True
                    continue
            
            # SHORT Entry: EMA(3) < EMA(10) AND RSI(14) < 30 AND Close < EMA(50)
            if row["ema3"] < row["ema10"] and row["rsi14"] < 30 and row["close"] < row["ema50"]:
                # Entry at low of last 5 1-min candles (use full day data for proper lookback)
                # Find current time index in full day data
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
                    current_trade = Trade(symbol, "SHORT", current_time, entry_price, size)
                    current_trade.stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    current_trade.target = entry_price * (1 - PROFIT_TARGET_PCT)
                    entered_today = True
                    continue
    
    # Handle any remaining open position at end of data
    if current_trade is not None:
        last_row = df_day.iloc[-1]
        exit_price = last_row["close"]
        if current_trade.side == "LONG":
            current_trade.pnl = (exit_price - current_trade.entry_price) * current_trade.size
        else:
            current_trade.pnl = (current_trade.entry_price - exit_price) * current_trade.size
        
        current_trade.exit_time = last_row["time"]
        current_trade.exit_price = exit_price
        current_trade.status = "CLOSED"
        current_trade.reason = "EOD"
        trades.append(current_trade.to_dict())
    
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
    
    # Simulate trading for each stock
    for symbol in top10_stocks:
        stock_df = df[df["ticker"] == symbol].copy()
        if stock_df.empty:
            continue
            
        # Prepare indicators
        stock_df_with_indicators = prepare_stock_data_with_indicators(stock_df)
        
        if stock_df_with_indicators.empty:
            continue
        
        # Run simulation
        stock_trades = simulate_stock_trading(symbol, stock_df_with_indicators, date, start_equity)
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
        "metrics": metrics
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
                 num_workers: int = None, verbose: bool = False) -> dict:
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
    
    if use_multiprocessing and len(dates_to_process) > 1:
        # For multiprocessing, we need to process in batches
        # Process each date sequentially to maintain equity carry-forward
        num_workers = num_workers or min(cpu_count(), 4)
        
        with tqdm(dates_to_process, desc="Processing dates", unit="day") as pbar:
            for date in pbar:
                pbar.set_postfix({"date": str(date), "equity": f"₹{running_equity:,.0f}"})
                
                result = backtest_single_date((date, running_equity, DATA_DIR))
                daily_results.append(result)
                
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
    
    args = parser.parse_args()
    
    # Update data directory if specified
    global DATA_DIR
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    
    # Determine date range
    if args.all:
        available_dates = get_available_dates(DATA_DIR)
        if not available_dates:
            print(f"No data files found in {DATA_DIR}")
            sys.exit(1)
        start_date = str(available_dates[0])
        end_date = str(available_dates[-1])
    elif args.start_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # Default: start from 2025-08-01
        start_date = "2025-08-01"
        end_date = args.end_date
    
    # Run backtest
    results = run_backtest(
        start_date=start_date,
        end_date=end_date,
        use_multiprocessing=not args.no_multiprocessing,
        num_workers=args.workers,
        verbose=args.verbose
    )
    
    return results


if __name__ == "__main__":
    main()
