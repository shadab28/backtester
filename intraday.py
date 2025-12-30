"""
Intraday EMA-RSI Backtester
- Loads minute data CSVs from stock_data_aug_2025 (format: ticker,time,open,high,low,close,volume)
- At 09:25 select top 10 stocks by turnover (sum of volume*close for first 10 minutes 09:15-09:24)
- Indicators (on 10-min timeframe): EMA(3), EMA(10), RSI(14)
- EMA(50) on 1-hour timeframe
- Entry/exit rules as specified by user
- Generates trade log CSV and prints performance metrics

Usage: run the script in the repo root. Adjust paths/constants as needed.
"""
import glob
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

BASE_CAPITAL = 10_00_000  # 10 lac (Indian grouping)
DEFAULT_DATA_DIR = Path(__file__).parent / "stock_data_aug_2025"

# TRADE_LOG_DIR will be created to store per-date trade logs and summaries
TRADE_LOG_DIR = Path(__file__).parent / "trade_logs"
TRADE_LOG = None

# Strategy params
RISK_PER_TRADE_PCT = 0.005  # 0.5% of capital
STOP_LOSS_PCT = 0.005
PROFIT_TARGET_PCT = 0.02
TRAIL_START_PCT = 0.005
TRAIL_STEP_PCT = 0.0075

# Helper indicators

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def load_symbol_minute_df(filepath):
    df = pd.read_csv(filepath, parse_dates=["time"])  # expected columns
    df = df.sort_values("time").reset_index(drop=True)
    return df


def resample_to_10min(df):
    # df indexed by time
    d = df.set_index("time").resample("10min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    d = d.reset_index()
    return d


def resample_to_1h(df):
    d = df.set_index("time").resample("60min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    d = d.reset_index()
    return d


class Backtester:
    def __init__(self, date_str, start_equity=None, verbose=False, data_dir=None):
        self.date = pd.to_datetime(date_str).date()
        # allow passing a data_dir when Backtester is invoked programmatically
        if data_dir:
            data_path = Path(data_dir)
        else:
            data_path = DATA_DIR
        self.files = sorted(data_path.glob("dataNSE_*.csv"))
        self.symbol_dfs = {}  # ticker -> minute df
        self.load_date_files()
        self.trade_log = []
        # allow passing in a starting equity (NAV) so we can carry NAV across days
        self.equity = float(start_equity) if start_equity is not None else BASE_CAPITAL
        self.equity_curve = []
        self.verbose = verbose

    def load_date_files(self):
        # For simplicity each dataNSE file contains many tickers; we'll load and filter by date
        # We'll read one file at a time and build per-symbol minute frames containing
        # all historical data up to and including the target date. This ensures EMA/RSI
        # calculations use past candles.
        files = self.files
        symbol_frames = {}
        for f in files:
            df = pd.read_csv(f, parse_dates=["time"])  # ticker,time,open,high,low,close,volume
            # keep rows with time.date <= target date
            df = df[df["time"].dt.date <= self.date]
            if df.empty:
                continue
            for sym, g in df.groupby("ticker"):
                if sym not in symbol_frames:
                    symbol_frames[sym] = g[[
                        "time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]].copy()
                else:
                    symbol_frames[sym] = pd.concat([symbol_frames[sym], g[["time","open","high","low","close","volume"]]])
        # finalize
        for sym, sdf in symbol_frames.items():
            sdf = sdf.sort_values("time").reset_index(drop=True)
            self.symbol_dfs[sym] = sdf
        print(f"Loaded {len(self.symbol_dfs)} symbols for {self.date}")

    def compute_turnover_top10(self):
        # turnover = sum(volume * close) for first 10 minutes 09:15-09:24 inclusive
        turns = []
        start_t = datetime.combine(self.date, datetime.min.time()) + timedelta(hours=9, minutes=15)
        end_t = start_t + timedelta(minutes=9)
        for sym, df in self.symbol_dfs.items():
            mask = (df["time"] >= pd.Timestamp(start_t)) & (df["time"] <= pd.Timestamp(end_t))
            head = df.loc[mask]
            if head.empty:
                continue
            turnover = (head["volume"] * head["close"]).sum()
            turns.append((sym, turnover))
        turns.sort(key=lambda x: x[1], reverse=True)
        top10 = [s for s, t in turns[:10]]
        print("Top10 by turnover:", top10)
        return top10

    def prepare_indicators(self, sym, df_min):
        # compute 10-min and 1-hour TF indicators
        df_10 = resample_to_10min(df_min)
        df_1h = resample_to_1h(df_min)
        # merge indicators back to minute frame by forward fill
        df_10["ema3"] = ema(df_10["close"], span=3)
        df_10["ema10"] = ema(df_10["close"], span=10)
        df_10["rsi14"] = rsi(df_10["close"], period=14)
        df_1h["ema50"] = ema(df_1h["close"], span=50)
        # align to minute timestamps by reindexing
        df_10_idx = df_10.set_index("time")
        df_1h_idx = df_1h.set_index("time")
        m = df_min.set_index("time")
        # expand 10min/1h indicators to minute-level by resampling to 1min and forward-filling
        ema3_min = df_10_idx["ema3"].resample("1min").ffill()
        ema10_min = df_10_idx["ema10"].resample("1min").ffill()
        rsi14_min = df_10_idx["rsi14"].resample("1min").ffill()
        ema50_min = df_1h_idx["ema50"].resample("1min").ffill()
        m = m.assign(
            ema3 = ema3_min.reindex(m.index).ffill(),
            ema10 = ema10_min.reindex(m.index).ffill(),
            rsi14 = rsi14_min.reindex(m.index).ffill(),
            ema50 = ema50_min.reindex(m.index).ffill(),
        )
        m = m.reset_index()
        if self.verbose:
            # show earliest timestamp where indicators exist (robust)
            try:
                idx_ema3 = m['ema3'].first_valid_index()
                ema3_first = m.loc[idx_ema3, 'time'] if idx_ema3 is not None else None
            except Exception:
                ema3_first = None
            try:
                idx_ema50 = m['ema50'].first_valid_index()
                ema50_first = m.loc[idx_ema50, 'time'] if idx_ema50 is not None else None
            except Exception:
                ema50_first = None
            print(f"{sym}: ema3_first={ema3_first} ema50_first={ema50_first}")
        return m

    def run(self):
        # snapshot start-of-day equity for sizing
        self.start_equity = self.equity
        top10 = self.compute_turnover_top10()
        # for each symbol simulate through the day
        for sym in tqdm(top10, desc=f"{self.date} symbols"):
            df_min = self.symbol_dfs[sym]
            df = self.prepare_indicators(sym, df_min)
            self.simulate_symbol(sym, df)
        # after sim, compute performance
        trades_df = pd.DataFrame(self.trade_log)
        if not trades_df.empty and TRADE_LOG is not None:
            # ensure pnl column stored with 2 decimal places
            if "pnl" in trades_df.columns:
                trades_df["pnl"] = trades_df["pnl"].apply(lambda x: round(float(x), 2))
            trades_df.to_csv(TRADE_LOG, index=False, float_format='%.2f')
        metrics = self.compute_performance()
        return metrics

    def simulate_symbol(self, sym, df):
        position = None
        entry_price = None
        stop_loss = None
        target = None
        trail_active = False
        trail_stop = None
        # risk per trade is 0.5% of the start-of-day portfolio (portfolio-based risk)
        risk_amount = (self.start_equity) * RISK_PER_TRADE_PCT
        entered = False  # allow only single entry per stock per day

        # restrict simulation to the target date only (avoid trading on historical days)
        df_date = df[df['time'].dt.date == self.date].reset_index(drop=True)
        if df_date.empty:
            return
        # iterate by index so tqdm can show progress reliably
        for i in tqdm(range(len(df_date)), desc=f"Sim {self.date} {sym}", leave=False):
            row = df_date.iloc[i]
            t = row["time"]
            # only trade during regular intraday 09:25 onwards
            if t.time() < datetime.strptime("09:25", "%H:%M").time():
                continue

            # check entry if no position and not already entered today
            if position is None and not entered:
                # ensure indicators are present (use notnull)
                if (pd.notnull(row.get("ema3")) and pd.notnull(row.get("ema10")) and pd.notnull(row.get("rsi14")) and pd.notnull(row.get("ema50"))):
                    # long entry
                    cond_long = (row["ema3"] > row["ema10"] and row["rsi14"] > 60 and row["close"] > row["ema50"])
                    if cond_long:
                        if self.verbose:
                            print(f"{sym} {t} LONG cond met: ema3={row['ema3']:.3f} ema10={row['ema10']:.3f} rsi={row['rsi14']:.1f} close={row['close']:.2f} ema50={row['ema50']:.2f}")
                        entry_price = row["high"]
                        entry_price = round(float(entry_price), 2)
                        size = int(risk_amount / (entry_price * STOP_LOSS_PCT))
                        if size <= 0:
                            if self.verbose:
                                print(f"{sym} {t} sizing=0 risk_amount={risk_amount} entry_price={entry_price}")
                            continue
                        position = "LONG"
                        entered = True
                        stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                        target = entry_price * (1 + PROFIT_TARGET_PCT)
                        trail_active = False
                        trail_stop = None
                        self.trade_log.append({
                            "time": t,
                            "symbol": sym,
                            "side": position,
                            "entry_price": entry_price,
                            "size": size,
                            "status": "OPEN",
                        })
                        if self.verbose:
                            print(f"{sym} {t} ENTER LONG at {entry_price} size={size}")
                        continue
                    # short entry
                    cond_short = (row["ema3"] < row["ema10"] and row["rsi14"] < 30 and row["close"] < row["ema50"])
                    if cond_short:
                        if self.verbose:
                            print(f"{sym} {t} SHORT cond met: ema3={row['ema3']:.3f} ema10={row['ema10']:.3f} rsi={row['rsi14']:.1f} close={row['close']:.2f} ema50={row['ema50']:.2f}")
                        # use the per-day restricted frame (df_date) so we don't pick lows from older history
                        recent = df_date.iloc[max(0, i-4):i+1]
                        entry_price = recent["low"].min()
                        entry_price = round(float(entry_price), 2)
                        size = int(risk_amount / (entry_price * STOP_LOSS_PCT))
                        if size <= 0:
                            if self.verbose:
                                print(f"{sym} {t} sizing=0 risk_amount={risk_amount} entry_price={entry_price}")
                            continue
                        position = "SHORT"
                        entered = True
                        stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                        target = entry_price * (1 - PROFIT_TARGET_PCT)
                        trail_active = False
                        trail_stop = None
                        self.trade_log.append({
                            "time": t,
                            "symbol": sym,
                            "side": position,
                            "entry_price": entry_price,
                            "size": size,
                            "status": "OPEN",
                        })
                        if self.verbose:
                            print(f"{sym} {t} ENTER SHORT at {entry_price} size={size}")
                        continue


            # position exists - check exits
            if position is not None:
                last_trade = self.trade_log[-1]
                if position == "LONG":
                    # check stoploss hit
                    if row["low"] <= stop_loss:
                        exit_price = round(float(stop_loss), 2)
                        pnl = round((exit_price - entry_price) * last_trade["size"], 2)
                        self.equity += pnl
                        last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "SL"})
                        position = None
                        continue
                    # check target
                    if row["high"] >= target:
                        exit_price = round(float(target), 2)
                        pnl = round((exit_price - entry_price) * last_trade["size"], 2)
                        self.equity += pnl
                        last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "TG"})
                        position = None
                        continue
                    # trailing
                    if not trail_active and row["high"] >= entry_price * (1 + TRAIL_START_PCT):
                        trail_active = True
                        trail_stop = row["high"] * (1 - TRAIL_STEP_PCT)
                    if trail_active:
                        new_trail = row["high"] * (1 - TRAIL_STEP_PCT)
                        trail_stop = max(trail_stop, new_trail)
                        if row["low"] <= trail_stop:
                            exit_price = round(float(trail_stop), 2)
                            pnl = round((exit_price - entry_price) * last_trade["size"], 2)
                            self.equity += pnl
                            last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "TRAIL"})
                            position = None
                            continue
                else:  # SHORT
                    if row["high"] >= stop_loss:
                        exit_price = round(float(stop_loss), 2)
                        pnl = round((entry_price - exit_price) * last_trade["size"], 2)
                        self.equity += pnl
                        last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "SL"})
                        position = None
                        continue
                    if row["low"] <= target:
                        exit_price = round(float(target), 2)
                        pnl = round((entry_price - exit_price) * last_trade["size"], 2)
                        self.equity += pnl
                        last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "TG"})
                        position = None
                        continue
                    if not trail_active and row["low"] <= entry_price * (1 - TRAIL_START_PCT):
                        trail_active = True
                        trail_stop = row["low"] * (1 + TRAIL_STEP_PCT)
                    if trail_active:
                        new_trail = row["low"] * (1 + TRAIL_STEP_PCT)
                        trail_stop = min(trail_stop, new_trail)
                        if row["high"] >= trail_stop:
                            exit_price = round(float(trail_stop), 2)
                            pnl = round((entry_price - exit_price) * last_trade["size"], 2)
                            self.equity += pnl
                            last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "TRAIL"})
                            position = None
                            continue

            # EOD forced close at 15:29
            eod_time = datetime.strptime("15:29", "%H:%M").time()
            if position is not None and t.time() >= eod_time:
                exit_price = row.get("close", None)
                if exit_price is None:
                    exit_price = row.get("low") if position == "LONG" else row.get("high")
                exit_price = round(float(exit_price), 2)
                if position == "LONG":
                    pnl = round((exit_price - entry_price) * last_trade["size"], 2)
                else:
                    pnl = round((entry_price - exit_price) * last_trade["size"], 2)
                self.equity += pnl
                last_trade.update({"exit_time": t, "exit_price": exit_price, "pnl": pnl, "status": "CLOSED", "reason": "EOD"})
                position = None

            # record equity curve snapshot
            self.equity_curve.append({"time": t, "equity": self.equity})

    def compute_performance(self):
        trades = pd.DataFrame(self.trade_log)
        # handle case with no trades gracefully (still return metrics)
        if trades.empty:
            print("No trades executed")
            metrics = {
                "date": str(self.date),
                "trades": 0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "sharpe": None,
                "max_drawdown": 0.0,
                "eod_equity": float(self.equity),
                "net_pnl": float(self.equity - BASE_CAPITAL),
            }
            return metrics
        # process trades
        trades["pnl"] = trades["pnl"].astype(float)
        # compute total return relative to the day's starting NAV
        start_nav = getattr(self, 'start_equity', BASE_CAPITAL)
        total_return = trades["pnl"].sum() / start_nav if start_nav != 0 else 0.0
        wins = trades[trades["pnl"] > 0]
        win_rate = len(wins) / len(trades) * 100

        # equity curve
        eq = pd.DataFrame(self.equity_curve).drop_duplicates(subset=["time"]).set_index("time").sort_index()
        eq = eq[~eq.index.duplicated(keep='first')]
        # build minute-resolution equity series
        full_idx = pd.date_range(eq.index.min(), eq.index.max(), freq="1min")
        eq_series = eq["equity"].reindex(full_idx).ffill()
        returns = eq_series.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252*6.5*60) if returns.std() != 0 else np.nan
        # max drawdown
        roll_max = eq_series.cummax()
        drawdown = (eq_series - roll_max) / roll_max
        max_dd = drawdown.min()

        print("Trades:", len(trades))
        print(f"Total Return: {round(total_return*100,2):.2f}%")
        print(f"Win Rate: {round(win_rate,2):.2f}%")
        print(f"Sharpe (approx): {round(sharpe,2) if sharpe is not None and not np.isnan(sharpe) else 'nan'}")
        print(f"Max Drawdown: {round(max_dd*100,2):.2f}%")
        print(f"EOD Equity: {round(self.equity,2)} | Net PnL: {round(self.equity-start_nav,2)}")
        if TRADE_LOG is not None:
            print(f"Trade log written to {TRADE_LOG}")
        else:
            print("TRADE_LOG not set; not writing trade CSV")

        # build metrics dict to allow programmatic summaries
        metrics = {
            "date": str(self.date),
            "trades": int(len(trades)),
            "total_return": float(total_return),
            "win_rate": float(win_rate),
            "sharpe": float(sharpe) if not np.isnan(sharpe) else None,
            "max_drawdown": float(max_dd),
            "eod_equity": float(self.equity),
            "net_pnl": float(self.equity - start_nav),
        }
        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Intraday EMA-RSI backtester")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to folder with dataNSE_*.csv files")
    parser.add_argument("--date", default=None, help="Date to backtest (YYYY-MM-DD). If omitted uses earliest date in files")
    parser.add_argument("--verbose", action="store_true", help="Verbose debug prints for indicator availability and entries")
    parser.add_argument("--all", action="store_true", help="Run backtest for all dates found in data files")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    files = sorted(DATA_DIR.glob("dataNSE_*.csv"))
    if not files:
        # try fallback locations and provide debug info
        possible = [Path(__file__).parent / 'stock_data', Path(__file__).parent / 'stock_data_aug_2025', Path(__file__).parent / 'stock_data_jul_2025']
        for p in possible:
            f2 = sorted(p.glob('dataNSE_*.csv')) if p.exists() else []
            if f2:
                DATA_DIR = p
                files = f2
                print(f"Using fallback data dir: {DATA_DIR}")
                break
        if not files:
            print("No data files found in", DATA_DIR)
            print("Searched these locations:")
            print(" -", Path(args.data_dir))
            for p in possible:
                print(" -", p, "exists:" , p.exists())
    else:
        if args.all:
            # collect unique dates across files
            dates = set()
            for f in files:
                sample = pd.read_csv(f, parse_dates=["time"], usecols=["time"])
                dates.update(sample["time"].dt.date.unique().tolist())
            dates = sorted(dates)
            print(f"Found {len(dates)} dates. Processing all...")
            created = []
            summary_rows = []
            TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
            # carry NAV across days so each day's sizing and pct returns use prior day's EOD
            running_nav = BASE_CAPITAL
            for d in dates:
                TRADE_LOG = TRADE_LOG_DIR / f"trade_log_{d}.csv"
                bt = Backtester(str(d), start_equity=running_nav, verbose=args.verbose)
                metrics = bt.run()
                # compute daily percent return relative to that day's starting NAV
                start_nav = float(running_nav)
                eod = float(metrics.get("eod_equity", running_nav))
                pct = (eod - start_nav) / start_nav if start_nav != 0 else 0.0
                metrics["daily_pct"] = pct
                # add cumulative NAV column (EOD equity for plotting)
                metrics["cum_nav"] = eod
                # carry forward NAV to next day
                running_nav = eod
                created.append(TRADE_LOG)
                summary_rows.append(metrics)
            # combined daily summary
            # round numeric metrics to 2 decimals and write single daily_all summary
            summary_all = pd.DataFrame(summary_rows).fillna("")
            for col in ["total_return", "daily_pct", "win_rate", "sharpe", "max_drawdown", "eod_equity", "net_pnl", "cum_nav"]:
                if col in summary_all.columns:
                    summary_all[col] = pd.to_numeric(summary_all[col], errors='coerce').round(2)
            summary_all_path = TRADE_LOG_DIR / "summary_daily_all.csv"
            summary_all.to_csv(summary_all_path, index=False)
            print("Created trade logs:")
            for p in created:
                print(" -", p)
            print("Daily summary written to", summary_all_path)
        else:
            if args.date:
                date0 = pd.to_datetime(args.date).date()
            else:
                sample = pd.read_csv(files[0], parse_dates=["time"]) 
                date0 = sample["time"].dt.date.min()
            # set trade log path per date
            TRADE_LOG = Path(__file__).parent / f"trade_log_{date0}.csv"
            bt = Backtester(str(date0), verbose=args.verbose)
            bt.run()
