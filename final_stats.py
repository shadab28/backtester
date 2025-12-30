"""Aggregate all trade logs and produce final performance metrics.

Deliverables produced by this script:
 - Reads per-day trade logs in `trade_logs/` (files named trade_log_YYYY-MM-DD.csv)
 - Generates an aggregated trade-level CSV and a small final-stats CSV in `trade_logs/`
 - Prints Performance Metrics: Return, Max Drawdown, Win rate %, Sharpe Ratio (daily if daily summary exists)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import intraday


DATA_DIR = Path(__file__).parent
TRADE_LOG_DIR = DATA_DIR / "trade_logs"


def read_trades(trade_dir: Path):
    files = sorted(trade_dir.glob("trade_log_*.csv"))
    if not files:
        raise SystemExit(f"No trade_log_*.csv files found in {trade_dir}")
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, parse_dates=["time", "exit_time"]) 
            dfs.append(d)
        except Exception:
            continue
    trades = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return trades


def read_daily_summary(trade_dir: Path):
    # prefer summary_daily_all_from_start.csv then summary_daily_all.csv
    for name in ("summary_daily_all_from_start.csv", "summary_daily_all.csv"):
        p = trade_dir / name
        if p.exists():
            try:
                return pd.read_csv(p, parse_dates=["date"]) 
            except Exception:
                continue
    return None


def compute_metrics(trades: pd.DataFrame, daily: pd.DataFrame | None):
    start_nav = float(getattr(intraday, 'BASE_CAPITAL', 1000000))
    total_trades = len(trades)
    wins = trades[trades['pnl'] > 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
    net_pnl = trades['pnl'].sum() if total_trades > 0 else 0.0

    # final NAV
    if daily is not None and 'cum_nav' in daily.columns:
        final_nav = float(daily['cum_nav'].iloc[-1])
    else:
        final_nav = start_nav + net_pnl

    total_return = (final_nav - start_nav) / start_nav if start_nav != 0 else np.nan

    # drawdown: prefer daily cum_nav series
    if daily is not None and 'cum_nav' in daily.columns:
        series = pd.Series(daily['cum_nav'].astype(float).values)
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        max_dd = float(drawdown.min())
    else:
        # fallback: compute equity path from trade exit times aggregated chronologically
        if total_trades == 0:
            max_dd = 0.0
        else:
            t = trades.sort_values('exit_time')
            eq = (t['pnl'].cumsum() + start_nav).rename('equity')
            roll_max = eq.cummax()
            drawdown = (eq - roll_max) / roll_max
            max_dd = float(drawdown.min())

    # Sharpe: prefer daily_pct from daily summary (assumed daily returns), else None
    sharpe = None
    if daily is not None and 'daily_pct' in daily.columns:
        daily_returns = pd.to_numeric(daily['daily_pct'], errors='coerce').dropna()
        if len(daily_returns) > 1 and daily_returns.std() != 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        # fallback: compute trade-level Sharpe on trade returns relative to start_nav
        if total_trades > 1:
            tr = (trades['pnl'] / start_nav).dropna()
            if tr.std() != 0:
                sharpe = float(tr.mean() / tr.std() * np.sqrt(len(tr)))

    metrics = {
        'start_nav': round(start_nav, 2),
        'final_nav': round(final_nav, 2),
        'net_pnl': round(net_pnl, 2),
        'total_return_pct': round(total_return * 100, 4) if not pd.isna(total_return) else None,
        'max_drawdown_pct': round(max_dd * 100, 4),
        'total_trades': int(total_trades),
        'win_rate_pct': round(win_rate, 2),
        'sharpe': round(sharpe, 4) if sharpe is not None else None,
    }
    return metrics


def main():
    trades = read_trades(TRADE_LOG_DIR)
    daily = read_daily_summary(TRADE_LOG_DIR)

    if trades.empty:
        print('No trades found')
        return

    # ensure numeric
    trades['pnl'] = pd.to_numeric(trades['pnl'], errors='coerce').fillna(0.0)

    metrics = compute_metrics(trades, daily)

    # write aggregated trades and final metrics
    out_agg = TRADE_LOG_DIR / 'all_trades_aggregated.csv'
    trades.to_csv(out_agg, index=False, float_format='%.2f')

    out_metrics = TRADE_LOG_DIR / 'final_stats.csv'
    pd.DataFrame([metrics]).to_csv(out_metrics, index=False)

    # print short report
    print('Final performance metrics:')
    for k, v in metrics.items():
        print(f' - {k}: {v}')
    print('\nAggregated trades written to', out_agg)
    print('Final metrics written to', out_metrics)


if __name__ == '__main__':
    main()
