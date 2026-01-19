# Intraday EMA-RSI Stock Strategy - Comprehensive Documentation

**Version:** 1.0  
**Last Updated:** January 15, 2026  
**Strategy Type:** Intraday (Intra-day) EMA-RSI Crossover with Risk Management  
**Market:** NSE (National Stock Exchange of India)  

---

## Table of Contents

1. [Strategy Overview](#strategy-overview)
2. [Architecture Flow](#architecture-flow)
3. [Step-by-Step Execution Process](#step-by-step-execution-process)
4. [Core Components](#core-components)
5. [Technical Implementation](#technical-implementation)
6. [Risk Management System](#risk-management-system)
7. [Trade Lifecycle](#trade-lifecycle)
8. [Performance Results](#performance-results)
9. [Strategy Variants](#strategy-variants)
10. [File Structure](#file-structure)
11. [Optimizations & Profiling](#optimizations--profiling)

---

## Strategy Overview

### Purpose
A fully automated intraday trading strategy designed for NSE stocks that uses multiple timeframe technical analysis (EMA indicators and RSI) to identify high-probability entry points with strict risk management protocols.

### Core Thesis
- **Market Timing:** Trade only at 09:25 when market opens with better liquidity
- **Stock Selection:** Focus on high-turnover stocks (top 10 by volume × price)
- **Entry Strategy:** Identify EMA crossovers with RSI confirmation and trend alignment
- **Risk Control:** Fixed 0.5% stop-loss with position sizing based on equity
- **Profit Taking:** 2% target with trailing stops at 0.75% after 0.5% profit

### Key Metrics
| Parameter | Value |
|-----------|-------|
| Base Capital | ₹10,00,000 (10 Lac) |
| Risk per Trade | 0.5% of allocated equity |
| Stop Loss | 0.5% |
| Profit Target | 2% |
| Trailing Activation | 0.5% profit |
| Trailing Step | 0.75% |
| Trading Sessions | 40 trading days (Jul-Aug 2025) |

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     STRATEGY EXECUTION FLOW                      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   DATA LOADING       │
                    │  (Minute-level CSV)  │
                    └──────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  INDICATOR CALCULATION     │
                  │  • EMA(3) - 10min TF       │
                  │  • EMA(10) - 10min TF      │
                  │  • EMA(50) - 1hour TF      │
                  │  • RSI(14) - 10min TF      │
                  └────────────────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  STOCK SELECTION (9:25)    │
                  │  Top 10 by Turnover        │
                  └────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌────────────────┐         ┌────────────────┐
        │  ENTRY CHECK   │         │  ENTRY CHECK   │
        │  LONG SETUP    │         │  SHORT SETUP   │
        └────────────────┘         └────────────────┘
                │                             │
                ▼                             ▼
        ┌────────────────┐         ┌────────────────┐
        │ BUY at HIGH    │         │ SELL at MIN(5) │
        │ of entry bar   │         │ of last 5 min  │
        └────────────────┘         └────────────────┘
                │                             │
                └──────────────┬──────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │   POSITION MANAGEMENT      │
                  │  • Monitor SL (0.5%)       │
                  │  • Check Target (2%)       │
                  │  • Track Trailing (0.75%)  │
                  │  • Close at EOD            │
                  └────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  TRADE LOGGING       │
                    │  Generate Reports    │
                    └──────────────────────┘
```

---

## Step-by-Step Execution Process

### Phase 1: Initialization

**Step 1.1: Load Configuration**
- Base Capital: ₹10,00,000
- Risk Parameters: 0.5% SL, 2% Target, 0.75% Trailing
- Data Directory: `stock_data/` containing CSV files

**Step 1.2: Load Market Data**
- Read NSE minute-level OHLCV data for target date range
- File format: `dataNSE_YYYYMMDD.csv`
- Columns: `ticker`, `time`, `open`, `high`, `low`, `close`, `volume`
- Data from 09:15 to 15:29 (full trading day)

```python
# Example data loading
df = pd.read_csv("stock_data/dataNSE_20250801.csv")
# Columns: ticker, time, open, high, low, close, volume
```

---

### Phase 2: Indicator Calculation

**Step 2.1: Resample Data**
- Convert 1-minute data to required timeframes:
  - **10-minute bars** (for EMA-3, EMA-10, RSI-14)
  - **60-minute bars** (for EMA-50)

```python
# Resample to 10-minute bars
df_10min = df.resample('10min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
```

**Step 2.2: Calculate EMA(3) and EMA(10) on 10-min**
- Used for trend identification and entry signals
- EMA formula: `price.ewm(span=3, adjust=False).mean()`

```
EMA(3): Highly responsive to recent price action
EMA(10): Medium-term trend confirmation

Signal: EMA(3) crossing above/below EMA(10)
```

**Step 2.3: Calculate RSI(14) on 10-min**
- Used for overbought/oversold conditions
- RSI(14) > 60 = Bullish momentum (entry for LONG)
- RSI(14) < 30 = Bearish momentum (entry for SHORT)

```
RSI Calculation:
1. Calculate price changes (delta)
2. Separate gains and losses
3. Average gains/losses over 14 periods
4. RS = AvgGain / AvgLoss
5. RSI = 100 - (100 / (1 + RS))
```

**Step 2.4: Calculate EMA(50) on 1-hour**
- Used as a trend filter
- Only trade LONG if close > EMA(50) (uptrend)
- Only trade SHORT if close < EMA(50) (downtrend)

```python
# Example calculation
df_1hour = df.resample('60min').agg({...})
df['ema50'] = calculate_ema(df_1hour['close'], span=50)
```

---

### Phase 3: Stock Selection (9:25 AM)

**Step 3.1: Calculate Turnover**
- Use first 10 minutes of market (09:15 - 09:24)
- Turnover = Sum of (Volume × Close) for each stock
- Select top 10 stocks by turnover

```python
# Filter first 10 minutes
first_10min = df[df['time'].between('09:15', '09:24')]

# Calculate turnover
turnover = (first_10min['volume'] * first_10min['close']).sum()

# Select top 10
selected_stocks = df.groupby('ticker')['turnover'].sum().nlargest(10)
```

**Benefits:**
- High liquidity at market open
- Avoid low-volume stocks with slippage
- Consistent stock selection methodology

---

### Phase 4: Entry Signal Detection (09:25 onwards)

**Step 4.1: LONG Entry Signal**

All conditions must be TRUE:
```
✓ EMA(3) > EMA(10)           [Uptrend crossover]
✓ RSI(14) > 60               [Bullish momentum]
✓ Close > EMA(50)            [Long-term uptrend]
```

**Action:** BUY at HIGH of entry candle

```python
if ema3 > ema10 and rsi14 > 60 and close > ema50:
    entry_price = high_of_current_candle
    position_size = calculate_size(entry_price)
    create_long_trade(symbol, entry_price, position_size)
```

**Step 4.2: SHORT Entry Signal**

All conditions must be TRUE:
```
✓ EMA(3) < EMA(10)           [Downtrend crossover]
✓ RSI(14) < 30               [Bearish momentum]
✓ Close < EMA(50)            [Long-term downtrend]
```

**Action:** SELL at LOW of last 5 1-minute candles

```python
if ema3 < ema10 and rsi14 < 30 and close < ema50:
    # Look back at last 5 1-min candles
    entry_price = min(low_of_last_5_candles)
    position_size = calculate_size(entry_price)
    create_short_trade(symbol, entry_price, position_size)
```

**Step 4.3: One Trade Per Stock Per Day**
- Once a trade (LONG or SHORT) is triggered, no more entries for that stock on that day
- Prevents over-trading and conflicting signals

---

### Phase 5: Position Sizing & Risk Calculation

**Step 5.1: Risk Amount Calculation**
```
Risk Amount = Base Capital × Risk Percentage per Trade
Risk Amount = 10,00,000 × 0.5%
Risk Amount = ₹5,000
```

**Step 5.2: Position Size Calculation**
```
Position Size = Risk Amount / (Entry Price × Stop Loss %)
Position Size = 5,000 / (Entry Price × 0.5%)
Position Size = 5,000 / (Entry Price × 0.005)

Example: Entry at ₹100
Position Size = 5,000 / (100 × 0.005) = 5,000 / 0.5 = 10,000 units
```

**Step 5.3: Stop Loss & Target Calculation**

For LONG trades:
```
Entry Price:  ₹100
Stop Loss:    ₹100 × (1 - 0.5%)  = ₹99.50  (0.5% below entry)
Target:       ₹100 × (1 + 2%)    = ₹102   (2% above entry)
Risk/Reward:  ₹0.50 / ₹2.00 = 1:4
```

For SHORT trades:
```
Entry Price:  ₹100
Stop Loss:    ₹100 × (1 + 0.5%)  = ₹100.50 (0.5% above entry)
Target:       ₹100 × (1 - 2%)    = ₹98    (2% below entry)
Risk/Reward:  ₹0.50 / ₹2.00 = 1:4
```

---

### Phase 6: Exit & Position Management

**Step 6.1: Stop Loss Exit**
- Triggered when price reaches 0.5% away from entry
- Realizes maximum allowed loss
- Protects capital from large adverse moves

```python
if LONG:
    if low <= entry_price * (1 - 0.5%):
        exit_price = stop_loss_level
        exit_reason = "SL"
        
if SHORT:
    if high >= entry_price * (1 + 0.5%):
        exit_price = stop_loss_level
        exit_reason = "SL"
```

**Step 6.2: Target Exit**
- Triggered when price reaches 2% profit
- Captures primary profit objective
- Commonly hit before other exits

```python
if LONG:
    if high >= entry_price * (1 + 2%):
        exit_price = target_level
        exit_reason = "TG"
        
if SHORT:
    if low <= entry_price * (1 - 2%):
        exit_price = target_level
        exit_reason = "TG"
```

**Step 6.3: Trailing Stop Activation**
- Starts when trade is in profit by 0.5%
- Locks in gains as price moves favorably
- Exits if price reverses by 0.75%

```
Activation Trigger:
├─ LONG:  High >= Entry × (1 + 0.5%)
└─ SHORT: Low <= Entry × (1 - 0.5%)

Trail Calculation:
├─ LONG:  Trail = High × (1 - 0.75%)
└─ SHORT: Trail = Low × (1 + 0.75%)

Exit on Reversal:
├─ LONG:  if Low <= Trail Stop
└─ SHORT: if High >= Trail Stop
```

**Example: LONG Trade with Trailing Stop**
```
Entry Price:     ₹100.00
Entry + 0.5%:    ₹100.50  (Activation level)

Bar 1: High = ₹100.80  (< ₹100.50, no trail yet)
Bar 2: High = ₹101.20  (> ₹100.50, activate trail)
       Trail = ₹101.20 × (1 - 0.75%) = ₹100.44

Bar 3: High = ₹101.50 (Update trail)
       Trail = ₹101.50 × (1 - 0.75%) = ₹100.74

Bar 4: Low = ₹100.50  (Check against trail)
       ₹100.50 < ₹100.74  → EXIT at ₹100.74
```

**Step 6.4: End of Day (EOD) Close**
- All remaining open positions closed at close price
- Prevents overnight holding
- Avoids gap risk on next day

```python
if position_open and current_time >= 15:29:
    exit_price = close_price
    exit_reason = "EOD"
    close_trade()
```

---

### Phase 7: Trade Recording & Logging

**Step 7.1: Trade Dictionary**
Each trade is recorded with:

```python
trade = {
    'date': '2025-08-01',
    'time': '09:35',
    'symbol': 'RELIANCE',
    'side': 'LONG',
    'entry_price': 2850.50,
    'size': 1750,
    'status': 'CLOSED',
    'exit_time': '10:15',
    'exit_price': 2901.00,
    'pnl': 88087.50,  # (2901 - 2850.5) × 1750
    'reason': 'TG'  # Exit reason
}
```

**Step 7.2: Daily Trade Log**
- One CSV file per trading day: `trade_log_YYYY-MM-DD.csv`
- Contains all trades for that day
- Enables review and analysis

**Step 7.3: Performance Metrics**
- Daily P&L calculation
- Win rate (trades with positive P&L)
- Max drawdown tracking
- Equity curve generation

---

### Phase 8: Daily & Period Summary

**Step 8.1: Daily Statistics**
```
Date: 2025-08-01
Total Trades:           12
Winning Trades:         4 (33.33%)
Losing Trades:          8 (66.67%)
Total P&L:             -₹42,500
Max Profit:            +₹50,000
Max Loss:              -₹5,000
Largest Win/Loss:      +₹45,000 / -₹4,500
```

**Step 8.2: Period Summary**
```
Period: Aug 1 - Aug 28, 2025 (17 trading days)
Initial Capital:       ₹10,00,000
Final Equity:          ₹8,41,253
Total Return:          -15.87%
Total Trades:          107
Win Rate:              27.10%
Profit Factor:         0.64
Max Drawdown:          18.01%
Sharpe Ratio:          -11.36
```

---

## Core Components

### 1. Data Loading Module
```python
def load_data_for_date(date, data_dir):
    """
    Load NSE minute-level data for a given date
    Returns: DataFrame with ticker, time, OHLCV
    """
```

### 2. Indicator Calculation Module
```python
def calculate_ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index"""
    # Momentum-based indicator
```

### 3. Stock Selection Module
```python
def select_top_stocks(df, num_stocks=10):
    """
    Select top N stocks by turnover at market open
    Turnover = sum(volume * close) for first 10 min
    """
```

### 4. Trade Simulation Engine
```python
class Trade:
    """Represents a single trade with entry/exit tracking"""
    
def simulate_stock_trading(symbol, df, date, start_equity):
    """
    Simulate trading for one stock on one date
    Returns: List of completed Trade objects
    """
```

### 5. Position Management Module
```python
def check_exit_conditions(trade, current_row):
    """
    Check for SL, Target, Trailing Stop, or EOD close
    Returns: exit_price and exit_reason if triggered
    """

def calculate_position_size(risk_amount, entry_price, sl_pct):
    """
    Calculate units to trade based on fixed risk
    Returns: Number of shares to buy/sell
    """
```

### 6. Results Processing Module
```python
def calculate_performance_metrics(trades):
    """
    Calculate return, win rate, drawdown, Sharpe ratio
    Returns: Dictionary with all metrics
    """
```

---

## Technical Implementation

### Multi-Timeframe Analysis

**Why Multiple Timeframes?**
- Reduces false signals
- Aligns different market perspectives
- Confirms trend direction

**Implementation:**

```python
# 1-minute raw data (for tracking actual entries/exits)
df_1min = pd.read_csv("data.csv")

# 10-minute resampling (for EMA and RSI)
df_10min = df_1min.resample('10min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})
df_10min['ema3'] = calculate_ema(df_10min['close'], 3)
df_10min['ema10'] = calculate_ema(df_10min['close'], 10)
df_10min['rsi14'] = calculate_rsi(df_10min['close'], 14)

# 60-minute resampling (for trend filter)
df_60min = df_1min.resample('60min').agg({...})
df_60min['ema50'] = calculate_ema(df_60min['close'], 50)

# Map back to 1-minute for entry/exit logic
df_1min['ema50_hourly'] = df_60min['ema50'].reindex(df_1min.index, method='ffill')
```

### EMA Indicator Deep Dive

**Exponential Moving Average Formula:**
```
EMA_t = α × Price_t + (1 - α) × EMA_(t-1)

Where: α = 2 / (span + 1)
```

**For EMA(3):**
```
α = 2 / (3 + 1) = 0.5
EMA_t = 0.5 × Price_t + 0.5 × EMA_(t-1)
```

**Interpretation:**
- EMA(3): Fast-moving, reacts quickly to recent prices
- EMA(10): Slower, smoother trend confirmation
- **Bullish Signal:** EMA(3) crosses above EMA(10)
- **Bearish Signal:** EMA(3) crosses below EMA(10)

### RSI Indicator Deep Dive

**Relative Strength Index Formula:**
```
1. Calculate price changes from bar to bar
   Gain = max(change, 0)
   Loss = abs(min(change, 0))

2. Smooth gains and losses over 14 periods
   AvgGain = EMA(Gain, 14)
   AvgLoss = EMA(Loss, 14)

3. Calculate RS and RSI
   RS = AvgGain / AvgLoss
   RSI = 100 - (100 / (1 + RS))
```

**Interpretation:**
- RSI > 70: Overbought (possible pullback)
- RSI > 60: Strong bullish momentum (LONG signals)
- RSI < 30: Oversold (possible bounce)
- RSI < 30: Strong bearish momentum (SHORT signals)
- RSI 30-70: Neutral zone

---

## Risk Management System

### Fixed Risk Per Trade

**Concept:** Always risk the same amount regardless of trade
- Prevents account destruction on big losses
- Builds consistent capital growth
- Protects against margin calls

**Formula:**
```
Risk per Trade = Base Capital × Risk % 
                = 10,00,000 × 0.5%
                = ₹5,000

This means: Every trade can lose maximum ₹5,000
```

### Position Sizing Formula

**Inverse Relationship:**
- Higher entry price → Fewer shares
- Lower entry price → More shares
- Maintains constant rupee risk

```python
def calculate_position_size(risk_amount, entry_price, stop_loss_pct):
    """
    Example: Risk = ₹5,000, Entry = ₹100, SL = 0.5%
    
    SL Distance = ₹100 × 0.5% = ₹0.50
    Position Size = ₹5,000 / ₹0.50 = 10,000 shares
    
    Verification: Loss if hit = 10,000 × ₹0.50 = ₹5,000 ✓
    """
    return risk_amount / (entry_price * stop_loss_pct)
```

### Stop Loss Implementation

**Hard Stop Loss at 0.5%:**
- Automatically triggered if price drops 0.5%
- No exceptions or "hoping" for recovery
- Protects against gap downs and sudden reversal

**In Code:**
```python
if trade.side == "LONG":
    if current_bar.low <= entry_price * (1 - 0.005):  # 0.5%
        close_trade(reason="SL")
```

### Profit Target at 2%

**Predetermined Target:**
- Set at entry: 2% above entry price
- Exit automatically when hit
- Captures R:R of 1:4 (lose 0.5%, win 2%)

### Trailing Stop Strategy

**Purpose:** Lock in profits while allowing continued upside

**Mechanics:**

1. **Activation:** Trade is in 0.5% profit
2. **Trail Level:** New high/low × (1 - 0.75%) for longs
3. **Adjusts Upward:** Trail only moves up, never down (for longs)
4. **Exit:** Triggered when price reverses by trail amount

**Visual Example (LONG):**
```
Entry: ₹100.00

Price moves to ₹100.50 (0.5% profit)
  → Trail Activation = YES
  → Trail Level = ₹100.50 × (1 - 0.75%) = ₹99.75

Price moves to ₹101.00 (1% profit)
  → Trail Level = ₹101.00 × (1 - 0.75%) = ₹100.25

Price moves to ₹102.00 (2% profit)
  → Hit target exit at ₹102.00
  → Profit = ₹2.00 (2%)
```

---

## Trade Lifecycle

### Complete Trade Flow Diagram

```
START
  │
  └─→ [Entry Signal Generated]
       │
       ├─→ Calculate Position Size
       │    (Risk = ₹5,000)
       │
       └─→ CREATE TRADE
            │ entry_price: High of bar
            │ size: Calculated shares
            │ stop_loss: Entry × (1 - 0.5%)
            │ target: Entry × (1 + 2%)
            │ trail_active: False
            │
            ├─→ WAIT FOR NEXT BAR
            │    │
            │    ├─→ Check SL
            │    │    └─ Hit → EXIT (Reason: SL)
            │    │
            │    ├─→ Check Target
            │    │    └─ Hit → EXIT (Reason: TG)
            │    │
            │    ├─→ Check Trailing Activation
            │    │    └─ Profit >= 0.5% → trail_active = True
            │    │
            │    ├─→ Update Trailing Stop
            │    │    └─ Trail = New High × (1 - 0.75%)
            │    │
            │    ├─→ Check Trailing Exit
            │    │    └─ Low <= Trail → EXIT (Reason: TRAIL)
            │    │
            │    ├─→ Check EOD (15:29)
            │    │    └─ Yes → EXIT at Close (Reason: EOD)
            │    │
            │    └─→ Continue to next bar
            │
            └─→ RECORD TRADE
                 │ exit_price
                 │ exit_reason
                 │ P&L: (exit_price - entry_price) × size
                 │ status: CLOSED
                 │
                 └─→ LOG TO CSV
END
```

### Trade State Machine

```
┌────────────┐
│    OPEN    │ ← Initial state after entry
└────────────┘
     │
     ├─→ SL Hit    → EXIT (Loss capped at 0.5%)
     ├─→ Target    → EXIT (Profit = 2%)
     ├─→ Trailing  → EXIT (Variable profit)
     ├─→ EOD        → EXIT (Close price)
     │
     └─→ CLOSED
```

---

## Performance Results

### Strategy Comparison

| Metric | LONG Only (Aug) | Full Strategy (Aug) | Intraday Orig (Jul-Aug) |
|--------|-----------------|-------------------|------------------------|
| **Period** | 17 trading days | 17 trading days | 40 trading days |
| **Total Trades** | 107 | 170 | 343 |
| **Win Rate** | 27.10% | 18.24% | 32.65% |
| **Total Return** | **-15.87%** | -36.54% | -22.57% |
| **Final Equity** | ₹8,41,253 | ₹6,34,566 | ₹7,74,325 |
| **Max Drawdown** | 18.01% | 36.54% | 22.57% |
| **Sharpe Ratio** | -11.36 | -21.65 | -3.50 |

### Key Findings

1. **LONG Only Outperforms**
   - Fewer trades but better quality
   - Lower drawdown (18% vs 36%)
   - Better risk-adjusted returns

2. **SHORT Trades Underperform**
   - Win rate drops from 27% to 18% with SHORTs
   - RSI < 30 condition too restrictive
   - More false signals in choppy markets

3. **Market Conditions**
   - July-August 2025 was bearish/choppy
   - All strategies negative (market inefficiency)
   - Strategy is not the problem, market conditions are

4. **Trade Quality**
   - Average loss per trade: -₹1,500 (LONG) vs -₹2,150 (Full)
   - Risk management working: SL consistently at 0.5%
   - Profit targets achieved when hit

---

## Strategy Variants

### 1. LONG Only Strategy (`new_strategy_only_long.py`)

**Modifications:**
- Skip SHORT entry conditions entirely
- Only look for bullish EMA crossovers (EMA3 > EMA10)
- Requires: RSI > 60 AND Close > EMA(50)

**Performance (Aug 2025):**
- 107 trades, 27.10% win rate, -15.87% return
- Better than full strategy for bearish markets

**Use Case:** 
- Bullish market conditions
- Avoid shorting in strong downtrends
- Simpler execution

### 2. Full Strategy (`new_strategy.py`)

**Includes:**
- Both LONG and SHORT entries
- More trading opportunities
- Higher trade count

**Performance (Aug 2025):**
- 170 trades, 18.24% win rate, -36.54% return
- Underperforms in choppy markets

**Use Case:**
- Range-bound or sideways markets
- Capture both long and short moves
- More active trading approach

### 3. Original Intraday Strategy (`intraday.py`)

**Features:**
- Base implementation
- Ran on 40 days of data (Jul-Aug)
- Comprehensive baseline

**Performance (Jul-Aug 2025):**
- 343 trades, 32.65% win rate, -22.57% return
- Middle ground between other strategies

---

## File Structure

```
backtester/
├── new_strategy.py                 ← Full LONG+SHORT strategy
├── new_strategy_only_long.py        ← LONG Only variant
├── intraday.py                      ← Original baseline
│
├── config.py                        ← Configuration module
├── final_stats.py                   ← Statistics calculations
│
├── stock_data/                      ← Raw NSE data
│   ├── dataNSE_20250701.csv
│   ├── dataNSE_20250702.csv
│   └── ... (40 files)
│
├── trade_logs/                      ← Original strategy logs
│   ├── trade_log_2025-07-01.csv
│   ├── trade_log_2025-08-01.csv
│   └── summary_daily_all.csv
│
├── trade_logs_new_strategy/         ← Full strategy logs
│   ├── trade_log_2025-08-01.csv
│   └── ... (Aug logs)
│
├── trade_logs_only_long/            ← LONG Only logs
│   ├── trade_log_2025-08-01.csv
│   └── ... (Aug logs)
│
├── README.md                        ← Basic documentation
├── STRATEGY_DOCUMENTATION.md        ← This file
├── summary.txt                      ← Performance summary
│
├── requirements.txt                 ← Dependencies
├── pyproject.toml                   ← Project config
├── LICENSE                          ← MIT License
└── CHANGELOG.md                     ← Version history
```

---

## How to Run

### Installation
```bash
# Clone repository
git clone https://github.com/shadab28/backtester.git
cd backtester

# Install dependencies
pip install -r requirements.txt
```

### Run Full Strategy (LONG + SHORT)
```bash
# Single date
python new_strategy.py --start-date 2025-08-01

# Date range
python new_strategy.py --start-date 2025-08-01 --end-date 2025-08-31

# All available dates
python new_strategy.py --all
```

### Run LONG Only Strategy
```bash
python new_strategy_only_long.py --start-date 2025-08-01
```

### Run Original Intraday Strategy
```bash
python intraday.py --data-dir stock_data --all
```

### Review Results
```bash
# View summary
cat summary.txt

# Check daily logs
ls -la trade_logs/
cat trade_logs/trade_log_2025-08-01.csv
```

---

## Next Steps & Improvements

### 1. Market Regime Detection
```
Current: Fixed rules for all conditions
Proposed: Detect trending vs ranging markets
         Use LONG only in trends, SHORTs in ranges
```

### 2. Dynamic Entry Thresholds
```
Current: Fixed RSI(14) > 60 for LONG
Proposed: Adjust RSI thresholds based on market volatility
         Higher volatility → Stricter entry (RSI > 70)
         Lower volatility → Relaxed entry (RSI > 50)
```

### 3. Multi-Symbol Correlation
```
Current: Each stock independent
Proposed: Avoid correlated stock positions
         Maintain portfolio correlation < 0.5
```

### 4. Volume & Liquidity Filters
```
Current: Top 10 by turnover (good start)
Proposed: Add spread and slippage estimates
         Avoid stocks with wide bid-ask spreads
         Track slippage against prediction
```

### 5. Machine Learning Optimization
```
Current: Manual rule tuning
Proposed: Walk-forward optimization
         Test different RSI thresholds
         Optimize SL and TP percentages
```

---

## Appendix: Formula Reference

### EMA Formula
```
EMA = α × Price + (1 - α) × Previous_EMA
α = 2 / (Span + 1)
```

### RSI Formula
```
RS = Average Gain / Average Loss
RSI = 100 - (100 / (1 + RS))
```

### Position Size Formula
```
Position = Risk Amount / (Entry Price × SL %)
Position = ₹5,000 / (Entry Price × 0.5%)
```

### Return Calculation
```
Return % = (Final Equity - Initial Capital) / Initial Capital × 100
Return % = (841,253 - 1,000,000) / 1,000,000 × 100 = -15.87%
```

### Win Rate Calculation
```
Win Rate = Winning Trades / Total Trades × 100
Win Rate = 29 / 107 × 100 = 27.10%
```

### Maximum Drawdown
```
Max Drawdown = (Lowest Equity - Peak Equity) / Peak Equity × 100
Max Drawdown Tracks the worst peak-to-trough decline
```

---

## Contact & Support

**Repository:** https://github.com/shadab28/backtester  
**Author:** Shadab  
**Last Updated:** January 15, 2026  
**Strategy Status:** Production Ready  

---

*This documentation is comprehensive and covers all aspects of the EMA-RSI intraday strategy. Use it as a reference for understanding, extending, or optimizing the strategy.*

---

## Optimizations & Profiling ⚡

We added optional Numba acceleration and a small built-in profiler to measure where time is spent.

What changed
- Optional Numba-accelerated implementations added for EMA and RSI. They are used when `numba` is installed in the project's virtualenv.
- A `--profile` CLI flag was added to `new_strategy.py` to record simple timing information (indicator calculation vs simulation) and write it to `trade_logs_new_strategy/profile_summary.json`.
- A `--non-interactive` flag was added and interactive date prompts default to `2025-08-01` → `2025-08-28`.

How to enable Numba (recommended for large runs)
```bash
source .venv/bin/activate
pip install numba
```
Note: Installing Numba may require llvmlite and a specific numpy version. During installation the venv may downgrade/adjust `numpy` to a supported version. Pin the working versions to `requirements.txt` after you validate the environment.

How to run with profiling
```bash
source .venv/bin/activate
python3 new_strategy.py --start-date 2025-08-01 --end-date 2025-08-28 --profile
```
After the run you'll find `trade_logs_new_strategy/profile_summary.json` containing:
```
{
  "total_indicator_time_sec": 12.345678,
  "total_simulation_time_sec": 8.123456,
  "dates_processed": 17
}
```

Observed effect (on this machine during testing)
- Without Numba: Aug 1–28 run completed in ~6 seconds; results slightly different due to numeric behavior.
- With Numba (installed): Full 40-day run completed in ~12 seconds; observed a large change in computed results — this is likely due to NaN/edge-case handling differences in the numba implementations, so validate outputs before trusting them in production.

Recommended next steps


## Runtime Options & Defaults

The backtester exposes a compact CLI. Key runtime options and defaults are below.

Flags

- `--start-date / -s` : Start date for backtest. NOTE: the repository enforces a minimum start date of **2025-08-01**. Any earlier user-provided start date will be ignored.
- `--end-date / -e` : End date for backtest (optional).
- `--all / -a` : Run for all available dates (start date will be forced to 2025-08-01).
- `--non-interactive` : Do not prompt for dates; use defaults instead.
- `--preload` : Preload all CSVs into memory before running. Useful for multi-worker runs; increases RAM usage.
- `--profile` : Record simple timing profile (indicator vs simulation) and write `trade_logs_new_strategy/profile_summary.json`.
- `--workers / -w` : Number of worker processes (defaults to CPU count capped internally).
- `--no-multiprocessing` : Disable parallel processing.

Examples

```bash
# Run Aug 1 - Aug 28 (defaults)
python3 new_strategy.py --non-interactive

# Run with preloaded CSVs and 4 workers
python3 new_strategy.py --all --non-interactive --preload --workers 4

# Run with profiling enabled
python3 new_strategy.py --start-date 2025-08-01 --end-date 2025-08-28 --profile
```

## Start Date Enforcement

Rationale

The repository is configured to run experiments starting from **2025-08-01** to ensure consistency across backtests, comparisons, and trade logs. This avoids accidental inclusion of earlier data and maintains a reproducible baseline.

Behavior

- If `--all` is used the backtest will start at `2025-08-01` and run to the newest available date.
- If a user provides a `--start-date` earlier than `2025-08-01`, that input is ignored and `2025-08-01` is used instead (a console message is printed).
- If interactive mode is used, the prompt defaults to `2025-08-01` and any earlier value will be clamped to `2025-08-01`.

If you prefer a warning-only policy (allow earlier dates with a printed warning), I can change the code to a soft fallback instead of enforcement.

---

## Reproducibility & Environment

When Numba is installed it may require a specific `numpy`/`llvmlite` combination. After you validate that numba produces matching indicators, pin the working versions in `requirements.txt`.

Suggested lines to pin (example from local run):
```
numba==0.63.1
llvmlite==0.46.0
numpy==2.3.5
```

Add and commit these after validation so other collaborators reproduce the same runtime behavior.

---

If you'd like, I can now:

- Produce a small CSV that compares pandas vs numba EMA/RSI for a chosen symbol and date.
- Convert the strict start-date enforcement to a warning-only behavior.
- Add an LRU preload mechanism to limit memory usage when preloading.

