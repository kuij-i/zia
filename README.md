# zia

Zia is a risk-governed Forex trading agent. It trades the **OANDA practice (paper) account** by default:
a deterministic technical strategy proposes trades, Claude reviews each one as an advisory second
opinion, and an independent **Risk Governor** has the final say before any order reaches the broker.

> **Risk notice.** This is experimental software. Forex trading carries substantial risk of loss.
> Backtest results are hypothetical and are not evidence of future profitability. Live trading is
> disabled by default and was not exercised during development.

## How a decision is made

```
completed candles ─► strategy signal ─► LLM review ─► Risk Governor ─► approved order ─► broker ─► verified result
      (OANDA)        (EMA/RSI/ATR)     (approve/reject)   (final say)    (with SL + TP)            │
                                                                                                   ▼
                                         every step is recorded in the SQLite journal ◄────────────┘
```

| Stage | Module | May do | May not do |
|---|---|---|---|
| Market data | `zia/broker/*` | fetch candles, prices, account | — |
| Strategy | `zia/strategy.py` | propose BUY/SELL with ATR stop/target *distances* | touch broker, network, DB, LLM, risk |
| LLM review | `zia/llm_reviewer.py` | `approve`/`reject` + confidence + rationale | size, stops, new trades, orders, credentials |
| Risk Governor | `zia/risk.py` | size the position, build the order, veto anything | be overridden by the LLM |
| Execution | `zia/broker/oanda.py` | send the order, verify fill + SL/TP actually exist | accept anything but a governor-approved order |
| Journal | `zia/journal.py` | persist evaluations, verdicts, checks, orders, fills, P&L | store secrets |

**The Risk Governor checks** practice/live environment consistency (and live confirmation), the kill switch,
the LLM verdict (if review is enabled), equity, max daily loss, max drawdown, max open trades, one position
per pair, whether the instrument is tradeable, price sanity and freshness, max spread (in pips), the trading
session (optional), that a stop-loss and take-profit are present and on the correct side of entry, the minimum
stop distance, minimum reward:risk, currency conversion, position size and margin. Only the governor can
construct an `ApprovedOrder`, and brokers accept nothing else.

**Position sizing** is deterministic: `units = floor(equity × risk% ÷ (|entry − stop| × quote→account rate))`.
JPY pairs use a 0.01 pip and convert yen losses to the account currency, so USD_JPY is sized correctly. The
default of 1% per trade is a development setting, not a recommendation.

**Fail-closed behavior:**
- If the LLM times out, errors, refuses, truncates or returns malformed output, the trade is rejected.
- If the configuration is ambiguous, Zia refuses to start.
- If a fill arrives without a stop-loss or take-profit, the trade is closed immediately and flagged.
- An HTTP 201 without a fill transaction does not count as a success.

**Candles:** Zia acts only on completed candles. It records the last processed candle per pair in SQLite,
before it acts, so a restart never trades the same candle twice.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your OANDA *practice* token/account and Anthropic key
```

Get a free practice account and API token from OANDA ("Manage API Access" in the fxTrade Practice portal).

## Usage

```bash
zia run --once          # one decision cycle against the practice account
zia run                 # loop: wakes shortly after each candle closes
zia status              # mode, account, positions, recent journal entries
zia close-all           # close every open position (asks for confirmation)

zia backtest --synthetic 3000                       # offline demo on seeded random-walk data (NOT market data)
zia backtest --synthetic 3000 --slippage-pips 0.5 --commission 3.5   # with explicit trading costs
zia backtest --synthetic 3000 --swap-long -2.5 --swap-short 0.8        # with financing (annual %)
zia backtest --pair EUR_USD --from 2025-01-01       # OANDA historical candles (needs credentials)
zia backtest --pair USD_JPY --csv data.csv          # CSV: time,open,high,low,close[,volume]
zia backtest --pair EUR_USD --from 2025-01-01 --llm # also call the LLM reviewer per signal (costs API calls)

# Several pairs on one shared account (risk limits apply across the portfolio, as in `zia run`)
zia backtest --pairs EUR_USD,GBP_USD,USD_JPY --from 2025-01-01
zia backtest --csv EUR_USD=eur.csv --csv USD_JPY=jpy.csv
zia backtest --pairs EUR_USD,GBP_USD,USD_JPY --synthetic 3000
```

**Kill switch:** create a file named `ZIA_KILL` in the working directory, or set `ZIA_KILL_SWITCH=true`. New
trades stop immediately. Existing positions keep their broker-side stop-loss and take-profit.

**Backtest costs and assumptions:**
- **Spread** (`--spread-pips`, default 1.0): buys fill at the ask, sells at the bid, and exits trigger on the opposite side.
- **Slippage** (`--slippage-pips`, default 0.2): a fixed number of pips against you on market entries, manual closes and stop-loss exits. Take-profits are limit orders and fill at their price.
- **Commission** (`--commission`, default 0): charged in account currency per 100k units on entry and again on exit. OANDA's standard pricing is spread-only, so set this for commission-based accounts.
- **Swap/financing** (`--swap-long`, `--swap-short`, default 0): annual % of notional for long and short positions; negative means you pay. Charged at each 5pm New York rollover a trade is held through, three days on Wednesday to cover the weekend, and none on Saturday or Sunday. Real rates change with interest-rate policy, so take current rates from your broker.
- **Gaps:** if a candle opens beyond a stop-loss (common after weekends or news), the stop fills at that worse opening price plus slippage. Take-profits that gap still fill at their limit price, so gaps can only hurt results. The results table shows how many stops gapped and what the gaps cost.
- **Not modelled:** financing rates changing over the backtest period. If a single candle touches both the stop-loss and the take-profit, the stop-loss is assumed to fill first.
- **Sizing:** position sizing ignores these costs, as it does in live trading. A stopped-out trade therefore loses slightly more than the 1% risk budget.

With several pairs, all of them trade one account: the Risk Governor's limits (max open trades, daily loss, drawdown, margin, one position per pair) apply across the portfolio, and a second table breaks results down per pair. Pairs may cover different date ranges; each is evaluated only when it has a new completed candle.

The results table lists the commission paid, the slippage cost, the swap earned or paid, and the cost of stops that gapped. The backtester reuses the same `Agent`, strategy and Risk Governor that trade live, running on `SimBroker`.

## Walk-forward optimization

`zia walkforward` checks whether tuning the strategy on past data would have held up on data it hadn't seen:

1. Split the data into rolling windows: train on `--train` candles (default 2000), then test on the next `--test` candles (default 500), then slide forward by `--test` and repeat.
2. On each training window, backtest every combination in the `--grid` and pick the one with the best **return % ÷ max drawdown %** (drawdown floored at 1%). Combinations with fewer than `--min-trades` trades (default 5) are skipped; if none qualify, that fold isn't traded.
3. Trade the following test window with the chosen parameters. Only these out-of-sample test windows are reported.

```bash
zia walkforward --synthetic 3000 --train 1500 --test 500
zia walkforward --pairs EUR_USD,USD_JPY --from 2024-01-01 \
    --grid ema_fast=10,20,30 --grid ema_slow=50,100 --grid sl_atr_mult=1.5,2
```

- **Grid:** any strategy setting (`ema_fast`, `ema_slow`, `rsi_period`, `rsi_long_min`, `sl_atr_mult`, `tp_atr_mult`, …), repeated once per setting. Invalid combinations, such as a fast EMA that isn't faster than the slow one, are skipped. The default grid is `ema_fast=10,20 ema_slow=50,100`.
- **Fixed during the search:** risk limits and the cost options (`--spread-pips`, `--slippage-pips`, `--commission`, `--swap-long/--swap-short`, same defaults as `zia backtest`). There is no LLM review.
- **Warm-up:** each test window's indicators warm up on the candles just before it. Those candles are never traded.
- **Speed:** combinations run in parallel, one process per CPU by default (`--workers`). Each fold runs one backtest per combination, so large grids get slow.
- **Output:** a table per fold (test period, chosen parameters, training score, test return, drawdown, trades) and a summary: compounded out-of-sample return, worst fold drawdown, trades and win rate, and how often each value was chosen. If the chosen parameters change a lot from fold to fold, the edge probably isn't stable.

## Live trading

Live trading is off. All three of these are required to turn it on:
1. `OANDA_ENV=live`
2. `ZIA_LIVE=true`
3. Typing an exact confirmation phrase at the CLI prompt on every `zia run` / `zia close-all`.

Zia refuses to start if only one of the two settings is present, if `ZIA_LIVE=true` is paired with a practice
endpoint, or if the broker's environment differs from the configured one. Nothing the LLM returns can change
the mode.

## Configuration

Everything is configured through environment variables or `.env`; see `.env.example`. Settings are defined and
validated in `zia/config.py`. Strategy parameters use `ZIA_STRATEGY_*` (e.g. `ZIA_STRATEGY_EMA_FAST=20`).
The reviewer model is `ZIA_MODEL` (default `claude-opus-5-5`). The reviewer uses the Anthropic SDK's structured
output with an `effort` setting, so it needs a model that supports both; an unsupported model just makes every
review fail closed.

## Development

```bash
pytest          # all external services are faked; real network access is blocked in tests
ruff check . && ruff format --check .
```
