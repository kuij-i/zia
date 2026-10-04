"""Offline backtest: replays historical candles through the *same* Agent, strategy and
Risk Governor, executing on the SimBroker.

Assumptions (explicit):
- Fills at candle close +/- half a fixed spread.
- Fixed slippage in pips against the trader on entries and stop-loss exits; take-profits
  fill at their limit price.
- A stop gapped through at a candle's open fills at that (worse) open.
- Commission per 100k units on each side.
- Swap/financing at annual long/short rates, applied at each 17:00 New York rollover
  (Wednesday x3, none at weekends).
- Stop-loss assumed to fill before take-profit when both are touched within one candle.
- Open trades are marked to the final close (exit costs not yet charged).
- Position sizing ignores these costs, exactly as in live trading.

Results are hypothetical and say nothing about future performance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from zia.agent import Agent
from zia.broker.sim import SimBroker
from zia.config import StrategyParams
from zia.journal import Journal
from zia.llm_reviewer import DisabledReviewer, Reviewer
from zia.models import Candle, TradingEnvironment
from zia.risk import RiskGovernor, RiskLimits


@dataclass(frozen=True)
class PairStats:
    """One instrument's share of a (possibly multi-pair) backtest."""

    instrument: str
    candles: int
    signals: int
    risk_rejections: int
    trades: int
    wins: int
    losses: int
    net_pnl: float  # realized (net of costs) plus unrealized on trades still open
    open_at_end: int

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return self.wins / closed if closed else 0.0


@dataclass(frozen=True)
class BacktestResult:
    instrument: str  # comma-separated when several pairs share the account
    candles: int
    start: datetime | None
    end: datetime | None
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_pnl: float
    return_pct: float
    max_drawdown_pct: float
    starting_balance: float
    ending_equity: float
    spread_pips: float
    slippage_pips: float
    commission_per_100k: float
    total_commission: float
    total_slippage: float
    swap_long_pct: float
    swap_short_pct: float
    total_swap: float
    gapped_stops: int
    total_gap_cost: float
    llm_used: bool
    signals: int
    risk_rejections: int
    open_at_end: int
    per_pair: tuple[PairStats, ...] = ()

    @property
    def instruments(self) -> list[str]:
        return self.instrument.split(",")

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("Mode", "BACKTEST (hypothetical, not practice/live results)"),
            ("Instruments" if len(self.instruments) > 1 else "Instrument", self.instrument),
            (
                "Account",
                "one shared account; risk limits apply across all pairs"
                if len(self.instruments) > 1
                else "single pair",
            ),
            (
                "Period",
                f"{self.start:%Y-%m-%d %H:%M} -> {self.end:%Y-%m-%d %H:%M}"
                if self.start and self.end
                else "n/a",
            ),
            ("Candles", str(self.candles)),
            ("Signals", str(self.signals)),
            ("Risk rejections", str(self.risk_rejections)),
            ("Trades", str(self.trades)),
            ("Wins / losses", f"{self.wins} / {self.losses}"),
            ("Win rate", f"{self.win_rate:.1%}"),
            ("Net P&L", f"{self.net_pnl:,.2f}"),
            ("Return", f"{self.return_pct:.2f}%"),
            ("Max drawdown", f"{self.max_drawdown_pct:.2f}%"),
            (
                "Start balance / end equity",
                f"{self.starting_balance:,.2f} / {self.ending_equity:,.2f}",
            ),
            ("Spread", f"{self.spread_pips:g} pips"),
            ("Slippage", f"{self.slippage_pips:g} pips on entries and stop-loss exits"),
            ("Commission", f"{self.commission_per_100k:,.2f} per 100k units per side"),
            ("Commission paid", f"{self.total_commission:,.2f}"),
            ("Slippage cost", f"{self.total_slippage:,.2f}"),
            (
                "Swap/financing",
                f"long {self.swap_long_pct:+g}% / short {self.swap_short_pct:+g}% p.a., "
                "17:00 New York rollover, Wed x3",
            ),
            ("Swap earned (+) / paid (-)", f"{self.total_swap:+,.2f}"),
            ("Stops gapped through", str(self.gapped_stops)),
            ("Gap cost", f"{self.total_gap_cost:,.2f}"),
            ("Not modelled", "financing-rate changes"),
            ("LLM review", "on" if self.llm_used else "off"),
            ("Open at end (marked to close)", str(self.open_at_end)),
        ]


def run_backtest(
    instrument: str,
    candles: list[Candle],
    *,
    params: StrategyParams,
    limits: RiskLimits,
    balance: float = 10_000.0,
    spread_pips: float = 1.0,
    slippage_pips: float = 0.0,
    commission_per_100k: float = 0.0,
    swap_long_pct: float = 0.0,
    swap_short_pct: float = 0.0,
    step: timedelta = timedelta(hours=1),
    timeframe: str = "H1",
    reviewer: Reviewer | None = None,
    journal: Journal | None = None,
    trade_from: datetime | None = None,
) -> BacktestResult:
    """Backtest a single pair: a portfolio of one (see ``run_portfolio_backtest``)."""
    return run_portfolio_backtest(
        {instrument: candles},
        params=params,
        limits=limits,
        balance=balance,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_100k=commission_per_100k,
        swap_long_pct=swap_long_pct,
        swap_short_pct=swap_short_pct,
        step=step,
        timeframe=timeframe,
        reviewer=reviewer,
        journal=journal,
        trade_from=trade_from,
    )


def run_portfolio_backtest(
    candles_by_pair: dict[str, list[Candle]],
    *,
    params: StrategyParams,
    limits: RiskLimits,
    balance: float = 10_000.0,
    spread_pips: float | dict[str, float] = 1.0,
    slippage_pips: float = 0.0,
    commission_per_100k: float = 0.0,
    swap_long_pct: float = 0.0,
    swap_short_pct: float = 0.0,
    step: timedelta = timedelta(hours=1),
    timeframe: str = "H1",
    reviewer: Reviewer | None = None,
    journal: Journal | None = None,
    trade_from: datetime | None = None,
) -> BacktestResult:
    """Replay one or more pairs through a single shared account, as ``zia run`` trades.

    Time advances over the union of all pairs' candle close times. A pair with no new
    candle at a step is skipped for that step (the agent's duplicate-candle guard).

    Candles opening before ``trade_from`` are indicator history only: the agent does not
    evaluate them, and the reported period, candle counts and stats start at
    ``trade_from``. Walk-forward test windows use this for their warm-up.
    """
    if not candles_by_pair or not any(candles_by_pair.values()):
        raise ValueError("no candles to backtest")
    instruments = list(candles_by_pair)
    reviewer = reviewer or DisabledReviewer()
    llm_used = not isinstance(reviewer, DisabledReviewer)
    if not llm_used:
        limits = replace(limits, require_llm_approval=False)
    journal = journal or Journal(":memory:")
    sim = SimBroker(
        candles_by_pair,
        balance=balance,
        spread_pips=spread_pips,
        step=step,
        slippage_pips=slippage_pips,
        commission_per_100k=commission_per_100k,
        swap_long_pct=swap_long_pct,
        swap_short_pct=swap_short_pct,
    )
    # Sim prices are stamped at the simulated "now", so freshness is exact by construction.
    agent = Agent(
        broker=sim,
        reviewer=reviewer,
        governor=RiskGovernor(limits),
        journal=journal,
        environment=TradingEnvironment.BACKTEST,
        instruments=instruments,
        timeframe=timeframe,
        strategy_params=params,
        candle_count=params.warmup + 50,
        record_equity=False,
    )

    timeline = sorted({c.time + step for series in candles_by_pair.values() for c in series})
    peak = balance
    max_dd = 0.0
    signals = dict.fromkeys(instruments, 0)
    rejections = dict.fromkeys(instruments, 0)
    for now in timeline:
        sim.advance_to(now)
        if trade_from is not None and now - step < trade_from:
            continue  # warm-up history: visible to indicators, never traded
        for outcome in agent.run_cycle(now):
            if outcome.outcome not in ("no_signal", "duplicate", "no_data"):
                signals[outcome.instrument] += 1
            if outcome.outcome == "risk_rejected":
                rejections[outcome.instrument] += 1
        nav = sim.get_account().nav
        peak = max(peak, nav)
        max_dd = max(max_dd, (peak - nav) / peak * 100 if peak > 0 else 0.0)

    per_pair = []
    for instrument in instruments:
        trades = [t for t in sim.trades if t.instrument == instrument]
        closed = [t for t in trades if not t.is_open]
        still_open = [t for t in trades if t.is_open]
        net = sum(t.realized_pl for t in closed) + sum(
            sim.unrealized_pl(t) + t.swap - t.commission for t in still_open
        )
        per_pair.append(
            PairStats(
                instrument=instrument,
                candles=sum(
                    1
                    for c in candles_by_pair[instrument]
                    if trade_from is None or c.time >= trade_from
                ),
                signals=signals[instrument],
                risk_rejections=rejections[instrument],
                trades=len(trades),
                wins=sum(1 for t in closed if t.realized_pl > 0),
                losses=sum(1 for t in closed if t.realized_pl <= 0),
                net_pnl=net,
                open_at_end=len(still_open),
            )
        )

    all_times = [
        c.time
        for series in candles_by_pair.values()
        for c in series
        if trade_from is None or c.time >= trade_from
    ] or [c.time for series in candles_by_pair.values() for c in series]
    ending = sim.get_account().nav
    wins = sum(p.wins for p in per_pair)
    losses = sum(p.losses for p in per_pair)
    return BacktestResult(
        instrument=",".join(instruments),
        candles=sum(p.candles for p in per_pair),
        start=min(all_times),
        end=max(all_times),
        trades=len(sim.trades),
        wins=wins,
        losses=losses,
        win_rate=wins / (wins + losses) if wins + losses else 0.0,
        net_pnl=ending - balance,
        return_pct=(ending - balance) / balance * 100,
        max_drawdown_pct=max_dd,
        starting_balance=balance,
        ending_equity=ending,
        spread_pips=spread_pips
        if isinstance(spread_pips, int | float)
        else max(spread_pips.values()),
        slippage_pips=slippage_pips,
        commission_per_100k=commission_per_100k,
        total_commission=sum(t.commission for t in sim.trades),
        total_slippage=sum(t.slippage_cost for t in sim.trades),
        swap_long_pct=swap_long_pct,
        swap_short_pct=swap_short_pct,
        total_swap=sum(t.swap for t in sim.trades),
        gapped_stops=sum(t.exit_reason == "stop_loss_gap" for t in sim.trades),
        total_gap_cost=sum(t.gap_cost for t in sim.trades),
        llm_used=llm_used,
        signals=sum(signals.values()),
        risk_rejections=sum(rejections.values()),
        open_at_end=sum(p.open_at_end for p in per_pair),
        per_pair=tuple(per_pair),
    )
