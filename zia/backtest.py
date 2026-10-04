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
class BacktestResult:
    instrument: str
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

    def rows(self) -> list[tuple[str, str]]:
        return [
            ("Mode", "BACKTEST (hypothetical, not practice/live results)"),
            ("Instrument", self.instrument),
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
) -> BacktestResult:
    reviewer = reviewer or DisabledReviewer()
    llm_used = not isinstance(reviewer, DisabledReviewer)
    if not llm_used:
        limits = replace(limits, require_llm_approval=False)
    journal = journal or Journal(":memory:")
    sim = SimBroker(
        {instrument: candles},
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
        instruments=[instrument],
        timeframe=timeframe,
        strategy_params=params,
        candle_count=params.warmup + 50,
        record_equity=False,
    )

    peak = balance
    max_dd = 0.0
    signals = rejections = 0
    for bar in candles:
        now = bar.time + step
        sim.advance_to(now)
        for outcome in agent.run_cycle(now):
            if outcome.outcome not in ("no_signal", "duplicate", "no_data"):
                signals += 1
            if outcome.outcome == "risk_rejected":
                rejections += 1
        nav = sim.get_account().nav
        peak = max(peak, nav)
        max_dd = max(max_dd, (peak - nav) / peak * 100 if peak > 0 else 0.0)

    closed = [t for t in sim.trades if not t.is_open]
    open_trades = [t for t in sim.trades if t.is_open]
    ending = sim.get_account().nav
    wins = sum(1 for t in closed if t.realized_pl > 0)
    losses = sum(1 for t in closed if t.realized_pl <= 0)
    return BacktestResult(
        instrument=instrument,
        candles=len(candles),
        start=candles[0].time if candles else None,
        end=candles[-1].time if candles else None,
        trades=len(sim.trades),
        wins=wins,
        losses=losses,
        win_rate=wins / len(closed) if closed else 0.0,
        net_pnl=ending - balance,
        return_pct=(ending - balance) / balance * 100,
        max_drawdown_pct=max_dd,
        starting_balance=balance,
        ending_equity=ending,
        spread_pips=spread_pips,
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
        signals=signals,
        risk_rejections=rejections,
        open_at_end=len(open_trades),
    )
