"""Offline backtest: replays historical candles through the *same* Agent, strategy and
Risk Governor, executing on the SimBroker.

Assumptions (explicit): fills at candle close +/- half a fixed spread; no slippage,
financing/swap or commission; stop-loss assumed to fill before take-profit when both are
touched within one candle; open trades are marked to the final close. Results are
hypothetical and say nothing about future performance.
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
            ("Spread assumption", f"{self.spread_pips} pips, no slippage/swap/commission"),
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
    sim = SimBroker({instrument: candles}, balance=balance, spread_pips=spread_pips, step=step)
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
        llm_used=llm_used,
        signals=signals,
        risk_rejections=rejections,
        open_at_end=len(open_trades),
    )
