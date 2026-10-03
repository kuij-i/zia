"""The trading loop.

Per instrument, per *new completed* candle:
    market data -> strategy signal -> LLM review (advisory) -> Risk Governor (final)
    -> approved order -> broker -> verified result -> journal
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from zia.broker.base import Broker, BrokerError
from zia.config import GRANULARITY_SECONDS, StrategyParams
from zia.journal import Journal
from zia.llm_reviewer import Reviewer
from zia.models import ExecutionResult, TradingEnvironment
from zia.risk import RiskContext, RiskGovernor
from zia.strategy import STRATEGY_VERSION, evaluate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleOutcome:
    instrument: str
    outcome: str  # duplicate | no_signal | llm_rejected | risk_rejected | filled | ...
    detail: str = ""
    evaluation_id: int | None = None


class Agent:
    def __init__(
        self,
        *,
        broker: Broker,
        reviewer: Reviewer,
        governor: RiskGovernor,
        journal: Journal,
        environment: TradingEnvironment,
        instruments: list[str],
        timeframe: str,
        strategy_params: StrategyParams,
        candle_count: int = 300,
        live_confirmed: bool = False,
        kill_switch: Callable[[], bool] = lambda: False,
        record_equity: bool = True,
    ) -> None:
        self.broker = broker
        self.reviewer = reviewer
        self.governor = governor
        self.journal = journal
        self.environment = environment
        self.instruments = instruments
        self.timeframe = timeframe
        self.params = strategy_params
        self.candle_count = max(candle_count, strategy_params.warmup + 2)
        self.live_confirmed = live_confirmed
        self.kill_switch = kill_switch
        self.record_equity = record_equity

    @property
    def env(self) -> str:
        return self.environment.value

    # -- equity bookkeeping (for daily-loss and drawdown limits) --------------------------
    def _equity_marks(self, nav: float, now: datetime) -> tuple[float, float]:
        day_key = f"day_start_equity:{self.env}:{now.date().isoformat()}"
        day_start = self.journal.get_state(day_key)
        if day_start is None:
            self.journal.set_state(day_key, repr(nav))
            day_start = repr(nav)
        peak_key = f"peak_equity:{self.env}"
        peak = float(self.journal.get_state(peak_key) or nav)
        if nav > peak:
            peak = nav
        self.journal.set_state(peak_key, repr(peak))
        return float(day_start), peak

    def reconcile_trades(self, now: datetime) -> None:
        for trade_id in self.journal.open_trade_ids(self.env):
            try:
                info = self.broker.get_trade(trade_id)
            except BrokerError as exc:
                self.journal.event(
                    "reconcile_error", str(exc), level="WARNING", environment=self.env, now=now
                )
                continue
            if info.state == "closed":
                self.journal.record_trade_closed(
                    self.env, trade_id, info.realized_pl, info.close_time or now
                )
                log.info(
                    "trade_closed", extra={"trade_id": trade_id, "realized_pl": info.realized_pl}
                )

    # -- main cycle -----------------------------------------------------------------------
    def run_cycle(self, now: datetime | None = None) -> list[CycleOutcome]:
        now = now or datetime.now(UTC)
        if self.broker.environment is not self.environment:
            msg = (
                f"broker environment {self.broker.environment.value} != "
                f"configured {self.env}; halting"
            )
            self.journal.event(
                "environment_mismatch", msg, level="ERROR", environment=self.env, now=now
            )
            raise RuntimeError(msg)

        self.reconcile_trades(now)
        # Establish today's starting equity and the running peak every cycle, so the
        # daily-loss and drawdown limits see losses even on candles without signals.
        account = self.broker.get_account()
        self._equity_marks(account.nav, now)
        if self.record_equity:
            self.journal.record_equity(self.env, account.balance, account.nav, now)
        outcomes = []
        for instrument in self.instruments:
            try:
                outcomes.append(self.process_instrument(instrument, now))
            except Exception as exc:  # one bad instrument must not stop the others
                log.exception("instrument_cycle_failed", extra={"instrument": instrument})
                self.journal.event(
                    "cycle_error",
                    f"{type(exc).__name__}: {exc}",
                    level="ERROR",
                    environment=self.env,
                    data={"instrument": instrument},
                    now=now,
                )
                outcomes.append(CycleOutcome(instrument, "error", str(exc)))
        return outcomes

    def process_instrument(self, instrument: str, now: datetime) -> CycleOutcome:
        candles = self.broker.get_candles(instrument, self.timeframe, self.candle_count)
        done = [c for c in candles if c.complete]
        if not done:
            return CycleOutcome(instrument, "no_data", "no completed candles")
        last_time = done[-1].time
        prev = self.journal.last_processed(self.env, instrument, self.timeframe)
        if prev is not None and last_time <= prev:
            return CycleOutcome(instrument, "duplicate", f"candle {last_time.isoformat()} done")
        # Mark before acting: a crash after this point can skip a candle but never trade it twice.
        self.journal.mark_processed(self.env, instrument, self.timeframe, last_time)

        ev = evaluate(instrument, self.timeframe, done, self.params)
        common = dict(
            now=now,
            environment=self.env,
            instrument=instrument,
            timeframe=self.timeframe,
            candle_time=last_time,
            strategy_version=STRATEGY_VERSION,
            indicators=ev.indicators,
        )
        if ev.signal is None:
            eid = self.journal.record_evaluation(outcome="no_signal", note=ev.note, **common)
            return CycleOutcome(instrument, "no_signal", ev.note, eid)

        signal = ev.signal
        log.info(
            "signal",
            extra={"instrument": instrument, "side": signal.side.value, "reason": signal.reason},
        )

        if self.kill_switch():
            eid = self.journal.record_evaluation(
                outcome="kill_switch",
                note="kill switch active; signal ignored",
                signal=signal,
                **common,
            )
            return CycleOutcome(instrument, "kill_switch", "kill switch active", eid)

        review = self.reviewer.review(signal, done)
        if not review.approved:
            eid = self.journal.record_evaluation(
                outcome="llm_rejected",
                note=review.rationale,
                signal=signal,
                review=review,
                **common,
            )
            return CycleOutcome(instrument, "llm_rejected", review.rationale, eid)

        # Fresh broker state for the governor.
        account = self.broker.get_account()
        price = self.broker.get_price(instrument)
        positions = self.broker.get_open_positions()
        others = [i for i in self.instruments if i != instrument]
        try:
            mids = {k: p.mid for k, p in self.broker.get_prices(others).items()} if others else {}
        except BrokerError:
            mids = {}
        day_start, peak = self._equity_marks(account.nav, now)

        ctx = RiskContext(
            now=now,
            environment=self.environment,
            broker_environment=self.broker.environment,
            live_confirmed=self.live_confirmed,
            account=account,
            price=price,
            open_positions=positions,
            day_start_equity=day_start,
            peak_equity=peak,
            kill_switch=self.kill_switch(),
            mids=mids,
        )
        decision = self.governor.evaluate(signal, review, ctx)
        if not decision.approved or decision.order is None:
            eid = self.journal.record_evaluation(
                outcome="risk_rejected",
                note=decision.reason,
                signal=signal,
                review=review,
                risk=decision,
                **common,
            )
            log.info("risk_rejected", extra={"instrument": instrument, "reason": decision.reason})
            return CycleOutcome(instrument, "risk_rejected", decision.reason, eid)

        try:
            result = self.broker.place_order(decision.order)
        except BrokerError as exc:
            result = ExecutionResult("error", instrument, error=f"broker error: {exc}")

        eid = self.journal.record_evaluation(
            outcome=f"order_{result.status}",
            note=result.error or "",
            signal=signal,
            review=review,
            risk=decision,
            execution=result,
            **common,
        )
        if result.filled and result.trade_id:
            self.journal.record_trade_open(
                env=self.env,
                execution=result,
                side=signal.side.value,
                now=now,
                evaluation_id=eid,
            )
        log.info(
            "order_result",
            extra={
                "instrument": instrument,
                "status": result.status,
                "trade_id": result.trade_id,
                "error": result.error,
            },
        )
        return CycleOutcome(instrument, result.status, result.error or "", eid)

    # -- scheduling -----------------------------------------------------------------------
    def next_wake(self, now: datetime, delay_s: float) -> datetime:
        step = GRANULARITY_SECONDS[self.timeframe]
        epoch = int(now.timestamp())
        boundary = epoch - epoch % step + step
        return datetime.fromtimestamp(boundary, UTC) + timedelta(seconds=delay_s)

    def run_forever(self, delay_s: float = 15.0) -> None:
        self.journal.event("agent_start", "agent loop started", environment=self.env)
        try:
            while True:
                self.run_cycle()
                wake = self.next_wake(datetime.now(UTC), delay_s)
                time.sleep(max(1.0, (wake - datetime.now(UTC)).total_seconds()))
        finally:
            self.journal.event("agent_stop", "agent loop stopped", environment=self.env)
