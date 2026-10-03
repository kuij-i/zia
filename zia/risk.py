"""Risk Governor: deterministic position sizing and the final authority before execution.

The governor re-derives everything it needs from the Signal, live broker state and its own
limits. The LLM review is an *input* it may use to reject, never to approve something the
limits forbid, and it cannot influence size, stop or target.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from zia import instruments as inst
from zia.config import Settings
from zia.models import (
    _GOVERNOR_TOKEN,
    Account,
    ApprovedOrder,
    Position,
    Price,
    Review,
    RiskCheck,
    RiskDecision,
    Side,
    Signal,
    TradingEnvironment,
)


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_open_trades: int = 3
    max_spread_pips: float = 3.0
    min_stop_pips: float = 5.0
    min_reward_risk: float = 1.0
    max_units: int = 1_000_000
    max_margin_utilization_pct: float = 50.0
    max_price_age_s: float = 120.0
    session_start_hour_utc: int | None = None
    session_end_hour_utc: int | None = None
    require_llm_approval: bool = True
    min_llm_confidence: float = 0.6

    @classmethod
    def from_settings(cls, s: Settings, *, require_llm_approval: bool | None = None) -> RiskLimits:
        return cls(
            risk_per_trade_pct=s.risk_per_trade_pct,
            max_daily_loss_pct=s.max_daily_loss_pct,
            max_drawdown_pct=s.max_drawdown_pct,
            max_open_trades=s.max_open_trades,
            max_spread_pips=s.max_spread_pips,
            min_stop_pips=s.min_stop_pips,
            min_reward_risk=s.min_reward_risk,
            max_units=s.max_units,
            max_margin_utilization_pct=s.max_margin_utilization_pct,
            max_price_age_s=s.max_price_age_s,
            session_start_hour_utc=s.session_start_hour_utc,
            session_end_hour_utc=s.session_end_hour_utc,
            require_llm_approval=(
                s.llm_enabled if require_llm_approval is None else require_llm_approval
            ),
            min_llm_confidence=s.llm_min_confidence,
        )


@dataclass(frozen=True)
class RiskContext:
    """Everything the governor needs, fetched fresh from the broker before deciding."""

    now: datetime
    environment: TradingEnvironment  # what Zia was configured/confirmed for
    broker_environment: TradingEnvironment  # what the broker object reports
    live_confirmed: bool
    account: Account
    price: Price
    open_positions: Sequence[Position]
    day_start_equity: float
    peak_equity: float
    kill_switch: bool
    mids: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Sizing:
    units: int
    risk_amount: float
    quote_to_account: float


def position_size(
    *,
    instrument: str,
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    account_currency: str,
    mids: Mapping[str, float] | None = None,
    max_units: int = 1_000_000,
) -> Sizing | None:
    """Units such that hitting ``stop`` loses at most ``risk_pct`` of ``equity``.

    Loss per unit (quote currency) is |entry - stop|; it is converted to account currency
    before dividing into the risk budget. Rounds *down* so the budget is never exceeded.
    Returns None if the inputs make sizing impossible.
    """
    if equity <= 0 or risk_pct <= 0 or entry <= 0 or stop <= 0:
        return None
    # Strip float noise (e.g. 1.1 - 1.098 = 0.0020000000000000018) far below a pipette.
    distance = round(abs(entry - stop), 10)
    if distance <= 0:
        return None
    conv = inst.quote_to_account_rate(instrument, entry, account_currency, mids)
    if conv is None or conv <= 0:
        return None
    budget = equity * risk_pct / 100.0
    loss_per_unit = distance * conv
    units = min(math.floor(budget / loss_per_unit), max_units)
    if units < 1:
        return Sizing(0, 0.0, conv)
    return Sizing(units, units * loss_per_unit, conv)


class RiskGovernor:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def evaluate(self, signal: Signal, review: Review | None, ctx: RiskContext) -> RiskDecision:
        L = self.limits
        checks: list[RiskCheck] = []

        def check(name: str, passed: bool, detail: str) -> bool:
            checks.append(RiskCheck(name, bool(passed), detail))
            return bool(passed)

        # --- Environment & system state -------------------------------------------------
        env_ok = ctx.environment == ctx.broker_environment
        if ctx.environment is TradingEnvironment.LIVE:
            env_ok = env_ok and ctx.live_confirmed
        check(
            "environment",
            env_ok,
            f"configured={ctx.environment.value} broker={ctx.broker_environment.value} "
            f"live_confirmed={ctx.live_confirmed}",
        )
        check("kill_switch", not ctx.kill_switch, "active" if ctx.kill_switch else "inactive")
        check(
            "instrument_match",
            signal.instrument == ctx.price.instrument,
            f"signal={signal.instrument} price={ctx.price.instrument}",
        )

        # --- Advisory review (can only veto) --------------------------------------------
        if L.require_llm_approval:
            ok = (
                review is not None
                and review.source == "llm"
                and review.approved
                and review.confidence >= L.min_llm_confidence
            )
            detail = (
                "missing"
                if review is None
                else f"{review.source}:{review.decision} conf={review.confidence:.2f}"
            )
            check("llm_review", ok, detail)
        elif review is not None and not review.approved:
            check("llm_review", False, f"{review.source}:reject")

        # --- Account health --------------------------------------------------------------
        equity = ctx.account.nav
        check("equity_positive", equity > 0, f"nav={equity:.2f}")
        daily_loss_pct = (
            (ctx.day_start_equity - equity) / ctx.day_start_equity * 100
            if ctx.day_start_equity > 0
            else 100.0
        )
        check(
            "daily_loss",
            daily_loss_pct < L.max_daily_loss_pct,
            f"{daily_loss_pct:.2f}% vs max {L.max_daily_loss_pct}%",
        )
        drawdown_pct = (
            (ctx.peak_equity - equity) / ctx.peak_equity * 100 if ctx.peak_equity > 0 else 100.0
        )
        check(
            "max_drawdown",
            drawdown_pct < L.max_drawdown_pct,
            f"{drawdown_pct:.2f}% vs max {L.max_drawdown_pct}%",
        )

        # --- Exposure --------------------------------------------------------------------
        open_positions = [p for p in ctx.open_positions if p.units != 0]
        check(
            "max_open_trades",
            len(open_positions) < L.max_open_trades,
            f"{len(open_positions)} open vs max {L.max_open_trades}",
        )
        check(
            "one_position_per_pair",
            all(p.instrument != signal.instrument for p in open_positions),
            f"existing position in {signal.instrument}"
            if any(p.instrument == signal.instrument for p in open_positions)
            else "none",
        )

        # --- Market conditions -----------------------------------------------------------
        price = ctx.price
        check("tradeable", price.tradeable, f"tradeable={price.tradeable}")
        sane = 0 < price.bid <= price.ask and all(map(math.isfinite, (price.bid, price.ask)))
        check("price_sane", sane, f"bid={price.bid} ask={price.ask}")
        age = (ctx.now - price.time).total_seconds()
        check(
            "price_fresh",
            -5 <= age <= L.max_price_age_s,
            f"age={age:.0f}s vs max {L.max_price_age_s:.0f}s",
        )
        spread_pips = inst.to_pips(signal.instrument, price.ask - price.bid) if sane else math.inf
        check(
            "max_spread",
            spread_pips <= L.max_spread_pips,
            f"{spread_pips:.2f} pips vs max {L.max_spread_pips}",
        )
        if L.session_start_hour_utc is not None and L.session_end_hour_utc is not None:
            h = ctx.now.hour
            s, e = L.session_start_hour_utc, L.session_end_hour_utc
            in_session = s <= h < e if s <= e else (h >= s or h < e)
            check("trading_session", in_session, f"hour={h} session=[{s},{e})")

        # --- Order construction & protection ---------------------------------------------
        entry = price.ask if signal.side is Side.BUY else price.bid
        sl_dist, tp_dist = signal.sl_distance, signal.tp_distance
        stops_ok = math.isfinite(sl_dist) and math.isfinite(tp_dist) and sl_dist > 0 and tp_dist > 0
        check("stop_loss_present", stops_ok and sl_dist > 0, f"sl_distance={sl_dist}")
        check("take_profit_present", stops_ok and tp_dist > 0, f"tp_distance={tp_dist}")

        stop_loss = take_profit = 0.0
        if stops_ok and sane:
            if signal.side is Side.BUY:
                stop_loss, take_profit = entry - sl_dist, entry + tp_dist
            else:
                stop_loss, take_profit = entry + sl_dist, entry - tp_dist
            stop_loss = inst.round_price(signal.instrument, stop_loss)
            take_profit = inst.round_price(signal.instrument, take_profit)
        geometry_ok = (
            stop_loss > 0
            and take_profit > 0
            and (
                (signal.side is Side.BUY and stop_loss < entry < take_profit)
                or (signal.side is Side.SELL and take_profit < entry < stop_loss)
            )
        )
        check(
            "protection_geometry",
            geometry_ok,
            f"side={signal.side.value} entry={entry} sl={stop_loss} tp={take_profit}",
        )
        stop_pips = inst.to_pips(signal.instrument, abs(entry - stop_loss)) if geometry_ok else 0
        check(
            "min_stop_distance",
            stop_pips >= L.min_stop_pips,
            f"{stop_pips:.1f} pips vs min {L.min_stop_pips}",
        )
        rr = abs(take_profit - entry) / abs(entry - stop_loss) if geometry_ok else 0.0
        check("reward_risk", rr >= L.min_reward_risk, f"{rr:.2f} vs min {L.min_reward_risk}")

        # --- Sizing (deterministic, here only) -------------------------------------------
        sizing = None
        if geometry_ok and equity > 0:
            mids = dict(ctx.mids)
            mids.setdefault(signal.instrument, price.mid)
            sizing = position_size(
                instrument=signal.instrument,
                equity=equity,
                risk_pct=L.risk_per_trade_pct,
                entry=entry,
                stop=stop_loss,
                account_currency=ctx.account.currency,
                mids=mids,
                max_units=L.max_units,
            )
        check(
            "currency_conversion",
            sizing is not None,
            "ok" if sizing else f"cannot convert {signal.instrument} to {ctx.account.currency}",
        )
        units = sizing.units if sizing else 0
        budget = equity * L.risk_per_trade_pct / 100 if equity > 0 else 0
        check(
            "position_size",
            sizing is not None
            and 1 <= units <= L.max_units
            and sizing.risk_amount <= budget + 1e-9,
            f"units={units} risk={sizing.risk_amount if sizing else 0:.2f} budget={budget:.2f}",
        )

        # --- Margin ----------------------------------------------------------------------
        if sizing and units > 0:
            base_to_account = entry * sizing.quote_to_account
            required = units * base_to_account * ctx.account.margin_rate
            allowed = ctx.account.margin_available * L.max_margin_utilization_pct / 100
            check(
                "margin",
                ctx.account.margin_rate > 0 and required <= allowed,
                f"required={required:.2f} allowed={allowed:.2f}",
            )
        else:
            check("margin", False, "no valid size")

        failed = [c for c in checks if not c.passed]
        if failed:
            return RiskDecision(
                approved=False,
                checks=checks,
                reason="; ".join(f"{c.name}: {c.detail}" for c in failed),
                units=units or None,
            )

        assert sizing is not None
        order = ApprovedOrder(
            instrument=signal.instrument,
            side=signal.side,
            units=units,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=sizing.risk_amount,
            environment=ctx.environment,
            client_tag=f"zia-{signal.strategy_version}-{signal.candle_time:%Y%m%dT%H%M}",
            _token=_GOVERNOR_TOKEN,
        )
        return RiskDecision(
            approved=True, checks=checks, reason="all checks passed", order=order, units=units
        )
