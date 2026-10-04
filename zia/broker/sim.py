"""In-memory simulated broker for tests and offline backtesting.

Fills market orders at the candle close +/- half the configured spread, and closes trades
when a later candle's range touches stop-loss or take-profit. If both are touched in the
same candle the stop-loss is assumed to hit first (conservative).

Costs:
- Spread: buys fill at the ask, sells at the bid; exits trigger on the opposite side.
- Slippage: a fixed number of pips *against* the trader on market fills (entries, manual
  closes) and stop-loss exits. Take-profit exits are limit orders and fill at their price.
- Commission: charged per 100k units on each side (entry and exit), in account currency.
- Swap/financing: annual % of notional, separate for long and short (negative = paid),
  applied at each 17:00 New York rollover a trade is open through. Wednesday's rollover
  counts three days (covering the weekend); there is none on Saturday or Sunday.
- Gaps: if a candle opens beyond a stop-loss (e.g. after a weekend), the stop fills at
  that worse opening price (plus slippage), not at the stop price. Take-profits that gap
  still fill at their limit price, so gaps can only hurt results (conservative).
Changes in financing rates over time are not modelled.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

from zia import instruments as inst
from zia.broker.base import Broker, BrokerError, check_order_protection
from zia.models import (
    Account,
    ApprovedOrder,
    Candle,
    ExecutionResult,
    Position,
    Price,
    Side,
    TradeInfo,
    TradingEnvironment,
)

NEW_YORK = ZoneInfo("America/New_York")
ROLLOVER_TIME = time(17, 0)


def rollovers_between(start: datetime, end: datetime) -> list[tuple[datetime, int]]:
    """FX rollovers in (start, end] as (UTC instant, days financed).

    Rollover is 17:00 New York on weekdays; Wednesday covers three days (the weekend).
    """
    out = []
    day = start.astimezone(NEW_YORK).date()
    last = end.astimezone(NEW_YORK).date()
    while day <= last:
        if day.weekday() < 5:
            instant = datetime.combine(day, ROLLOVER_TIME, NEW_YORK).astimezone(UTC)
            if start < instant <= end:
                out.append((instant, 3 if day.weekday() == 2 else 1))
        day += timedelta(days=1)
    return out


@dataclass
class SimTrade:
    trade_id: str
    instrument: str
    side: Side
    units: int
    open_price: float
    stop_loss: float
    take_profit: float
    open_time: datetime
    close_price: float | None = None
    close_time: datetime | None = None
    realized_pl: float = 0.0  # net of commission and slippage
    exit_reason: str | None = None
    commission: float = 0.0  # entry + exit, account currency
    slippage_cost: float = 0.0  # entry + exit, account currency
    swap: float = 0.0  # financing so far, account currency (positive = earned)
    gap_cost: float = 0.0  # extra loss from a stop filling past its price, account currency

    @property
    def is_open(self) -> bool:
        return self.close_time is None

    @property
    def signed_units(self) -> int:
        return self.units if self.side is Side.BUY else -self.units


class SimBroker(Broker):
    def __init__(
        self,
        candles: dict[str, list[Candle]],
        *,
        balance: float = 10_000.0,
        currency: str = "USD",
        spread_pips: float | dict[str, float] = 1.0,
        margin_rate: float = 0.0333,
        environment: TradingEnvironment = TradingEnvironment.BACKTEST,
        step: timedelta = timedelta(hours=1),
        slippage_pips: float = 0.0,
        commission_per_100k: float = 0.0,
        swap_long_pct: float = 0.0,
        swap_short_pct: float = 0.0,
    ) -> None:
        if slippage_pips < 0 or commission_per_100k < 0:
            raise ValueError("slippage and commission must be non-negative")
        self.environment = environment
        self.step = step  # candle duration: a candle opened at T is complete at T + step
        self._candles = {k: sorted(v, key=lambda c: c.time) for k, v in candles.items()}
        self._cursor: dict[str, int] = {k: -1 for k in self._candles}
        self.balance = balance
        self.currency = currency
        self._spread = spread_pips
        self.margin_rate = margin_rate
        self.slippage_pips = slippage_pips
        self.commission_per_100k = commission_per_100k
        self.swap_long_pct = swap_long_pct
        self.swap_short_pct = swap_short_pct
        self._financed_until: datetime | None = None
        self.trades: list[SimTrade] = []
        self._ids = itertools.count(1)
        self.fail_next_order: str | None = None  # inject a broker failure in tests
        self.now: datetime | None = None
        self.tradeable = True

    # -- time control ---------------------------------------------------------------------
    def advance_to(self, when: datetime) -> None:
        """Make all candles that *closed* by ``when`` visible, settling SL/TP on each."""
        self.now = when
        for instrument, series in self._candles.items():
            idx = self._cursor[instrument]
            while idx + 1 < len(series) and series[idx + 1].time + self.step <= when:
                idx += 1
                self._cursor[instrument] = idx
                self._settle(instrument, series[idx])
        self._apply_financing(when)

    def visible(self, instrument: str) -> list[Candle]:
        return self._candles[instrument][: self._cursor[instrument] + 1]

    def spread_pips(self, instrument: str) -> float:
        if isinstance(self._spread, dict):
            return self._spread.get(instrument, 1.0)
        return self._spread

    def _slip(self, instrument: str) -> float:
        return self.slippage_pips * inst.pip_size(instrument)

    def _commission(self, units: int) -> float:
        return units / 100_000 * self.commission_per_100k

    # -- Broker API -----------------------------------------------------------------------
    def get_candles(self, instrument: str, granularity: str, count: int) -> list[Candle]:
        if instrument not in self._candles:
            raise BrokerError(f"unknown instrument {instrument}")
        return self.visible(instrument)[-count:]

    def get_prices(self, instruments: Sequence[str]) -> dict[str, Price]:
        out = {}
        for instrument in instruments:
            bars = self.visible(instrument) if instrument in self._candles else []
            if not bars:
                continue
            mid = bars[-1].close
            half = self.spread_pips(instrument) * inst.pip_size(instrument) / 2
            out[instrument] = Price(
                instrument, mid - half, mid + half, self.now or bars[-1].time, self.tradeable
            )
        return out

    def _mids(self) -> dict[str, float]:
        return {k: p.mid for k, p in self.get_prices(list(self._candles)).items()}

    def _to_account(self, instrument: str, quote_amount: float, price: float) -> float:
        rate = inst.quote_to_account_rate(instrument, price, self.currency, self._mids())
        if rate is None:
            raise BrokerError(f"cannot convert {instrument} P&L to {self.currency}")
        return quote_amount * rate

    def unrealized_pl(self, t: SimTrade) -> float:
        """Mark-to-market P&L of an open trade at the last visible close, account currency."""
        return self._unrealized(t)

    def _unrealized(self, t: SimTrade) -> float:
        bars = self.visible(t.instrument)
        if not bars:
            return 0.0
        mark = bars[-1].close
        return self._to_account(t.instrument, (mark - t.open_price) * t.signed_units, mark)

    def get_account(self) -> Account:
        upl = sum(self._unrealized(t) for t in self.trades if t.is_open)
        used = 0.0
        for t in self.trades:
            if t.is_open:
                bars = self.visible(t.instrument)
                px = bars[-1].close if bars else t.open_price
                used += self._to_account(t.instrument, px * t.units, px) * self.margin_rate
        nav = self.balance + upl
        return Account("SIM", self.currency, self.balance, nav, nav - used, self.margin_rate, upl)

    def get_open_positions(self) -> list[Position]:
        out: dict[str, Position] = {}
        for t in self.trades:
            if t.is_open:
                prev = out.get(t.instrument)
                units = (prev.units if prev else 0) + t.signed_units
                out[t.instrument] = Position(t.instrument, units, t.open_price, self._unrealized(t))
        return [p for p in out.values() if p.units != 0]

    def place_order(self, order: ApprovedOrder) -> ExecutionResult:
        check_order_protection(order)
        if self.fail_next_order:
            msg, self.fail_next_order = self.fail_next_order, None
            raise BrokerError(msg)
        price = self.get_price(order.instrument)
        slip = self._slip(order.instrument)
        fill = price.ask + slip if order.side is Side.BUY else price.bid - slip
        # Re-check protection against the actual fill.
        if order.side is Side.BUY and not order.stop_loss < fill < order.take_profit:
            return ExecutionResult(
                "rejected", order.instrument, error="stop/target invalid at fill"
            )
        if order.side is Side.SELL and not order.take_profit < fill < order.stop_loss:
            return ExecutionResult(
                "rejected", order.instrument, error="stop/target invalid at fill"
            )
        trade = SimTrade(
            trade_id=str(next(self._ids)),
            instrument=order.instrument,
            side=order.side,
            units=order.units,
            open_price=fill,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            open_time=self.now or price.time,
        )
        trade.commission = self._commission(order.units)
        trade.slippage_cost = self._to_account(order.instrument, slip * order.units, fill)
        self.balance -= trade.commission
        self.trades.append(trade)
        return ExecutionResult(
            "filled",
            order.instrument,
            order_id=trade.trade_id,
            trade_id=trade.trade_id,
            fill_price=fill,
            units_filled=float(order.signed_units),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            raw={"sim": True},
        )

    def close_position(self, instrument: str) -> ExecutionResult:
        open_trades = [t for t in self.trades if t.is_open and t.instrument == instrument]
        if not open_trades:
            return ExecutionResult("rejected", instrument, error="no open position")
        price = self.get_price(instrument)
        for t in open_trades:
            px = price.bid if t.side is Side.BUY else price.ask
            self._close(t, px, self.now or price.time, "manual", slipped=True)
        return ExecutionResult(
            "filled", instrument, units_filled=0.0, raw={"closed": len(open_trades)}
        )

    def get_trade(self, trade_id: str) -> TradeInfo:
        for t in self.trades:
            if t.trade_id == trade_id:
                return TradeInfo(
                    trade_id=t.trade_id,
                    instrument=t.instrument,
                    state="open" if t.is_open else "closed",
                    units=float(t.signed_units),
                    open_price=t.open_price,
                    realized_pl=t.realized_pl,
                    close_time=t.close_time,
                )
        raise BrokerError(f"unknown trade {trade_id}")

    # -- internals ------------------------------------------------------------------------
    def _settle(self, instrument: str, bar: Candle) -> None:
        half = self.spread_pips(instrument) * inst.pip_size(instrument) / 2
        for t in self.trades:
            if not t.is_open or t.instrument != instrument or bar.time < t.open_time:
                continue
            if t.side is Side.BUY:
                # Long exits on the bid.
                open_bid = bar.open - half
                if open_bid <= t.stop_loss:  # gapped through the stop
                    self._gap_close(t, open_bid, bar.time)
                elif bar.low - half <= t.stop_loss:
                    self._close(t, t.stop_loss, bar.time, "stop_loss", slipped=True)
                elif bar.high - half >= t.take_profit:
                    self._close(t, t.take_profit, bar.time, "take_profit")
            else:
                # Short exits on the ask.
                open_ask = bar.open + half
                if open_ask >= t.stop_loss:  # gapped through the stop
                    self._gap_close(t, open_ask, bar.time)
                elif bar.high + half >= t.stop_loss:
                    self._close(t, t.stop_loss, bar.time, "stop_loss", slipped=True)
                elif bar.low + half <= t.take_profit:
                    self._close(t, t.take_profit, bar.time, "take_profit")

    def _gap_close(self, t: SimTrade, open_price: float, when: datetime) -> None:
        """Stop-loss triggered by a gap: fill at the (worse) opening price."""
        t.gap_cost = self._to_account(
            t.instrument, abs(open_price - t.stop_loss) * t.units, open_price
        )
        self._close(t, open_price, when, "stop_loss_gap", slipped=True)

    def _close(
        self, t: SimTrade, price: float, when: datetime, reason: str, *, slipped: bool = False
    ) -> None:
        """Close at ``price``; market-style exits (stops, manual) also suffer slippage."""
        if slipped:
            slip = self._slip(t.instrument)
            price = price - slip if t.side is Side.BUY else price + slip
            t.slippage_cost += self._to_account(t.instrument, slip * t.units, price)
        gross = self._to_account(t.instrument, (price - t.open_price) * t.signed_units, price)
        exit_commission = self._commission(t.units)
        t.commission += exit_commission
        # Entry commission was already taken from the balance when the trade opened.
        self.balance += gross - exit_commission
        t.close_price, t.close_time, t.exit_reason = price, when, reason
        t.realized_pl = gross - t.commission + t.swap

    def _mark_at(self, instrument: str, instant: datetime, fallback: float) -> float:
        """Close of the last candle completed by ``instant``."""
        series = self._candles[instrument]
        times = [c.time + self.step for c in series]
        i = bisect.bisect_right(times, instant) - 1
        return series[i].close if i >= 0 else fallback

    def _apply_financing(self, when: datetime) -> None:
        if self._financed_until is None:
            self._financed_until = when
            return
        if not (self.swap_long_pct or self.swap_short_pct):
            self._financed_until = when
            return
        for instant, days in rollovers_between(self._financed_until, when):
            for t in self.trades:
                held = t.open_time < instant and (t.close_time is None or t.close_time > instant)
                if not held:
                    continue
                pct = self.swap_long_pct if t.side is Side.BUY else self.swap_short_pct
                px = self._mark_at(t.instrument, instant, t.open_price)
                notional = self._to_account(t.instrument, px * t.units, px)
                amount = notional * pct / 100 / 365 * days
                t.swap += amount
                self.balance += amount
                if not t.is_open:  # closed later in this same advance: book into its P&L
                    t.realized_pl += amount
        self._financed_until = when
