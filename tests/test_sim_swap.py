"""Swap/financing in the SimBroker: rollover calendar and exact amounts."""

from datetime import UTC, datetime, timedelta

import pytest

from tests.test_sim_costs import START, H, open_trade
from zia.backtest import run_backtest
from zia.broker.sim import SimBroker, rollovers_between
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.models import Candle, Side
from zia.risk import RiskLimits

# START is Monday 2025-01-06 00:00 UTC (winter: 17:00 New York = 22:00 UTC).
MON_ROLL = datetime(2025, 1, 6, 22, tzinfo=UTC)


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_rollover_calendar_weekdays_and_wednesday_triple():
    rolls = rollovers_between(utc(2025, 1, 6), utc(2025, 1, 13))  # Mon -> next Mon 00:00
    days = [(r.strftime("%a %H:%M"), d) for r, d in rolls]
    assert days == [
        ("Mon 22:00", 1),
        ("Tue 22:00", 1),
        ("Wed 22:00", 3),
        ("Thu 22:00", 1),
        ("Fri 22:00", 1),
    ]


def test_no_rollover_over_weekend():
    assert rollovers_between(utc(2025, 1, 10, 23), utc(2025, 1, 13, 21)) == []


def test_rollover_follows_new_york_dst():
    [(summer, _)] = rollovers_between(utc(2025, 7, 7, 12), utc(2025, 7, 8, 12))
    assert summer == utc(2025, 7, 7, 21)  # 17:00 EDT


def test_interval_is_half_open():
    assert rollovers_between(MON_ROLL, MON_ROLL + H) == []
    assert len(rollovers_between(MON_ROLL - H, MON_ROLL)) == 1


def flat(instrument_mid: float, hours: int = 96) -> list[Candle]:
    m = instrument_mid
    return [Candle(START + i * H, m, m, m, m) for i in range(hours)]


def sim(mid=1.1000, instrument="EUR_USD", **kw) -> SimBroker:
    return SimBroker({instrument: flat(mid)}, spread_pips=1.0, **kw)


SWAP = dict(swap_long_pct=-3.65, swap_short_pct=1.46)


def test_long_pays_one_day_then_wednesday_triple():
    s = sim(**SWAP)
    order, _ = open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    assert order.units == 50_000
    s.advance_to(MON_ROLL + H)
    t = s.trades[0]
    daily = 1.1 * 50_000 * -3.65 / 100 / 365  # -5.50
    assert t.swap == pytest.approx(daily)
    assert s.balance == pytest.approx(10_000 + daily)
    s.advance_to(MON_ROLL + 2 * timedelta(days=1) + H)  # past Tue and Wed rollovers
    assert t.swap == pytest.approx(daily * 5)


def test_short_earns_positive_rate():
    s = sim(**SWAP)
    order, _ = open_trade(s, "EUR_USD", Side.SELL, 0.0020, 0.0040, 1.1)
    s.advance_to(MON_ROLL + H)
    assert s.trades[0].swap == pytest.approx(1.1 * order.units * 1.46 / 100 / 365)
    assert s.trades[0].swap > 0


def test_usd_jpy_notional_in_usd():
    s = sim(150.0, "USD_JPY", **SWAP)
    order, _ = open_trade(s, "USD_JPY", Side.BUY, 0.30, 0.60, 150.0)
    s.advance_to(MON_ROLL + H)
    # Notional of USD_JPY in a USD account is simply the units.
    assert s.trades[0].swap == pytest.approx(order.units * -3.65 / 100 / 365)


def test_swap_is_part_of_realized_pnl():
    s = sim(**SWAP)
    open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(MON_ROLL + H)
    s.close_position("EUR_USD")
    t = s.trades[0]
    gross = (t.close_price - t.open_price) * t.units
    assert t.realized_pl == pytest.approx(gross + t.swap)
    assert s.balance == pytest.approx(10_000 + t.realized_pl)
    s.advance_to(MON_ROLL + timedelta(days=2))
    assert t.swap == pytest.approx(1.1 * 50_000 * -3.65 / 100 / 365)  # no more after close


def test_trade_closed_before_rollover_pays_nothing():
    s = sim(**SWAP)
    open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(MON_ROLL - H)
    s.close_position("EUR_USD")
    s.advance_to(MON_ROLL + H)
    assert s.trades[0].swap == 0


def test_zero_rates_charge_nothing():
    s = sim()
    open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(MON_ROLL + timedelta(days=3))
    assert s.trades[0].swap == 0
    assert s.balance == 10_000


def test_backtest_reports_swap():
    candles = synthetic_candles(1500, seed=7)
    params = StrategyParams()
    base = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits())
    paid = run_backtest(
        "EUR_USD", candles, params=params, limits=RiskLimits(),
        swap_long_pct=-5.0, swap_short_pct=-5.0,
    )  # fmt: skip
    assert base.total_swap == 0
    assert paid.total_swap < 0
    assert paid.net_pnl < base.net_pnl
    assert dict(paid.rows())["Swap/financing"].startswith("long -5% / short -5%")
