"""Stops that gap: fill at the worse opening price; take-profits keep their limit price."""

import pytest

from tests.test_sim_costs import START, H, open_trade
from zia.backtest import run_backtest
from zia.broker.sim import SimBroker
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.models import Candle, Side
from zia.risk import RiskLimits


def two_bars(mid: float, nxt: tuple[float, float, float, float]) -> list[Candle]:
    """A flat entry candle at ``mid``, then (open, high, low, close)."""
    o, h, lo, c = nxt
    return [Candle(START, mid, mid, mid, mid), Candle(START + H, o, h, lo, c)]


def eur(nxt, **kw) -> SimBroker:
    return SimBroker({"EUR_USD": two_bars(1.1000, nxt)}, spread_pips=1.0, **kw)


def test_long_stop_gapped_fills_at_open():
    s = eur((1.0950, 1.0960, 1.0940, 1.0955))
    order, _ = open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(START + 2 * H)
    t = s.trades[0]
    open_bid = 1.0950 - 0.00005
    assert t.exit_reason == "stop_loss_gap"
    assert t.close_price == pytest.approx(open_bid)
    assert t.gap_cost == pytest.approx((order.stop_loss - open_bid) * order.units)
    # Loss = the governor's risk budget plus the gap.
    assert -t.realized_pl == pytest.approx(order.risk_amount + t.gap_cost)


def test_gapped_stop_also_slips():
    s = eur((1.0950, 1.0960, 1.0940, 1.0955), slippage_pips=0.5)
    open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(START + 2 * H)
    assert s.trades[0].close_price == pytest.approx(1.0950 - 0.00005 - 0.00005)


def test_short_stop_gapped_up_usd_jpy():
    s = SimBroker({"USD_JPY": two_bars(150.0, (151.0, 151.2, 150.9, 151.1))}, spread_pips=1.0)
    order, _ = open_trade(s, "USD_JPY", Side.SELL, 0.30, 0.60, 150.0)
    s.advance_to(START + 2 * H)
    t = s.trades[0]
    open_ask = 151.0 + 0.005
    assert t.exit_reason == "stop_loss_gap"
    assert t.close_price == pytest.approx(open_ask)
    assert t.gap_cost == pytest.approx((open_ask - order.stop_loss) * order.units / open_ask)
    assert t.realized_pl < -order.risk_amount


def test_favourable_gap_through_take_profit_fills_at_limit():
    s = eur((1.1100, 1.1110, 1.1090, 1.1105))
    order, _ = open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(START + 2 * H)
    t = s.trades[0]
    assert t.exit_reason == "take_profit"
    assert t.close_price == pytest.approx(order.take_profit)
    assert t.gap_cost == 0


def test_stop_touched_intrabar_without_gap_fills_at_stop():
    s = eur((1.0995, 1.1000, 1.0970, 1.0990))
    order, _ = open_trade(s, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    s.advance_to(START + 2 * H)
    t = s.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.close_price == pytest.approx(order.stop_loss)
    assert t.gap_cost == 0


def test_backtest_reports_gaps():
    candles = synthetic_candles(1200, seed=7)
    r = run_backtest("EUR_USD", candles, params=StrategyParams(), limits=RiskLimits())
    rows = dict(r.rows())
    # Synthetic candles open at the previous close, so they never gap.
    assert r.gapped_stops == 0 and r.total_gap_cost == 0
    assert rows["Stops gapped through"] == "0"
    assert "gaps" not in rows["Not modelled"]
