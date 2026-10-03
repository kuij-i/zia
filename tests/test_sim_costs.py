"""Exact accounting for the SimBroker's slippage and commission model."""

from datetime import timedelta

import pytest

from tests.conftest import APPROVE, make_signal
from zia.backtest import run_backtest
from zia.broker.sim import SimBroker
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.models import Candle, Side, TradingEnvironment
from zia.risk import RiskContext, RiskGovernor, RiskLimits

BT = TradingEnvironment.BACKTEST
START = synthetic_candles(1)[0].time
H = timedelta(hours=1)


def series(mid: float, second: tuple[float, float]) -> list[Candle]:
    """A flat candle at ``mid`` (entry), then one with the given (high, low)."""
    high, low = second
    return [Candle(START, mid, mid, mid, mid), Candle(START + H, mid, high, low, mid)]


def open_trade(sim: SimBroker, instrument: str, side: Side, sl: float, tp: float, ref: float):
    sim.advance_to(START + H)
    ctx = RiskContext(
        now=sim.now, environment=BT, broker_environment=BT, live_confirmed=False,
        account=sim.get_account(), price=sim.get_price(instrument), open_positions=[],
        day_start_equity=10_000, peak_equity=10_000, kill_switch=False,
    )  # fmt: skip
    signal = make_signal(instrument, side, ref=ref, sl_distance=sl, tp_distance=tp)
    decision = RiskGovernor(RiskLimits()).evaluate(signal, APPROVE, ctx)
    assert decision.approved, decision.reason
    result = sim.place_order(decision.order)
    assert result.filled
    return decision.order, result


def eur_sim(second, **costs) -> SimBroker:
    return SimBroker({"EUR_USD": series(1.1000, second)}, spread_pips=1.0, **costs)


COSTS = dict(slippage_pips=0.5, commission_per_100k=5.0)


def test_buy_entry_slips_and_pays_commission():
    sim = eur_sim((1.1000, 1.1000), **COSTS)
    order, res = open_trade(sim, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    assert order.entry_price == pytest.approx(1.10005)  # ask
    assert res.fill_price == pytest.approx(1.10010)  # ask + 0.5 pip
    assert order.units == 50_000
    t = sim.trades[0]
    assert t.commission == pytest.approx(2.5)  # 0.5 lots * 5
    assert t.slippage_cost == pytest.approx(2.5)  # 0.00005 * 50k
    assert sim.balance == pytest.approx(10_000 - 2.5)


def test_buy_take_profit_fills_at_limit_without_slippage():
    sim = eur_sim((1.1045, 1.0995), **COSTS)
    order, _ = open_trade(sim, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    sim.advance_to(START + 2 * H)
    t = sim.trades[0]
    assert t.exit_reason == "take_profit"
    assert t.close_price == pytest.approx(order.take_profit)
    gross = (order.take_profit - 1.10010) * 50_000
    assert t.commission == pytest.approx(5.0)
    assert t.slippage_cost == pytest.approx(2.5)  # entry only
    assert t.realized_pl == pytest.approx(gross - 5.0)
    assert sim.balance == pytest.approx(10_000 + gross - 5.0)


def test_buy_stop_loss_slips_on_exit():
    sim = eur_sim((1.1005, 1.0975), **COSTS)
    order, _ = open_trade(sim, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    sim.advance_to(START + 2 * H)
    t = sim.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.close_price == pytest.approx(order.stop_loss - 0.00005)
    assert t.realized_pl == pytest.approx((t.close_price - 1.10010) * 50_000 - 5.0)
    assert t.slippage_cost == pytest.approx(5.0)  # entry + exit
    # Loss exceeds the governor's 1% budget by exactly the modelled costs.
    assert -t.realized_pl == pytest.approx(order.risk_amount + 5.0 + 5.0)
    assert sim.balance == pytest.approx(10_000 + t.realized_pl)


def test_usd_jpy_sell_stop_loss_costs_in_account_currency():
    sim = SimBroker({"USD_JPY": series(150.0, (150.35, 150.0))}, spread_pips=1.0, **COSTS)
    order, res = open_trade(sim, "USD_JPY", Side.SELL, 0.30, 0.60, 150.0)
    assert res.fill_price == pytest.approx(149.990)  # bid 149.995 - 0.5 pip (0.005)
    sim.advance_to(START + 2 * H)
    t = sim.trades[0]
    assert t.exit_reason == "stop_loss"
    exit_px = order.stop_loss + 0.005  # shorts exit on the ask, slipped upward
    assert t.close_price == pytest.approx(exit_px)
    commission = order.units / 100_000 * 5.0 * 2
    gross_usd = (exit_px - 149.990) * -order.units / exit_px
    assert t.commission == pytest.approx(commission)
    assert t.realized_pl == pytest.approx(gross_usd - commission)
    assert t.slippage_cost == pytest.approx(
        0.005 * order.units / 149.990 + 0.005 * order.units / exit_px
    )


def test_manual_close_slips():
    sim = eur_sim((1.1000, 1.1000), **COSTS)
    open_trade(sim, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    sim.close_position("EUR_USD")
    t = sim.trades[0]
    assert t.exit_reason == "manual"
    assert t.close_price == pytest.approx(1.09995 - 0.00005)  # bid - slip
    assert t.slippage_cost == pytest.approx(5.0)


def test_zero_costs_match_previous_behaviour():
    sim = eur_sim((1.1005, 1.0975))
    order, res = open_trade(sim, "EUR_USD", Side.BUY, 0.0020, 0.0040, 1.1)
    assert res.fill_price == order.entry_price
    sim.advance_to(START + 2 * H)
    t = sim.trades[0]
    assert t.commission == 0 and t.slippage_cost == 0
    assert t.realized_pl == pytest.approx(-order.risk_amount)


@pytest.mark.parametrize("kw", [{"slippage_pips": -0.1}, {"commission_per_100k": -1}])
def test_negative_costs_rejected(kw):
    with pytest.raises(ValueError):
        SimBroker({}, **kw)


def test_backtest_reports_costs_and_they_reduce_pnl():
    candles = synthetic_candles(1500, seed=7)
    params = StrategyParams()
    free = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits())
    costly = run_backtest(
        "EUR_USD", candles, params=params, limits=RiskLimits(),
        slippage_pips=0.5, commission_per_100k=5.0,
    )  # fmt: skip
    assert free.total_commission == 0 and free.total_slippage == 0
    assert costly.total_commission > 0 and costly.total_slippage > 0
    assert costly.net_pnl < free.net_pnl
    rows = dict(costly.rows())
    assert rows["Slippage"].startswith("0.5 pips")
    assert rows["Commission"].startswith("5.00 per 100k")
