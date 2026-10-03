from datetime import timedelta

import pytest

from tests.conftest import REJECT, FakeReviewer
from zia.backtest import run_backtest
from zia.broker.base import BrokerError
from zia.broker.sim import SimBroker
from zia.config import StrategyParams
from zia.data import load_csv, synthetic_candles
from zia.models import Candle, Side
from zia.risk import RiskLimits


def test_backtest_runs_and_reports(params):
    candles = synthetic_candles(2000, seed=7)
    r = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits())
    assert r.candles == 2000
    assert r.trades > 0
    assert r.wins + r.losses + r.open_at_end == r.trades
    assert 0 <= r.win_rate <= 1
    assert r.max_drawdown_pct >= 0
    assert r.ending_equity == pytest.approx(r.starting_balance + r.net_pnl)
    assert not r.llm_used
    assert "BACKTEST" in dict(r.rows())["Mode"]


def test_backtest_usd_jpy(params):
    candles = synthetic_candles(2000, start_price=150.0, seed=3)
    r = run_backtest("USD_JPY", candles, params=params, limits=RiskLimits())
    assert r.trades > 0
    # 1% risk on 10k: no single loss should be wildly larger than the budget (+ spread).
    assert r.max_drawdown_pct < 50


def test_backtest_is_deterministic(params):
    candles = synthetic_candles(1500, seed=11)
    a = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits())
    b = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits())
    assert a == b


def test_backtest_with_rejecting_reviewer_trades_nothing(params):
    candles = synthetic_candles(1500, seed=7)
    r = run_backtest(
        "EUR_USD", candles, params=params, limits=RiskLimits(), reviewer=FakeReviewer(REJECT)
    )
    assert r.llm_used and r.trades == 0 and r.signals > 0


def test_spread_assumption_applied_to_fills():
    start = synthetic_candles(1)[0].time
    series = bars([(1.1, 1.1, 1.1, 1.1)] * 2, start)
    sim = SimBroker(
        {"EUR_USD": series, "USD_JPY": bars([(150.0,) * 4] * 2, start)},
        spread_pips={"EUR_USD": 2.0, "USD_JPY": 1.5},
    )
    sim.advance_to(start + timedelta(hours=1))
    eur, jpy = sim.get_price("EUR_USD"), sim.get_price("USD_JPY")
    assert eur.ask - eur.bid == pytest.approx(0.0002)
    assert jpy.ask - jpy.bid == pytest.approx(0.015)
    assert eur.mid == pytest.approx(1.1)


def test_wider_spread_never_improves_results(params):
    candles = synthetic_candles(1500, seed=7)
    tight = run_backtest("EUR_USD", candles, params=params, limits=RiskLimits(), spread_pips=0.5)
    wide = run_backtest(
        "EUR_USD", candles, params=params, limits=RiskLimits(max_spread_pips=10), spread_pips=8.0
    )
    assert wide.net_pnl <= tight.net_pnl


def test_load_csv(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(
        "time,open,high,low,close,volume\n"
        "2025-01-06T11:00:00Z,1.1,1.2,1.0,1.15,5\n"
        "2025-01-06T10:00:00Z,1.0,1.1,0.9,1.1,3\n"
    )
    candles = load_csv(p)
    assert [c.close for c in candles] == [1.1, 1.15]
    assert candles[0].time.tzinfo is not None


# -- SimBroker mechanics --------------------------------------------------------------------
def bars(prices, start):
    return [
        Candle(start + i * timedelta(hours=1), o, h, lo, c)
        for i, (o, h, lo, c) in enumerate(prices)
    ]


def test_sim_stop_loss_hit_first_when_both_touched():
    from tests.conftest import APPROVE, make_account, make_signal
    from zia.models import TradingEnvironment
    from zia.risk import RiskContext, RiskGovernor

    start = synthetic_candles(1)[0].time
    series = bars([(1.1, 1.1, 1.1, 1.1), (1.1, 1.2, 1.0, 1.1)], start)
    sim = SimBroker({"EUR_USD": series}, spread_pips=1.0)
    sim.advance_to(start + timedelta(hours=1))
    price = sim.get_price("EUR_USD")
    ctx = RiskContext(
        now=sim.now,
        environment=TradingEnvironment.BACKTEST,
        broker_environment=TradingEnvironment.BACKTEST,
        live_confirmed=False,
        account=make_account(),
        price=price,
        open_positions=[],
        day_start_equity=10_000,
        peak_equity=10_000,
        kill_switch=False,
    )
    d = RiskGovernor(RiskLimits()).evaluate(make_signal(side=Side.BUY), APPROVE, ctx)
    assert d.approved, d.reason
    res = sim.place_order(d.order)
    assert res.filled
    sim.advance_to(start + timedelta(hours=2))
    t = sim.trades[0]
    assert t.exit_reason == "stop_loss"
    assert t.realized_pl == pytest.approx(-d.order.risk_amount, rel=1e-6)
    assert sim.get_trade(t.trade_id).state == "closed"


def test_sim_only_shows_completed_candles():
    start = synthetic_candles(1)[0].time
    series = bars([(1.1, 1.1, 1.1, 1.1)] * 3, start)
    sim = SimBroker({"EUR_USD": series})
    sim.advance_to(start + timedelta(minutes=90))
    assert len(sim.get_candles("EUR_USD", "H1", 10)) == 1


def test_sim_unknown_instrument():
    with pytest.raises(BrokerError):
        SimBroker({}).get_candles("EUR_USD", "H1", 5)


def test_strategy_params_warmup():
    assert StrategyParams().warmup == 150
