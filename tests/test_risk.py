from dataclasses import replace
from datetime import timedelta

import pytest

from tests.conftest import APPROVE, NOW, REJECT, make_account, make_price, make_signal
from zia.models import (
    ApprovedOrder,
    Position,
    Review,
    Side,
    TradingEnvironment,
)
from zia.risk import RiskContext, RiskGovernor, RiskLimits

P = TradingEnvironment.PRACTICE


def ctx(**kw) -> RiskContext:
    base = dict(
        now=NOW,
        environment=P,
        broker_environment=P,
        live_confirmed=False,
        account=make_account(),
        price=make_price(),
        open_positions=[],
        day_start_equity=10_000.0,
        peak_equity=10_000.0,
        kill_switch=False,
        mids={},
    )
    base.update(kw)
    return RiskContext(**base)


def failed(decision) -> set[str]:
    return {c.name for c in decision.checks if not c.passed}


gov = RiskGovernor(RiskLimits())


def test_happy_path_buy_builds_protected_order():
    d = gov.evaluate(make_signal(), APPROVE, ctx())
    assert d.approved, d.reason
    o = d.order
    assert isinstance(o, ApprovedOrder)
    price = make_price()
    assert o.entry_price == price.ask
    assert o.stop_loss == pytest.approx(price.ask - 0.0020, abs=1e-5)
    assert o.take_profit == pytest.approx(price.ask + 0.0040, abs=1e-5)
    assert o.stop_loss < o.entry_price < o.take_profit
    assert o.units == pytest.approx(50_000, abs=1)
    assert o.risk_amount <= 100.0 + 1e-6
    assert o.environment is P


def test_happy_path_sell_uses_bid_and_inverted_protection():
    d = gov.evaluate(make_signal(side=Side.SELL), APPROVE, ctx())
    assert d.approved, d.reason
    o = d.order
    assert o.entry_price == make_price().bid
    assert o.take_profit < o.entry_price < o.stop_loss
    assert o.signed_units < 0


def test_usd_jpy_order_sizing_and_rounding():
    sig = make_signal("USD_JPY", ref=150.0, sl_distance=0.30, tp_distance=0.60)
    d = gov.evaluate(sig, APPROVE, ctx(price=make_price("USD_JPY", 150.0)))
    assert d.approved, d.reason
    assert d.order.units == pytest.approx(50_000, abs=50)
    assert round(d.order.stop_loss, 3) == d.order.stop_loss


def test_order_cannot_be_constructed_outside_governor():
    with pytest.raises(PermissionError):
        ApprovedOrder("EUR_USD", Side.BUY, 1000, 1.1, 1.09, 1.12, 10.0, P, "x")


def test_llm_approval_then_governor_rejects_on_limits():
    d = gov.evaluate(make_signal(), APPROVE, ctx(open_positions=[Position("EUR_USD", 1000, 1.1)]))
    assert not d.approved
    assert d.order is None
    assert "one_position_per_pair" in failed(d)


@pytest.mark.parametrize(
    ("review", "name"),
    [
        (None, "llm_review"),
        (REJECT, "llm_review"),
        (Review("approve", 0.3, "meh", source="llm"), "llm_review"),
        (Review("reject", 0.0, "err", source="error"), "llm_review"),
        (Review("approve", 1.0, "off", source="disabled"), "llm_review"),
    ],
)
def test_llm_review_required_when_enabled(review, name):
    d = gov.evaluate(make_signal(), review, ctx())
    assert not d.approved and name in failed(d)


def test_llm_not_required_when_disabled_but_rejection_still_vetoes():
    g = RiskGovernor(RiskLimits(require_llm_approval=False))
    assert g.evaluate(make_signal(), None, ctx()).approved
    assert not g.evaluate(make_signal(), REJECT, ctx()).approved


def test_daily_loss_stop():
    d = gov.evaluate(
        make_signal(), APPROVE, ctx(account=make_account(9_650), day_start_equity=10_000)
    )
    assert "daily_loss" in failed(d)
    d2 = gov.evaluate(
        make_signal(), APPROVE, ctx(account=make_account(9_750), day_start_equity=10_000)
    )
    assert d2.approved


def test_max_drawdown():
    d = gov.evaluate(
        make_signal(),
        APPROVE,
        ctx(account=make_account(8_900), day_start_equity=8_900, peak_equity=10_000),
    )
    assert "max_drawdown" in failed(d)


def test_max_open_trades():
    positions = [Position(i, 1000, 1.0) for i in ("GBP_USD", "USD_JPY", "AUD_USD")]
    d = gov.evaluate(make_signal(), APPROVE, ctx(open_positions=positions))
    assert "max_open_trades" in failed(d)


def test_zero_unit_positions_are_ignored():
    d = gov.evaluate(make_signal(), APPROVE, ctx(open_positions=[Position("EUR_USD", 0, 1.1)]))
    assert d.approved


def test_spread_limit():
    d = gov.evaluate(make_signal(), APPROVE, ctx(price=make_price(spread_pips=3.5)))
    assert "max_spread" in failed(d)
    assert gov.evaluate(make_signal(), APPROVE, ctx(price=make_price(spread_pips=2.9))).approved


def test_spread_limit_in_jpy_pips():
    sig = make_signal("USD_JPY", ref=150.0, sl_distance=0.30, tp_distance=0.60)
    ok = gov.evaluate(sig, APPROVE, ctx(price=make_price("USD_JPY", 150.0, spread_pips=2.0)))
    bad = gov.evaluate(sig, APPROVE, ctx(price=make_price("USD_JPY", 150.0, spread_pips=4.0)))
    assert ok.approved and "max_spread" in failed(bad)


def test_missing_stop_loss_rejected():
    d = gov.evaluate(make_signal(sl_distance=0.0), APPROVE, ctx())
    assert {"stop_loss_present", "protection_geometry"} <= failed(d)
    assert d.order is None


def test_missing_take_profit_rejected():
    d = gov.evaluate(make_signal(tp_distance=0.0), APPROVE, ctx())
    assert "take_profit_present" in failed(d)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.002])
def test_invalid_stop_values_rejected(bad):
    d = gov.evaluate(make_signal(sl_distance=bad), APPROVE, ctx())
    assert not d.approved


def test_stop_too_tight():
    d = gov.evaluate(make_signal(sl_distance=0.0003, tp_distance=0.0010), APPROVE, ctx())
    assert "min_stop_distance" in failed(d)


def test_reward_risk_minimum():
    d = gov.evaluate(make_signal(sl_distance=0.0020, tp_distance=0.0010), APPROVE, ctx())
    assert "reward_risk" in failed(d)


def test_kill_switch():
    d = gov.evaluate(make_signal(), APPROVE, ctx(kill_switch=True))
    assert "kill_switch" in failed(d)


def test_environment_mismatch_rejected():
    d = gov.evaluate(make_signal(), APPROVE, ctx(broker_environment=TradingEnvironment.LIVE))
    assert "environment" in failed(d)


def test_live_requires_confirmation():
    L = TradingEnvironment.LIVE
    d = gov.evaluate(make_signal(), APPROVE, ctx(environment=L, broker_environment=L))
    assert "environment" in failed(d)
    d2 = gov.evaluate(
        make_signal(), APPROVE, ctx(environment=L, broker_environment=L, live_confirmed=True)
    )
    assert d2.approved


def test_untradeable_and_stale_price():
    assert "tradeable" in failed(
        gov.evaluate(make_signal(), APPROVE, ctx(price=make_price(tradeable=False)))
    )
    stale = make_price(time=NOW - timedelta(minutes=10))
    assert "price_fresh" in failed(gov.evaluate(make_signal(), APPROVE, ctx(price=stale)))


def test_insane_price():
    p = replace(make_price(), bid=1.2, ask=1.1)
    assert "price_sane" in failed(gov.evaluate(make_signal(), APPROVE, ctx(price=p)))


def test_instrument_mismatch():
    d = gov.evaluate(make_signal("GBP_USD"), APPROVE, ctx())
    assert "instrument_match" in failed(d)


def test_insufficient_margin():
    d = gov.evaluate(make_signal(), APPROVE, ctx(account=make_account(margin_available=100)))
    assert "margin" in failed(d)


def test_tiny_account_cannot_size():
    d = gov.evaluate(
        make_signal(sl_distance=0.05, tp_distance=0.10),
        APPROVE,
        ctx(account=make_account(nav=1.0), day_start_equity=1.0, peak_equity=1.0),
    )
    assert "position_size" in failed(d)


def test_no_currency_conversion_rejected():
    sig = make_signal("EUR_GBP", ref=0.85)
    d = gov.evaluate(sig, APPROVE, ctx(price=make_price("EUR_GBP", 0.85)))
    assert "currency_conversion" in failed(d)


def test_trading_session():
    g = RiskGovernor(RiskLimits(session_start_hour_utc=7, session_end_hour_utc=10))
    assert "trading_session" in failed(g.evaluate(make_signal(), APPROVE, ctx()))  # 12:00 UTC
    g2 = RiskGovernor(RiskLimits(session_start_hour_utc=22, session_end_hour_utc=13))
    assert g2.evaluate(make_signal(), APPROVE, ctx()).approved  # wraps midnight


def test_size_respects_configured_risk_pct():
    g = RiskGovernor(RiskLimits(risk_per_trade_pct=0.5))
    d = g.evaluate(make_signal(), APPROVE, ctx())
    assert d.order.risk_amount <= 50.0 + 1e-6
    assert d.order.units == pytest.approx(25_000, abs=1)


def test_every_decision_records_all_checks():
    d = gov.evaluate(make_signal(), APPROVE, ctx())
    names = {c.name for c in d.checks}
    for expected in (
        "environment",
        "kill_switch",
        "llm_review",
        "daily_loss",
        "max_drawdown",
        "max_open_trades",
        "one_position_per_pair",
        "max_spread",
        "price_fresh",
        "stop_loss_present",
        "take_profit_present",
        "position_size",
        "margin",
    ):
        assert expected in names
