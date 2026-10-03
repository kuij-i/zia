import json
import logging

import httpx
import pytest

from tests.conftest import APPROVE, NOW, make_account, make_price, make_signal
from zia.broker.base import BrokerError
from zia.broker.oanda import OandaBroker
from zia.models import Side, TradingEnvironment
from zia.risk import RiskContext, RiskGovernor, RiskLimits

P = TradingEnvironment.PRACTICE
TOKEN = "test-oanda-token-abcdef"
ACCT = "101-001-1234567-001"


def broker(handler, env=P):
    return OandaBroker(
        environment=env, api_key=TOKEN, account_id=ACCT, transport=httpx.MockTransport(handler)
    )


def approved_order(side=Side.BUY, env=P):
    ctx = RiskContext(
        now=NOW,
        environment=env,
        broker_environment=env,
        live_confirmed=True,
        account=make_account(),
        price=make_price(),
        open_positions=[],
        day_start_equity=10_000,
        peak_equity=10_000,
        kill_switch=False,
    )
    d = RiskGovernor(RiskLimits()).evaluate(make_signal(side=side), APPROVE, ctx)
    assert d.approved, d.reason
    return d.order


def test_practice_url_and_auth_header():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(
            200,
            json={
                "account": {
                    "id": ACCT,
                    "currency": "USD",
                    "balance": "10000.0",
                    "NAV": "10010.5",
                    "marginAvailable": "9000",
                    "marginRate": "0.0333",
                    "unrealizedPL": "10.5",
                }
            },
        )

    acct = broker(handler).get_account()
    assert seen["url"].startswith("https://api-fxpractice.oanda.com/v3/accounts/")
    assert seen["auth"] == f"Bearer {TOKEN}"
    assert acct.nav == 10010.5 and acct.currency == "USD" and acct.margin_rate == 0.0333


def test_repr_hides_token():
    b = broker(lambda r: httpx.Response(200, json={}))
    assert TOKEN not in repr(b)


def test_requires_credentials():
    with pytest.raises(BrokerError):
        OandaBroker(environment=P, api_key="", account_id=ACCT)
    with pytest.raises(BrokerError):
        OandaBroker(environment=TradingEnvironment.BACKTEST, api_key=TOKEN, account_id=ACCT)


def test_candles_parsing_with_nanoseconds():
    def handler(request):
        assert request.url.params["granularity"] == "H1"
        assert request.url.params["price"] == "M"
        return httpx.Response(
            200,
            json={
                "candles": [
                    {
                        "time": "2025-01-06T10:00:00.000000000Z",
                        "complete": True,
                        "volume": 120,
                        "mid": {"o": "1.10000", "h": "1.10200", "l": "1.09900", "c": "1.10100"},
                    },
                    {
                        "time": "2025-01-06T11:00:00.000000000Z",
                        "complete": False,
                        "volume": 10,
                        "mid": {"o": "1.10100", "h": "1.10150", "l": "1.10050", "c": "1.10120"},
                    },
                ]
            },
        )

    candles = broker(handler).get_candles("EUR_USD", "H1", 2)
    assert len(candles) == 2
    assert candles[0].complete and not candles[1].complete
    assert candles[0].close == 1.101
    assert candles[0].time.isoformat() == "2025-01-06T10:00:00+00:00"


def test_malformed_candles_raise():
    b = broker(lambda r: httpx.Response(200, json={"candles": [{"time": "x"}]}))
    with pytest.raises(BrokerError):
        b.get_candles("EUR_USD", "H1", 1)


def test_http_error_raises():
    b = broker(lambda r: httpx.Response(401, json={"errorMessage": "Insufficient authorization"}))
    with pytest.raises(BrokerError, match="401"):
        b.get_account()


def test_transport_error_raises():
    def handler(request):
        raise httpx.ConnectTimeout("timeout")

    with pytest.raises(BrokerError):
        broker(handler).get_prices(["EUR_USD"])


def test_pricing_and_positions():
    def handler(request):
        if request.url.path.endswith("/pricing"):
            return httpx.Response(
                200,
                json={
                    "prices": [
                        {
                            "instrument": "USD_JPY",
                            "time": "2025-01-06T10:00:01.123456789Z",
                            "tradeable": True,
                            "bids": [{"price": "150.010"}],
                            "asks": [{"price": "150.025"}],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "positions": [
                    {
                        "instrument": "EUR_USD",
                        "unrealizedPL": "-3.2",
                        "long": {"units": "0"},
                        "short": {"units": "-1000", "averagePrice": "1.1"},
                    }
                ]
            },
        )

    b = broker(handler)
    p = b.get_price("USD_JPY")
    assert p.bid == 150.010 and p.ask == 150.025 and p.tradeable
    [pos] = b.get_open_positions()
    assert pos.units == -1000 and pos.average_price == 1.1


def fill_body(units: str):
    return {
        "orderCreateTransaction": {"id": "10"},
        "orderFillTransaction": {
            "id": "11",
            "orderID": "10",
            "price": "1.10005",
            "units": units,
            "tradeOpened": {"tradeID": "12"},
        },
    }


def test_order_payload_and_verified_fill():
    order = approved_order()
    sent = {}

    def handler(request):
        if request.method == "POST":
            sent.update(json.loads(request.content))
            return httpx.Response(201, json=fill_body(str(order.units)))
        assert request.url.path.endswith("/trades/12")
        return httpx.Response(
            200,
            json={
                "trade": {
                    "id": "12",
                    "stopLossOrder": {"price": "1.09805"},
                    "takeProfitOrder": {"price": "1.10405"},
                }
            },
        )

    res = broker(handler).place_order(order)
    o = sent["order"]
    assert o["type"] == "MARKET" and o["timeInForce"] == "FOK"
    assert o["units"] == str(order.units)
    assert o["stopLossOnFill"]["price"] == f"{order.stop_loss:.5f}"
    assert o["takeProfitOnFill"]["price"] == f"{order.take_profit:.5f}"
    assert res.filled and res.trade_id == "12" and res.fill_price == 1.10005
    assert res.stop_loss == 1.09805


def test_sell_order_units_negative():
    order = approved_order(Side.SELL)
    sent = {}

    def handler(request):
        if request.method == "POST":
            sent.update(json.loads(request.content))
            return httpx.Response(201, json=fill_body(str(-order.units)))
        return httpx.Response(
            200,
            json={
                "trade": {
                    "id": "12",
                    "stopLossOrder": {"price": "1"},
                    "takeProfitOrder": {"price": "1"},
                }
            },
        )

    assert broker(handler).place_order(order).filled
    assert sent["order"]["units"] == str(-order.units)


def test_http_201_without_fill_is_not_success():
    order = approved_order()
    body = {
        "orderCreateTransaction": {"id": "10"},
        "orderCancelTransaction": {"reason": "MARKET_HALTED"},
    }
    res = broker(lambda r: httpx.Response(201, json=body)).place_order(order)
    assert res.status == "rejected" and "MARKET_HALTED" in res.error


def test_order_reject_response():
    order = approved_order()
    body = {"orderRejectTransaction": {"rejectReason": "INSUFFICIENT_MARGIN"}}
    res = broker(lambda r: httpx.Response(400, json=body)).place_order(order)
    assert res.status == "rejected" and res.error == "INSUFFICIENT_MARGIN"


def test_partial_fill_flagged_as_error():
    order = approved_order()
    res = broker(lambda r: httpx.Response(201, json=fill_body("1"))).place_order(order)
    assert res.status == "error"


def test_fill_without_protection_is_closed_and_flagged():
    order = approved_order()
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(201, json=fill_body(str(order.units)))
        if request.method == "GET":
            return httpx.Response(200, json={"trade": {"id": "12"}})
        return httpx.Response(200, json={})

    res = broker(handler).place_order(order)
    assert res.status == "error" and "closed" in res.error
    assert ("PUT", f"/v3/accounts/{ACCT}/trades/12/close") in calls


def test_order_for_other_environment_refused():
    order = approved_order(env=TradingEnvironment.LIVE)
    with pytest.raises(BrokerError):
        broker(lambda r: httpx.Response(500)).place_order(order)


def test_token_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    b = broker(lambda r: httpx.Response(500, json={"errorMessage": "x"}))
    with pytest.raises(BrokerError) as ei:
        b.get_account()
    assert TOKEN not in str(ei.value)
    assert TOKEN not in caplog.text


def test_close_position():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "positions": [
                        {
                            "instrument": "EUR_USD",
                            "long": {"units": "1000", "averagePrice": "1.1"},
                            "short": {"units": "0"},
                        }
                    ]
                },
            )
        assert json.loads(request.content) == {"longUnits": "ALL"}
        return httpx.Response(
            200,
            json={
                "longOrderFillTransaction": {"orderID": "20", "price": "1.1010", "units": "-1000"}
            },
        )

    res = broker(handler).close_position("EUR_USD")
    assert res.filled and res.units_filled == -1000


def test_get_trade_closed():
    body = {
        "trade": {
            "id": "12",
            "instrument": "EUR_USD",
            "state": "CLOSED",
            "initialUnits": "1000",
            "price": "1.1",
            "realizedPL": "-12.5",
            "closeTime": "2025-01-06T12:00:00.000000000Z",
        }
    }
    t = broker(lambda r: httpx.Response(200, json=body)).get_trade("12")
    assert t.state == "closed" and t.realized_pl == -12.5


def test_candles_range_pages_and_drops_incomplete():
    from datetime import UTC, datetime

    def handler(request):
        start = request.url.params["from"]
        if start.startswith("2025-01-06T00"):
            return httpx.Response(
                200,
                json={
                    "candles": [
                        {
                            "time": f"2025-01-06T0{h}:00:00.000000000Z",
                            "complete": True,
                            "mid": {"o": "1", "h": "1", "l": "1", "c": "1"},
                        }
                        for h in range(3)
                    ]
                    + [
                        {
                            "time": "2025-01-06T03:00:00.000000000Z",
                            "complete": False,
                            "mid": {"o": "1", "h": "1", "l": "1", "c": "1"},
                        }
                    ]
                },
            )
        raise AssertionError("unexpected second page")

    out = broker(handler).get_candles_range(
        "EUR_USD", "H1", datetime(2025, 1, 6, tzinfo=UTC), datetime(2025, 1, 6, 2, tzinfo=UTC)
    )
    assert [c.time.hour for c in out] == [0, 1]
