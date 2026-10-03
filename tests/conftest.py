from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import httpx
import pytest

from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.models import Account, Candle, Price, Review, Side, Signal, TradingEnvironment
from zia.strategy import evaluate


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """No real credentials, no stray .env/kill file, and no real network in any test."""
    for key in list(os.environ):
        if key.startswith(("OANDA_", "ZIA_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)

    def _blocked(self, request):
        raise RuntimeError(f"real network access blocked in tests: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)


@pytest.fixture
def params() -> StrategyParams:
    return StrategyParams()


NOW = datetime(2025, 3, 3, 12, 0, tzinfo=UTC)


def find_signal_series(
    side: Side, params: StrategyParams | None = None, seed: int = 7, instrument: str = "EUR_USD"
) -> list[Candle]:
    """Shortest prefix of a seeded series whose last candle produces a signal of ``side``."""
    return list(_find_signal_series(side, params or StrategyParams(), seed, instrument))


@lru_cache(maxsize=32)
def _find_signal_series(
    side: Side, params: StrategyParams, seed: int, instrument: str
) -> tuple[Candle, ...]:
    start_price = 150.0 if instrument.endswith("JPY") else 1.10
    candles = synthetic_candles(3000, start_price=start_price, seed=seed)
    for i in range(params.warmup, len(candles)):
        ev = evaluate(instrument, "H1", candles[: i + 1], params)
        if ev.signal is not None and ev.signal.side is side:
            return tuple(candles[: i + 1])
    raise AssertionError("no signal found in synthetic series")


def make_signal(
    instrument: str = "EUR_USD",
    side: Side = Side.BUY,
    ref: float = 1.1000,
    sl_distance: float = 0.0020,
    tp_distance: float = 0.0040,
    candle_time: datetime = NOW - timedelta(hours=1),
) -> Signal:
    return Signal(
        instrument=instrument,
        timeframe="H1",
        side=side,
        candle_time=candle_time,
        reference_price=ref,
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        reason="test",
        strategy_version="test_v1",
        indicators={"atr": sl_distance / 1.5},
    )


def make_account(
    nav: float = 10_000.0,
    currency: str = "USD",
    margin_rate: float = 0.0333,
    margin_available: float | None = None,
) -> Account:
    return Account(
        "TEST",
        currency,
        nav,
        nav,
        nav if margin_available is None else margin_available,
        margin_rate,
    )


def make_price(
    instrument: str = "EUR_USD",
    mid: float = 1.1000,
    spread_pips: float = 1.0,
    time: datetime = NOW,
    tradeable: bool = True,
) -> Price:
    pip = 0.01 if instrument.endswith("JPY") else 0.0001
    half = spread_pips * pip / 2
    return Price(instrument, mid - half, mid + half, time, tradeable)


APPROVE = Review("approve", 0.9, "looks fine", source="llm", model="test")
REJECT = Review("reject", 0.8, "weak trend", source="llm", model="test")


class FakeReviewer:
    def __init__(self, review: Review = APPROVE) -> None:
        self.review_value = review
        self.calls = 0

    def review(self, signal, candles):
        self.calls += 1
        return self.review_value


@pytest.fixture
def practice_env() -> TradingEnvironment:
    return TradingEnvironment.PRACTICE
