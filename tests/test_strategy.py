from dataclasses import replace

import pytest

from tests.conftest import find_signal_series
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.models import Side
from zia.strategy import STRATEGY_VERSION, evaluate


def test_insufficient_history_gives_no_signal(params):
    candles = synthetic_candles(params.warmup - 1)
    ev = evaluate("EUR_USD", "H1", candles, params)
    assert ev.signal is None
    assert "insufficient" in ev.note


def test_no_candles(params):
    ev = evaluate("EUR_USD", "H1", [], params)
    assert ev.signal is None and ev.candle_time is None


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_crossover_produces_signal_with_atr_stops(side, params):
    candles = find_signal_series(side, params)
    ev = evaluate("EUR_USD", "H1", candles, params)
    sig = ev.signal
    assert sig is not None and sig.side is side
    assert sig.strategy_version == STRATEGY_VERSION
    assert sig.candle_time == candles[-1].time
    assert sig.reference_price == candles[-1].close
    assert sig.sl_distance == pytest.approx(ev.indicators["atr"] * params.sl_atr_mult)
    assert sig.tp_distance == pytest.approx(ev.indicators["atr"] * params.tp_atr_mult)
    if side is Side.BUY:
        assert ev.indicators["ema_fast"] > ev.indicators["ema_slow"]
        assert ev.indicators["ema_fast_prev"] <= ev.indicators["ema_slow_prev"]
        assert params.rsi_long_min < ev.indicators["rsi"] < params.rsi_long_max
    else:
        assert ev.indicators["ema_fast"] < ev.indicators["ema_slow"]
        assert params.rsi_short_min < ev.indicators["rsi"] < params.rsi_short_max


def test_rsi_filter_blocks_signal(params):
    candles = find_signal_series(Side.BUY, params)
    strict = StrategyParams(rsi_long_min=99.0, rsi_long_max=100.0)
    ev = evaluate("EUR_USD", "H1", candles, strict)
    assert ev.signal is None
    assert "filtered" in ev.note


def test_no_signal_one_candle_later(params):
    """A crossover is an event on one candle; the next candle should not repeat it."""
    candles = find_signal_series(Side.BUY, params)
    full = synthetic_candles(3000, seed=7)
    nxt = full[: len(candles) + 1]
    ev = evaluate("EUR_USD", "H1", nxt, params)
    assert ev.signal is None or ev.signal.candle_time != candles[-1].time


def test_incomplete_candle_is_ignored(params):
    candles = find_signal_series(Side.BUY, params)
    forming = replace(candles[-1], time=candles[-1].time.replace(year=2030), complete=False)
    ev = evaluate("EUR_USD", "H1", [*candles, forming], params)
    assert ev.candle_time == candles[-1].time
    assert ev.signal is not None


def test_strategy_is_deterministic(params):
    candles = find_signal_series(Side.SELL, params)
    a = evaluate("EUR_USD", "H1", candles, params)
    b = evaluate("EUR_USD", "H1", list(candles), params)
    assert a == b


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        StrategyParams(ema_fast=50, ema_slow=20)
