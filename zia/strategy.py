"""Deterministic baseline strategy: EMA(fast/slow) crossover filtered by RSI, ATR stops.

Pure function of the supplied candles and parameters. No broker, network, database, LLM
or risk logic lives here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from zia.config import StrategyParams
from zia.data import completed, to_frame
from zia.indicators import atr, ema, rsi
from zia.models import Candle, Side, Signal

STRATEGY_VERSION = "ema_rsi_atr_v1"


@dataclass(frozen=True)
class Evaluation:
    instrument: str
    timeframe: str
    candle_time: datetime | None  # last completed candle, None if there is no data
    signal: Signal | None
    indicators: dict[str, float]
    note: str


def evaluate(
    instrument: str,
    timeframe: str,
    candles: Sequence[Candle],
    params: StrategyParams,
) -> Evaluation:
    bars = completed(candles)
    if not bars:
        return Evaluation(instrument, timeframe, None, None, {}, "no completed candles")
    last_time = bars[-1].time
    if len(bars) < params.warmup:
        return Evaluation(
            instrument,
            timeframe,
            last_time,
            None,
            {},
            f"insufficient history ({len(bars)} < {params.warmup})",
        )

    df = to_frame(bars)
    fast = ema(df["close"], params.ema_fast)
    slow = ema(df["close"], params.ema_slow)
    rsi_s = rsi(df["close"], params.rsi_period)
    atr_s = atr(df["high"], df["low"], df["close"], params.atr_period)

    ind = {
        "close": float(df["close"].iloc[-1]),
        "ema_fast": float(fast.iloc[-1]),
        "ema_slow": float(slow.iloc[-1]),
        "ema_fast_prev": float(fast.iloc[-2]),
        "ema_slow_prev": float(slow.iloc[-2]),
        "rsi": float(rsi_s.iloc[-1]),
        "atr": float(atr_s.iloc[-1]),
    }
    if any(math.isnan(v) or math.isinf(v) for v in ind.values()) or ind["atr"] <= 0:
        return Evaluation(instrument, timeframe, last_time, None, ind, "indicators not ready")

    crossed_up = ind["ema_fast_prev"] <= ind["ema_slow_prev"] and ind["ema_fast"] > ind["ema_slow"]
    crossed_down = (
        ind["ema_fast_prev"] >= ind["ema_slow_prev"] and ind["ema_fast"] < ind["ema_slow"]
    )
    r = ind["rsi"]

    side: Side | None = None
    reason = "no crossover"
    if crossed_up:
        if params.rsi_long_min < r < params.rsi_long_max:
            side = Side.BUY
            reason = (
                f"EMA{params.ema_fast} crossed above EMA{params.ema_slow}; "
                f"RSI {r:.1f} in ({params.rsi_long_min:g}, {params.rsi_long_max:g})"
            )
        else:
            reason = f"bullish crossover filtered: RSI {r:.1f}"
    elif crossed_down:
        if params.rsi_short_min < r < params.rsi_short_max:
            side = Side.SELL
            reason = (
                f"EMA{params.ema_fast} crossed below EMA{params.ema_slow}; "
                f"RSI {r:.1f} in ({params.rsi_short_min:g}, {params.rsi_short_max:g})"
            )
        else:
            reason = f"bearish crossover filtered: RSI {r:.1f}"

    if side is None:
        return Evaluation(instrument, timeframe, last_time, None, ind, reason)

    signal = Signal(
        instrument=instrument,
        timeframe=timeframe,
        side=side,
        candle_time=last_time,
        reference_price=ind["close"],
        sl_distance=ind["atr"] * params.sl_atr_mult,
        tp_distance=ind["atr"] * params.tp_atr_mult,
        reason=reason,
        strategy_version=STRATEGY_VERSION,
        indicators=ind,
    )
    return Evaluation(instrument, timeframe, last_time, signal, ind, reason)
