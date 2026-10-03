"""Market data helpers: candle lists <-> DataFrames, CSV loading, synthetic data."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from zia.models import Candle


def completed(candles: Sequence[Candle]) -> list[Candle]:
    return [c for c in candles if c.complete]


def to_frame(candles: Sequence[Candle]) -> pd.DataFrame:
    arr = np.array([(c.open, c.high, c.low, c.close, c.volume) for c in candles], dtype=float)
    arr = arr.reshape(-1, 5)
    index = pd.DatetimeIndex([c.time for c in candles], name="time")
    return pd.DataFrame(arr, index=index, columns=["open", "high", "low", "close", "volume"])


def parse_time(value: str) -> datetime:
    """Parse RFC3339 (OANDA uses nanoseconds) or ISO timestamps into aware UTC datetimes."""
    v = value.strip().replace("Z", "+00:00")
    if "." in v:
        head, rest = v.split(".", 1)
        frac = ""
        tz = ""
        for i, ch in enumerate(rest):
            if not ch.isdigit():
                frac, tz = rest[:i], rest[i:]
                break
        else:
            frac = rest
        v = f"{head}.{frac[:6].ljust(6, '0')}{tz}"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def load_csv(path: Path) -> list[Candle]:
    """Load candles from CSV with columns time,open,high,low,close[,volume]."""
    out: list[Candle] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                Candle(
                    time=parse_time(row["time"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(float(row.get("volume") or 0)),
                    complete=True,
                )
            )
    out.sort(key=lambda c: c.time)
    return out


def synthetic_candles(
    n: int,
    start_price: float = 1.1000,
    seed: int = 7,
    start: datetime | None = None,
    step: timedelta = timedelta(hours=1),
    volatility: float = 0.0012,
) -> list[Candle]:
    """Seeded random-walk candles with regime drift. For tests and demos only: not real data."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2025, 1, 6, tzinfo=UTC)
    price = start_price
    candles: list[Candle] = []
    drift = 0.0
    for i in range(n):
        if i % 150 == 0:
            drift = rng.normal(0, volatility * 0.25)
        ret = rng.normal(drift, volatility)
        open_ = price
        close = max(price * (1 + ret), 1e-6)
        wick = abs(rng.normal(0, volatility * 0.5)) * price
        high = max(open_, close) + wick
        low = min(open_, close) - wick
        candles.append(
            Candle(
                time=start + i * step,
                open=open_,
                high=high,
                low=max(low, 1e-6),
                close=close,
                volume=int(rng.integers(100, 1000)),
            )
        )
        price = close
    return candles
