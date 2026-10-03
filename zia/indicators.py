"""Technical indicators on pandas Series (pure functions, no I/O)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. 100 when there were no losses in the window."""
    delta = close.diff().to_numpy()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=close.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=close.index)
    gain.iloc[0] = loss.iloc[0] = np.nan
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + avg_gain / avg_loss)
    out = np.where(avg_loss == 0, 100.0, out)
    out = np.where(np.isnan(avg_gain), np.nan, out)
    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h, lo, c = high.to_numpy(), low.to_numpy(), close.to_numpy()
    prev = np.concatenate(([np.nan], c[:-1]))
    tr = np.fmax(h - lo, np.fmax(np.abs(h - prev), np.abs(lo - prev)))
    return pd.Series(tr, index=high.index)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
