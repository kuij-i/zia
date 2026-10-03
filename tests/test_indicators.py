import math

import pandas as pd
import pytest

from zia.indicators import atr, ema, rsi, true_range


def test_ema_matches_recursive_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(s, 3)
    alpha = 2 / (3 + 1)
    expected = [1.0]
    for x in s.iloc[1:]:
        expected.append(alpha * x + (1 - alpha) * expected[-1])
    assert out.tolist() == pytest.approx(expected)


def test_ema_constant_series_is_constant():
    assert ema(pd.Series([2.5] * 50), 20).tolist() == pytest.approx([2.5] * 50)


def test_rsi_all_gains_is_100_and_all_losses_is_0():
    up = pd.Series([float(i) for i in range(1, 40)])
    down = pd.Series([float(i) for i in range(40, 1, -1)])
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)


def test_rsi_bounds_and_warmup():
    s = pd.Series([1.0, 1.2, 1.1, 1.3, 1.25, 1.4, 1.35, 1.3, 1.45, 1.5] * 5)
    out = rsi(s, 14)
    assert out.iloc[:13].isna().all()
    valid = out.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_rsi_alternating_is_near_50():
    s = pd.Series([1.0, 1.1] * 100)
    assert rsi(s, 14).iloc[-1] == pytest.approx(50, abs=5)


def test_true_range_uses_previous_close_gaps():
    high = pd.Series([10.0, 12.0])
    low = pd.Series([9.0, 11.5])
    close = pd.Series([9.5, 11.8])
    tr = true_range(high, low, close)
    assert tr.iloc[0] == pytest.approx(1.0)
    assert tr.iloc[1] == pytest.approx(12.0 - 9.5)  # gap up from previous close


def test_atr_constant_range():
    n = 60
    high = pd.Series([1.1010] * n)
    low = pd.Series([1.0990] * n)
    close = pd.Series([1.1000] * n)
    out = atr(high, low, close, 14)
    assert math.isnan(out.iloc[0])
    assert out.iloc[-1] == pytest.approx(0.0020)
