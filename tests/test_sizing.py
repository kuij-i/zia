import random

import pytest

from zia import instruments as inst
from zia.risk import position_size


@pytest.mark.parametrize(
    ("instrument", "pip", "precision"),
    [("EUR_USD", 0.0001, 5), ("GBP_USD", 0.0001, 5), ("USD_JPY", 0.01, 3), ("EUR_JPY", 0.01, 3)],
)
def test_pip_size_and_precision(instrument, pip, precision):
    assert inst.pip_size(instrument) == pip
    assert inst.price_precision(instrument) == precision


def test_to_pips_jpy_vs_non_jpy():
    assert inst.to_pips("EUR_USD", 0.0020) == pytest.approx(20)
    assert inst.to_pips("USD_JPY", 0.30) == pytest.approx(30)
    assert inst.to_pips("USD_JPY", 0.0020) == pytest.approx(0.2)


def test_format_price():
    assert inst.format_price("EUR_USD", 1.1) == "1.10000"
    assert inst.format_price("USD_JPY", 150.12345) == "150.123"


@pytest.mark.parametrize("bad", ["EURUSD", "eur_usd", "EUR-USD", "EUR_US", ""])
def test_invalid_instrument(bad):
    with pytest.raises(ValueError):
        inst.split(bad)


def test_quote_conversion_rules():
    assert inst.quote_to_account_rate("EUR_USD", 1.1, "USD") == 1.0
    assert inst.quote_to_account_rate("USD_JPY", 150.0, "USD") == pytest.approx(1 / 150)
    assert inst.quote_to_account_rate("EUR_USD", 1.1, "EUR") == pytest.approx(1 / 1.1)
    assert inst.quote_to_account_rate("GBP_JPY", 190.0, "USD", {"USD_JPY": 150.0}) == (
        pytest.approx(1 / 150)
    )
    assert inst.quote_to_account_rate("EUR_GBP", 0.85, "USD", {"GBP_USD": 1.27}) == 1.27
    assert inst.quote_to_account_rate("EUR_GBP", 0.85, "USD", {}) is None


def size(**kw):
    defaults = dict(equity=10_000.0, risk_pct=1.0, account_currency="USD")
    defaults.update(kw)
    return position_size(**defaults)


def test_eur_usd_sizing():
    s = size(instrument="EUR_USD", entry=1.1000, stop=1.0980)  # 20 pips
    assert s.units == 50_000
    assert s.risk_amount == pytest.approx(100.0)


def test_gbp_usd_sizing():
    s = size(instrument="GBP_USD", entry=1.2700, stop=1.2675)  # 25 pips
    assert s.units == 40_000
    assert s.risk_amount == pytest.approx(100.0)


def test_usd_jpy_sizing_converts_yen_loss_to_usd():
    s = size(instrument="USD_JPY", entry=150.00, stop=149.70)  # 30 pips = 0.30 JPY
    # 0.30 JPY per unit / 150 = 0.002 USD per unit -> 100 / 0.002
    assert s.units == 50_000
    assert s.risk_amount == pytest.approx(100.0)


def test_usd_jpy_short_sizing():
    s = size(instrument="USD_JPY", entry=150.00, stop=150.45)  # 45 pips above
    assert s.units == pytest.approx(33_333, abs=1)
    assert s.risk_amount <= 100.0


def test_usd_jpy_not_sized_like_non_jpy():
    """Treating 0.30 as if it were a 0.0001-pip pair would give a wildly wrong size."""
    s = size(instrument="USD_JPY", entry=150.00, stop=149.70)
    naive = int(100 / 0.30)  # ignoring conversion to USD
    assert s.units != naive
    assert s.units == 50_000


def test_risk_pct_scales_linearly():
    a = size(instrument="EUR_USD", entry=1.1, stop=1.098, risk_pct=0.5)
    b = size(instrument="EUR_USD", entry=1.1, stop=1.098, risk_pct=1.0)
    assert b.units == pytest.approx(2 * a.units, abs=1)


def test_account_in_base_currency():
    s = size(instrument="EUR_USD", entry=1.1, stop=1.098, account_currency="EUR")
    assert s.risk_amount <= 100.0 + 1e-9
    assert s.units == pytest.approx(55_000, abs=1)


def test_max_units_cap():
    s = size(instrument="EUR_USD", entry=1.1, stop=1.0999, max_units=10_000)
    assert s.units == 10_000


def test_too_small_equity_gives_zero_units():
    s = size(instrument="EUR_USD", entry=1.1, stop=1.0, equity=1.0, risk_pct=0.01)
    assert s.units == 0


@pytest.mark.parametrize(
    "kw",
    [
        dict(instrument="EUR_USD", entry=1.1, stop=1.1),
        dict(instrument="EUR_USD", entry=1.1, stop=0),
        dict(instrument="EUR_USD", entry=1.1, stop=1.09, equity=0),
        dict(instrument="EUR_USD", entry=1.1, stop=1.09, risk_pct=0),
        dict(instrument="EUR_GBP", entry=0.85, stop=0.84),  # no conversion available
    ],
)
def test_invalid_inputs_return_none(kw):
    assert size(**kw) is None


def test_never_exceeds_budget_randomized():
    rng = random.Random(1)
    for _ in range(2000):
        instrument = rng.choice(["EUR_USD", "GBP_USD", "USD_JPY"])
        jpy = instrument.endswith("JPY")
        entry = rng.uniform(140, 160) if jpy else rng.uniform(0.9, 1.4)
        pips = rng.uniform(5, 150)
        stop = entry - pips * inst.pip_size(instrument) * rng.choice([1, -1])
        equity = rng.uniform(500, 1_000_000)
        risk = rng.uniform(0.1, 2.0)
        s = size(instrument=instrument, entry=entry, stop=stop, equity=equity, risk_pct=risk)
        budget = equity * risk / 100
        assert s is not None
        assert s.risk_amount <= budget + 1e-6
        # Rounding down loses less than one unit's worth of risk (unless capped).
        if s.units < 1_000_000:
            assert budget - s.risk_amount <= abs(entry - stop) * s.quote_to_account + 1e-6
