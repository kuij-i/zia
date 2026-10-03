"""Instrument conventions: pip size, price precision and currency conversion."""

from __future__ import annotations

from collections.abc import Mapping


def split(instrument: str) -> tuple[str, str]:
    parts = instrument.split("_")
    if len(parts) != 2 or not all(len(p) == 3 and p.isalpha() and p.isupper() for p in parts):
        raise ValueError(f"Invalid instrument {instrument!r}, expected e.g. 'EUR_USD'")
    return parts[0], parts[1]


def pip_size(instrument: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    _, quote = split(instrument)
    return 0.01 if quote == "JPY" else 0.0001


def price_precision(instrument: str) -> int:
    """Decimal places OANDA accepts for order prices (fractional pips)."""
    _, quote = split(instrument)
    return 3 if quote == "JPY" else 5


def round_price(instrument: str, price: float) -> float:
    return round(price, price_precision(instrument))


def format_price(instrument: str, price: float) -> str:
    return f"{price:.{price_precision(instrument)}f}"


def to_pips(instrument: str, distance: float) -> float:
    return distance / pip_size(instrument)


def quote_to_account_rate(
    instrument: str,
    price: float,
    account_currency: str,
    mids: Mapping[str, float] | None = None,
) -> float | None:
    """Rate converting one unit of the instrument's quote currency to account currency.

    ``price`` is the instrument's own mid price. ``mids`` maps other instruments to mid
    prices for cross conversion. Returns None when no conversion is possible.
    """
    base, quote = split(instrument)
    if quote == account_currency:
        return 1.0
    if base == account_currency:
        return 1.0 / price if price > 0 else None
    mids = mids or {}
    direct = mids.get(f"{quote}_{account_currency}")
    if direct:
        return direct
    inverse = mids.get(f"{account_currency}_{quote}")
    if inverse:
        return 1.0 / inverse
    return None
