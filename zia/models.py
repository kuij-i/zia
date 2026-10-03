"""Broker-independent domain types shared across Zia.

Nothing in here knows about OANDA, HTTP, SQLite or the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal


class TradingEnvironment(StrEnum):
    PRACTICE = "practice"
    LIVE = "live"
    BACKTEST = "backtest"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Candle:
    time: datetime  # candle open time, UTC
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    complete: bool = True


@dataclass(frozen=True)
class Price:
    instrument: str
    bid: float
    ask: float
    time: datetime
    tradeable: bool = True

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class Account:
    account_id: str
    currency: str
    balance: float
    nav: float  # equity, including unrealized P&L
    margin_available: float
    margin_rate: float  # e.g. 0.0333 for 30:1
    unrealized_pl: float = 0.0


@dataclass(frozen=True)
class Position:
    instrument: str
    units: float  # signed: >0 long, <0 short
    average_price: float
    unrealized_pl: float = 0.0


@dataclass(frozen=True)
class Signal:
    """A trade proposal from the deterministic strategy.

    Stop-loss / take-profit are expressed as *distances* in price units so the Risk
    Governor can anchor them to the actual executable price at order time.
    """

    instrument: str
    timeframe: str
    side: Side
    candle_time: datetime
    reference_price: float
    sl_distance: float
    tp_distance: float
    reason: str
    strategy_version: str
    indicators: dict[str, float] = field(default_factory=dict)


ReviewDecision = Literal["approve", "reject"]


@dataclass(frozen=True)
class Review:
    """Advisory verdict on a Signal. It can only approve or reject."""

    decision: ReviewDecision
    confidence: float
    rationale: str
    source: Literal["llm", "disabled", "error"]
    model: str | None = None
    error: str | None = None

    @property
    def approved(self) -> bool:
        return self.decision == "approve"


@dataclass(frozen=True)
class RiskCheck:
    name: str
    passed: bool
    detail: str


_GOVERNOR_TOKEN = object()


@dataclass(frozen=True)
class ApprovedOrder:
    """An order that passed every Risk Governor check.

    Only ``zia.risk.RiskGovernor`` can construct one (it holds the token). Brokers accept
    nothing else.
    """

    instrument: str
    side: Side
    units: int
    entry_price: float  # executable price the order was sized against
    stop_loss: float
    take_profit: float
    risk_amount: float  # account currency at stake if the stop is hit
    environment: TradingEnvironment
    client_tag: str
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _GOVERNOR_TOKEN:
            raise PermissionError("ApprovedOrder can only be created by the Risk Governor")
        if self.units <= 0:
            raise ValueError("ApprovedOrder units must be positive")
        if self.stop_loss <= 0 or self.take_profit <= 0:
            raise ValueError("ApprovedOrder requires stop-loss and take-profit")

    @property
    def signed_units(self) -> int:
        return self.units if self.side is Side.BUY else -self.units

    def to_dict(self) -> dict[str, Any]:
        return {
            "instrument": self.instrument,
            "side": self.side.value,
            "units": self.units,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_amount": self.risk_amount,
            "environment": self.environment.value,
            "client_tag": self.client_tag,
        }


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    checks: list[RiskCheck]
    reason: str
    order: ApprovedOrder | None = None
    units: int | None = None


ExecutionStatus = Literal["filled", "rejected", "error"]


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecutionStatus
    instrument: str
    order_id: str | None = None
    trade_id: str | None = None
    fill_price: float | None = None
    units_filled: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def filled(self) -> bool:
        return self.status == "filled"


@dataclass(frozen=True)
class TradeInfo:
    trade_id: str
    instrument: str
    state: Literal["open", "closed"]
    units: float
    open_price: float
    realized_pl: float = 0.0
    close_time: datetime | None = None
