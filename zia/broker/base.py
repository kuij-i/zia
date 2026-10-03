"""Broker interface. Core trading logic depends only on this and ``zia.models``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from zia.models import (
    Account,
    ApprovedOrder,
    Candle,
    ExecutionResult,
    Position,
    Price,
    TradeInfo,
    TradingEnvironment,
)


class BrokerError(Exception):
    """Any failure talking to the broker (transport, HTTP status, malformed payload)."""


class Broker(ABC):
    environment: TradingEnvironment

    @abstractmethod
    def get_candles(self, instrument: str, granularity: str, count: int) -> list[Candle]:
        """Most recent candles, oldest first. The last one may be incomplete."""

    @abstractmethod
    def get_prices(self, instruments: Sequence[str]) -> dict[str, Price]: ...

    def get_price(self, instrument: str) -> Price:
        prices = self.get_prices([instrument])
        if instrument not in prices:
            raise BrokerError(f"no price returned for {instrument}")
        return prices[instrument]

    @abstractmethod
    def get_account(self) -> Account: ...

    @abstractmethod
    def get_open_positions(self) -> list[Position]: ...

    @abstractmethod
    def place_order(self, order: ApprovedOrder) -> ExecutionResult:
        """Submit a market order with attached stop-loss and take-profit.

        Must verify the broker's response and report the *actual* outcome.
        """

    @abstractmethod
    def close_position(self, instrument: str) -> ExecutionResult: ...

    @abstractmethod
    def get_trade(self, trade_id: str) -> TradeInfo: ...

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


def check_order_protection(order: ApprovedOrder) -> None:
    """Last-line guard every broker runs before sending anything."""
    if not isinstance(order, ApprovedOrder):
        raise BrokerError("only Risk Governor approved orders can be executed")
    if not (order.stop_loss > 0 and order.take_profit > 0):
        raise BrokerError("order is missing stop-loss or take-profit")
    if order.side.value == "buy" and not (order.stop_loss < order.entry_price < order.take_profit):
        raise BrokerError("invalid protection geometry for buy order")
    if order.side.value == "sell" and not (order.take_profit < order.entry_price < order.stop_loss):
        raise BrokerError("invalid protection geometry for sell order")
