"""OANDA v20 REST implementation of the Broker interface.

All OANDA request/response shapes stay in this module. The API token is only ever placed
in the Authorization header of the underlying httpx client and is never logged.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from zia import instruments as inst
from zia.broker.base import Broker, BrokerError, check_order_protection
from zia.data import parse_time
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

log = logging.getLogger(__name__)

BASE_URLS = {
    TradingEnvironment.PRACTICE: "https://api-fxpractice.oanda.com",
    TradingEnvironment.LIVE: "https://api-fxtrade.oanda.com",
}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class OandaBroker(Broker):
    def __init__(
        self,
        *,
        environment: TradingEnvironment,
        api_key: str,
        account_id: str,
        timeout_s: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if environment not in BASE_URLS:
            raise BrokerError(f"OANDA does not support environment {environment!r}")
        if not api_key or not account_id:
            raise BrokerError("OANDA_API_KEY and OANDA_ACCOUNT_ID are required")
        self.environment = environment
        self.base_url = BASE_URLS[environment]
        self.account_id = account_id
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339",
            },
            timeout=timeout_s,
            transport=transport,
        )

    def __repr__(self) -> str:
        return f"OandaBroker(environment={self.environment.value}, base_url={self.base_url})"

    def close(self) -> None:
        self._client.close()

    # -- transport ------------------------------------------------------------------------
    def _request(self, method: str, path: str, **kwargs: Any) -> tuple[int, dict[str, Any]]:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise BrokerError(f"{method} {path} failed: {type(exc).__name__}: {exc}") from exc
        try:
            body = resp.json() if resp.content else {}
        except ValueError as exc:
            raise BrokerError(f"{method} {path}: non-JSON response ({resp.status_code})") from exc
        if not isinstance(body, dict):
            raise BrokerError(f"{method} {path}: unexpected payload type")
        return resp.status_code, body

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        status, body = self._request("GET", path, params=params or None)
        if status != 200:
            raise BrokerError(f"GET {path} -> {status}: {body.get('errorMessage', body)}")
        return body

    @property
    def _acct(self) -> str:
        return f"/v3/accounts/{self.account_id}"

    # -- market data ----------------------------------------------------------------------
    def get_candles(self, instrument: str, granularity: str, count: int) -> list[Candle]:
        body = self._get(
            f"/v3/instruments/{instrument}/candles",
            granularity=granularity,
            count=min(count, 5000),
            price="M",
        )
        out = []
        try:
            for c in body["candles"]:
                mid = c["mid"]
                out.append(
                    Candle(
                        time=parse_time(c["time"]),
                        open=float(mid["o"]),
                        high=float(mid["h"]),
                        low=float(mid["l"]),
                        close=float(mid["c"]),
                        volume=int(c.get("volume", 0)),
                        complete=bool(c.get("complete", False)),
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"malformed candles for {instrument}: {exc}") from exc
        return out

    def get_candles_range(self, instrument: str, granularity: str, start, end=None) -> list[Candle]:
        """Historical candles between two datetimes (paged, completed only)."""
        out: list[Candle] = []
        cursor = start
        while True:
            params: dict[str, Any] = {
                "granularity": granularity,
                "price": "M",
                "from": cursor.isoformat().replace("+00:00", "Z"),
                "count": 5000,
            }
            body = self._get(f"/v3/instruments/{instrument}/candles", **params)
            batch = []
            for c in body.get("candles", []):
                if not c.get("complete"):
                    continue
                t = parse_time(c["time"])
                if end is not None and t >= end:
                    break
                mid = c["mid"]
                batch.append(
                    Candle(
                        t,
                        float(mid["o"]),
                        float(mid["h"]),
                        float(mid["l"]),
                        float(mid["c"]),
                        int(c.get("volume", 0)),
                        True,
                    )
                )
            new = [c for c in batch if not out or c.time > out[-1].time]
            out.extend(new)
            if len(body.get("candles", [])) < 5000 or not new or (end and out[-1].time >= end):
                return out
            cursor = out[-1].time

    def get_prices(self, instruments: Sequence[str]) -> dict[str, Price]:
        body = self._get(f"{self._acct}/pricing", instruments=",".join(instruments))
        out = {}
        try:
            for p in body["prices"]:
                bids, asks = p.get("bids") or [], p.get("asks") or []
                if not bids or not asks:
                    continue
                out[p["instrument"]] = Price(
                    instrument=p["instrument"],
                    bid=float(bids[0]["price"]),
                    ask=float(asks[0]["price"]),
                    time=parse_time(p["time"]),
                    tradeable=bool(p.get("tradeable", False)),
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"malformed pricing: {exc}") from exc
        return out

    # -- account --------------------------------------------------------------------------
    def get_account(self) -> Account:
        body = self._get(f"{self._acct}/summary")
        try:
            a = body["account"]
            return Account(
                account_id=str(a["id"]),
                currency=str(a["currency"]),
                balance=float(a["balance"]),
                nav=float(a["NAV"]),
                margin_available=float(a["marginAvailable"]),
                margin_rate=float(a["marginRate"]),
                unrealized_pl=_f(a.get("unrealizedPL")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"malformed account summary: {exc}") from exc

    def get_open_positions(self) -> list[Position]:
        body = self._get(f"{self._acct}/openPositions")
        out = []
        for p in body.get("positions", []):
            long_u = _f(p.get("long", {}).get("units"))
            short_u = _f(p.get("short", {}).get("units"))
            units = long_u + short_u  # short units are negative
            if units == 0:
                continue
            side = p["long"] if long_u else p["short"]
            out.append(
                Position(
                    instrument=p["instrument"],
                    units=units,
                    average_price=_f(side.get("averagePrice")),
                    unrealized_pl=_f(p.get("unrealizedPL")),
                )
            )
        return out

    def get_trade(self, trade_id: str) -> TradeInfo:
        body = self._get(f"{self._acct}/trades/{trade_id}")
        try:
            t = body["trade"]
            state = "open" if t["state"] == "OPEN" else "closed"
            return TradeInfo(
                trade_id=str(t["id"]),
                instrument=t["instrument"],
                state=state,
                units=_f(t.get("initialUnits")),
                open_price=_f(t.get("price")),
                realized_pl=_f(t.get("realizedPL")),
                close_time=parse_time(t["closeTime"]) if t.get("closeTime") else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BrokerError(f"malformed trade {trade_id}: {exc}") from exc

    # -- orders ---------------------------------------------------------------------------
    def place_order(self, order: ApprovedOrder) -> ExecutionResult:
        check_order_protection(order)
        if order.environment is not self.environment:
            raise BrokerError(
                f"order approved for {order.environment.value} but broker is "
                f"{self.environment.value}"
            )
        request = {
            "order": {
                "type": "MARKET",
                "instrument": order.instrument,
                "units": str(order.signed_units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
                "stopLossOnFill": {
                    "price": inst.format_price(order.instrument, order.stop_loss),
                    "timeInForce": "GTC",
                },
                "takeProfitOnFill": {
                    "price": inst.format_price(order.instrument, order.take_profit),
                    "timeInForce": "GTC",
                },
                "clientExtensions": {"tag": "zia", "comment": order.client_tag[:128]},
            }
        }
        status, body = self._request("POST", f"{self._acct}/orders", json=request)
        return self._verify_fill(order, status, body)

    def _verify_fill(self, order: ApprovedOrder, status: int, body: dict) -> ExecutionResult:
        """Treat the order as filled only if OANDA reports a fill and protection exists."""
        if status not in (200, 201):
            reason = (
                body.get("orderRejectTransaction", {}).get("rejectReason")
                or body.get("errorMessage")
                or f"HTTP {status}"
            )
            return ExecutionResult("rejected", order.instrument, error=str(reason), raw=body)

        fill = body.get("orderFillTransaction")
        if not fill:
            cancel = body.get("orderCancelTransaction", {})
            return ExecutionResult(
                "rejected",
                order.instrument,
                order_id=body.get("orderCreateTransaction", {}).get("id"),
                error=f"not filled: {cancel.get('reason', 'no fill transaction')}",
                raw=body,
            )

        trade_id = (fill.get("tradeOpened") or {}).get("tradeID")
        filled_units = _f(fill.get("units"))
        result_kwargs = dict(
            instrument=order.instrument,
            order_id=fill.get("orderID"),
            trade_id=trade_id,
            fill_price=_f(fill.get("price")) or None,
            units_filled=filled_units,
            raw=body,
        )
        if not trade_id or filled_units != order.signed_units:
            # Partial or netted fill (e.g. reduced an existing position): not what was approved.
            return ExecutionResult(
                "error",
                error=f"unexpected fill: units={filled_units} trade={trade_id}",
                **result_kwargs,
            )

        # Confirm the trade really carries a stop-loss and take-profit.
        try:
            trade = self._get(f"{self._acct}/trades/{trade_id}")["trade"]
        except (BrokerError, KeyError) as exc:
            return ExecutionResult(
                "error", error=f"filled but protection unverified: {exc}", **result_kwargs
            )
        sl = trade.get("stopLossOrder") or {}
        tp = trade.get("takeProfitOrder") or {}
        if not sl or not tp:
            log.error("trade_missing_protection", extra={"trade_id": trade_id})
            with contextlib.suppress(BrokerError):
                self._request("PUT", f"{self._acct}/trades/{trade_id}/close", json={"units": "ALL"})
            return ExecutionResult(
                "error",
                error="filled without stop-loss/take-profit; closed immediately",
                **result_kwargs,
            )
        return ExecutionResult(
            "filled",
            stop_loss=_f(sl.get("price")),
            take_profit=_f(tp.get("price")),
            **result_kwargs,
        )

    def close_position(self, instrument: str) -> ExecutionResult:
        positions = [p for p in self.get_open_positions() if p.instrument == instrument]
        if not positions:
            return ExecutionResult("rejected", instrument, error="no open position")
        payload = {"longUnits": "ALL"} if positions[0].units > 0 else {"shortUnits": "ALL"}
        status, body = self._request(
            "PUT", f"{self._acct}/positions/{instrument}/close", json=payload
        )
        if status != 200:
            return ExecutionResult(
                "rejected", instrument, error=str(body.get("errorMessage", status)), raw=body
            )
        fill = body.get("longOrderFillTransaction") or body.get("shortOrderFillTransaction")
        if not fill:
            return ExecutionResult("error", instrument, error="no close fill reported", raw=body)
        return ExecutionResult(
            "filled",
            instrument,
            order_id=fill.get("orderID"),
            fill_price=_f(fill.get("price")) or None,
            units_filled=_f(fill.get("units")),
            raw=body,
        )
