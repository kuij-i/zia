"""SQLite audit journal: every evaluation, review, risk decision, order and outcome."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from zia.logging_setup import redactor
from zia.models import ExecutionResult, Review, RiskDecision, Signal

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    environment TEXT NOT NULL,
    instrument TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    candle_time TEXT,
    strategy_version TEXT,
    outcome TEXT NOT NULL,
    note TEXT,
    signal_json TEXT,
    indicators_json TEXT,
    llm_decision TEXT,
    llm_confidence REAL,
    llm_rationale TEXT,
    llm_source TEXT,
    llm_model TEXT,
    llm_error TEXT,
    risk_approved INTEGER,
    risk_reason TEXT,
    risk_checks_json TEXT,
    units INTEGER,
    stop_loss REAL,
    take_profit REAL,
    order_request_json TEXT,
    execution_status TEXT,
    broker_response_json TEXT,
    fill_price REAL,
    trade_id TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS ix_eval_instrument ON evaluations(instrument, candle_time);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    instrument TEXT NOT NULL,
    side TEXT NOT NULL,
    units REAL NOT NULL,
    entry_price REAL,
    stop_loss REAL,
    take_profit REAL,
    open_time TEXT NOT NULL,
    state TEXT NOT NULL,
    realized_pl REAL,
    close_time TEXT,
    evaluation_id INTEGER,
    PRIMARY KEY (environment, trade_id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    environment TEXT,
    kind TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT,
    data_json TEXT
);

CREATE TABLE IF NOT EXISTS equity (
    ts TEXT NOT NULL,
    environment TEXT NOT NULL,
    balance REAL,
    nav REAL
);

CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _default(o: Any) -> Any:
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o) and not isinstance(o, type):
        return {k: v for k, v in asdict(o).items() if not k.startswith("_")}
    return str(o)


def dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = _default(obj)
    return json.dumps(redactor.scrub_obj(obj), default=_default, sort_keys=True)


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(UTC).isoformat() if dt else None


class Journal:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL" if self.path != ":memory:" else "SELECT 1")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- key/value state ------------------------------------------------------------------
    def get_state(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def last_processed(self, env: str, instrument: str, timeframe: str) -> datetime | None:
        v = self.get_state(f"last_candle:{env}:{instrument}:{timeframe}")
        return datetime.fromisoformat(v) if v else None

    def mark_processed(self, env: str, instrument: str, timeframe: str, t: datetime) -> None:
        self.set_state(f"last_candle:{env}:{instrument}:{timeframe}", _iso(t) or "")

    # -- events ---------------------------------------------------------------------------
    def event(
        self,
        kind: str,
        message: str,
        *,
        level: str = "INFO",
        environment: str | None = None,
        data: Any = None,
        now: datetime | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(ts, environment, kind, level, message, data_json) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (
                _iso(now or datetime.now(UTC)),
                environment,
                kind,
                level,
                redactor.scrub(message),
                dumps(data),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def record_equity(self, env: str, balance: float, nav: float, now: datetime) -> None:
        self.conn.execute(
            "INSERT INTO equity(ts, environment, balance, nav) VALUES(?, ?, ?, ?)",
            (_iso(now), env, balance, nav),
        )
        self.conn.commit()

    # -- evaluations ----------------------------------------------------------------------
    def record_evaluation(
        self,
        *,
        now: datetime,
        environment: str,
        instrument: str,
        timeframe: str,
        candle_time: datetime | None,
        strategy_version: str,
        outcome: str,
        note: str = "",
        indicators: dict | None = None,
        signal: Signal | None = None,
        review: Review | None = None,
        risk: RiskDecision | None = None,
        execution: ExecutionResult | None = None,
        error: str | None = None,
    ) -> int:
        order = risk.order if risk and risk.order else None
        cur = self.conn.execute(
            """INSERT INTO evaluations(
                ts, environment, instrument, timeframe, candle_time, strategy_version, outcome,
                note, signal_json, indicators_json, llm_decision, llm_confidence, llm_rationale,
                llm_source, llm_model, llm_error, risk_approved, risk_reason, risk_checks_json,
                units, stop_loss, take_profit, order_request_json, execution_status,
                broker_response_json, fill_price, trade_id, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _iso(now),
                environment,
                instrument,
                timeframe,
                _iso(candle_time),
                strategy_version,
                outcome,
                note,
                dumps(signal),
                dumps(indicators),
                review.decision if review else None,
                review.confidence if review else None,
                review.rationale if review else None,
                review.source if review else None,
                review.model if review else None,
                review.error if review else None,
                None if risk is None else int(risk.approved),
                risk.reason if risk else None,
                dumps([asdict(c) for c in risk.checks]) if risk else None,
                order.units if order else (risk.units if risk else None),
                order.stop_loss if order else None,
                order.take_profit if order else None,
                dumps(order.to_dict()) if order else None,
                execution.status if execution else None,
                dumps(execution.raw) if execution else None,
                execution.fill_price if execution else None,
                execution.trade_id if execution else None,
                redactor.scrub(error) if error else (execution.error if execution else None),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def recent_evaluations(self, limit: int = 20) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM evaluations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    # -- trades ---------------------------------------------------------------------------
    def record_trade_open(
        self,
        *,
        env: str,
        execution: ExecutionResult,
        side: str,
        now: datetime,
        evaluation_id: int | None = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO trades(trade_id, environment, instrument, side, units,
               entry_price, stop_loss, take_profit, open_time, state, realized_pl, close_time,
               evaluation_id) VALUES (?,?,?,?,?,?,?,?,?, 'open', NULL, NULL, ?)""",
            (
                execution.trade_id,
                env,
                execution.instrument,
                side,
                execution.units_filled,
                execution.fill_price,
                execution.stop_loss,
                execution.take_profit,
                _iso(now),
                evaluation_id,
            ),
        )
        self.conn.commit()

    def open_trade_ids(self, env: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT trade_id FROM trades WHERE environment=? AND state='open'", (env,)
        ).fetchall()
        return [r["trade_id"] for r in rows]

    def record_trade_closed(
        self, env: str, trade_id: str, realized_pl: float, close_time: datetime | None
    ) -> None:
        self.conn.execute(
            "UPDATE trades SET state='closed', realized_pl=?, close_time=? "
            "WHERE environment=? AND trade_id=?",
            (realized_pl, _iso(close_time), env, trade_id),
        )
        self.conn.commit()

    def trades(self, env: str | None = None) -> list[sqlite3.Row]:
        if env:
            return self.conn.execute(
                "SELECT * FROM trades WHERE environment=? ORDER BY open_time", (env,)
            ).fetchall()
        return self.conn.execute("SELECT * FROM trades ORDER BY open_time").fetchall()

    def daily_pnl(self, env: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT substr(close_time, 1, 10) AS day, SUM(realized_pl) AS realized_pl, "
            "COUNT(*) AS trades FROM trades WHERE environment=? AND state='closed' "
            "GROUP BY day ORDER BY day",
            (env,),
        ).fetchall()
