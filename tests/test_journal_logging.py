import json
import logging
from datetime import UTC, datetime

from tests.conftest import make_signal
from zia.journal import Journal
from zia.logging_setup import JsonFormatter, Redactor, redactor
from zia.models import ExecutionResult, Review

NOW = datetime(2025, 1, 6, 12, tzinfo=UTC)


def test_journal_persists_across_connections(tmp_path):
    path = tmp_path / "j.db"
    j = Journal(path)
    eid = j.record_evaluation(
        now=NOW,
        environment="practice",
        instrument="EUR_USD",
        timeframe="H1",
        candle_time=NOW,
        strategy_version="v1",
        outcome="llm_rejected",
        note="weak",
        indicators={"rsi": 55.0},
        signal=make_signal(),
        review=Review("reject", 0.7, "weak", source="llm", model="m"),
    )
    j.mark_processed("practice", "EUR_USD", "H1", NOW)
    j.record_trade_open(
        env="practice",
        now=NOW,
        side="buy",
        execution=ExecutionResult(
            "filled",
            "EUR_USD",
            trade_id="7",
            fill_price=1.1,
            units_filled=1000,
            stop_loss=1.09,
            take_profit=1.12,
        ),
    )
    j.close()

    j2 = Journal(path)
    row = j2.recent_evaluations(1)[0]
    assert row["id"] == eid and row["llm_rationale"] == "weak"
    assert json.loads(row["indicators_json"]) == {"rsi": 55.0}
    assert j2.last_processed("practice", "EUR_USD", "H1") == NOW
    assert j2.open_trade_ids("practice") == ["7"]
    j2.record_trade_closed("practice", "7", 12.5, NOW)
    assert j2.daily_pnl("practice")[0]["realized_pl"] == 12.5


def test_journal_scrubs_secrets(tmp_path):
    redactor.register("very-secret-token-xyz")
    j = Journal(tmp_path / "j.db")
    j.event(
        "x",
        "failed with very-secret-token-xyz",
        data={"Authorization": "Bearer abc", "nested": {"api_key": "k"}, "ok": 1},
    )
    row = j.conn.execute("SELECT * FROM events").fetchone()
    assert "very-secret-token-xyz" not in row["message"]
    data = json.loads(row["data_json"])
    assert data["Authorization"] == "***" and data["nested"]["api_key"] == "***"
    assert data["ok"] == 1


def test_redactor_patterns():
    r = Redactor()
    r.register("hunter2-secret")
    text = r.scrub("Authorization: Bearer abc.def-123 key=sk-ant-api03-XYZ pw=hunter2-secret")
    assert "abc.def-123" not in text and "sk-ant-api03-XYZ" not in text
    assert "hunter2-secret" not in text


def test_json_formatter_redacts_extras():
    redactor.register("another-secret-value")
    rec = logging.makeLogRecord(
        {
            "msg": "hello another-secret-value",
            "levelname": "INFO",
            "name": "t",
            "token": "abc",
            "instrument": "EUR_USD",
        }
    )
    out = json.loads(JsonFormatter().format(rec))
    assert out["event"] == "hello ***"
    assert out["token"] == "***"
    assert out["instrument"] == "EUR_USD"
