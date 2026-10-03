import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from tests.conftest import find_signal_series, make_signal
from zia.llm_reviewer import AnthropicReviewer, DisabledReviewer, ReviewOutput
from zia.models import Side


class FakeMessages:
    def __init__(self, response=None, exc=None):
        self.response, self.exc, self.calls = response, exc, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs


def reviewer(response=None, exc=None, **kw):
    msgs = FakeMessages(response, exc)
    created = {}

    def factory(**kwargs):
        created["client"] = FakeClient(msgs, **kwargs)
        return created["client"]

    r = AnthropicReviewer(
        model="test-model", api_key="sk-ant-test-key-123456", client_factory=factory, **kw
    )
    return r, msgs, created


def resp(parsed, stop="end_turn"):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop)


CANDLES = find_signal_series(Side.BUY)
SIGNAL = make_signal()


def test_approve():
    r, msgs, created = reviewer(
        resp(ReviewOutput(decision="approve", confidence=0.8, rationale="trend intact"))
    )
    out = r.review(SIGNAL, CANDLES)
    assert out.approved and out.source == "llm" and out.confidence == 0.8
    assert out.model == "test-model"
    assert created["client"].kwargs["timeout"] == 60.0
    assert created["client"].kwargs["max_retries"] == 1


def test_reject():
    r, *_ = reviewer(resp(ReviewOutput(decision="reject", confidence=0.7, rationale="choppy")))
    out = r.review(SIGNAL, CANDLES)
    assert not out.approved and out.source == "llm"


@pytest.mark.parametrize(
    "parsed",
    [
        None,
        {"decision": "approve", "confidence": 1.5, "rationale": "x"},
        {"decision": "buy_more", "confidence": 0.9, "rationale": "x"},
        {"decision": "approve", "confidence": 0.9, "rationale": "x", "units": 1_000_000},
        {"decision": "approve", "confidence": 0.9, "rationale": ""},
        "approve",
    ],
)
def test_malformed_output_rejects(parsed):
    r, *_ = reviewer(resp(parsed))
    out = r.review(SIGNAL, CANDLES)
    assert out.decision == "reject" and out.source == "error"


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", None])
def test_abnormal_stop_reason_rejects(stop):
    r, *_ = reviewer(resp(ReviewOutput(decision="approve", confidence=0.9, rationale="ok"), stop))
    out = r.review(SIGNAL, CANDLES)
    assert out.decision == "reject" and out.source == "error"


@pytest.mark.parametrize(
    "exc",
    [
        anthropic.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")),
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://example.invalid")),
        RuntimeError("boom"),
    ],
)
def test_api_failures_reject(exc):
    r, *_ = reviewer(exc=exc)
    out = r.review(SIGNAL, CANDLES)
    assert out.decision == "reject" and out.source == "error"
    assert out.error


def test_client_construction_failure_rejects():
    def factory(**kwargs):
        raise anthropic.AnthropicError("no credentials")

    r = AnthropicReviewer(model="m", client_factory=factory)
    assert r.review(SIGNAL, CANDLES).decision == "reject"


def test_request_contains_only_market_context():
    r, msgs, _ = reviewer(
        resp(ReviewOutput(decision="reject", confidence=0.5, rationale="x")), context_candles=10
    )
    r.review(SIGNAL, CANDLES)
    call = msgs.calls[0]
    assert call["model"] == "test-model"
    assert call["output_format"] is ReviewOutput
    assert "tools" not in call
    body = call["messages"][0]["content"]
    assert "sk-ant-test-key-123456" not in body
    payload = json.loads(body)
    assert set(payload) == {
        "instrument",
        "timeframe",
        "strategy_version",
        "signal",
        "indicators",
        "recent_completed_candles",
    }
    assert len(payload["recent_completed_candles"]) == 10
    assert payload["signal"]["side"] == "buy"


def test_disabled_reviewer():
    out = DisabledReviewer().review(SIGNAL, CANDLES)
    assert out.approved and out.source == "disabled"
