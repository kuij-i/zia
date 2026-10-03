"""Structured (JSON-lines) logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/=]+")
_KEYLIKE = re.compile(r"(?i)\b(sk-ant-[A-Za-z0-9\-_]+)")
_SENSITIVE_KEYS = re.compile(r"(?i)(authorization|api[_-]?key|token|secret|password)")


class Redactor:
    def __init__(self) -> None:
        self._secrets: set[str] = set()

    def register(self, *values: str) -> None:
        self._secrets.update(v for v in values if v and len(v) >= 6)

    def scrub(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "***")
        text = _BEARER.sub(r"\1***", text)
        return _KEYLIKE.sub("***", text)

    def scrub_obj(self, obj: object) -> object:
        if isinstance(obj, dict):
            return {
                k: "***" if _SENSITIVE_KEYS.search(str(k)) else self.scrub_obj(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list | tuple):
            return [self.scrub_obj(v) for v in obj]
        if isinstance(obj, str):
            return self.scrub(obj)
        return obj


redactor = Redactor()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _STANDARD_ATTRS and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redactor.scrub(json.dumps(redactor.scrub_obj(payload), default=str))


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS and not k.startswith("_")
        }
        base = f"{record.levelname:<7} {record.name}: {record.getMessage()}"
        if extras:
            base += " " + json.dumps(redactor.scrub_obj(extras), default=str)
        return redactor.scrub(base)


def setup_logging(
    level: str = "INFO", json_lines: bool = True, secrets: list[str] | None = None
) -> None:
    redactor.register(*(secrets or []))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_lines else PlainFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # Third-party HTTP loggers can include request details; keep them quiet.
    for noisy in ("httpx", "httpcore", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
