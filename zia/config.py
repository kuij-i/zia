"""Centralized, typed configuration.

Environment variables (or a local ``.env``) are the only source. ``OANDA_*`` and
``ANTHROPIC_API_KEY`` keep their conventional names; everything else is ``ZIA_*``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from zia import instruments as inst
from zia.models import TradingEnvironment

GRANULARITY_SECONDS: dict[str, int] = {
    "M1": 60,
    "M5": 300,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H4": 14400,
    "D": 86400,
}


class ConfigError(Exception):
    pass


class LiveTradingNotPermitted(ConfigError):
    pass


class StrategyParams(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZIA_STRATEGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    ema_fast: int = Field(20, ge=2)
    ema_slow: int = Field(50, ge=3)
    rsi_period: int = Field(14, ge=2)
    atr_period: int = Field(14, ge=2)
    rsi_long_min: float = 50.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 50.0
    sl_atr_mult: float = Field(1.5, gt=0)
    tp_atr_mult: float = Field(3.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> StrategyParams:
        if self.ema_fast >= self.ema_slow:
            raise ValueError("ema_fast must be smaller than ema_slow")
        return self

    @property
    def warmup(self) -> int:
        """Completed candles needed before a signal can be trusted."""
        return max(self.ema_slow, self.rsi_period, self.atr_period) * 3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ZIA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Broker ---
    oanda_env: Literal["practice", "live"] = Field(
        "practice", validation_alias=AliasChoices("OANDA_ENV")
    )
    oanda_api_key: SecretStr | None = Field(None, validation_alias=AliasChoices("OANDA_API_KEY"))
    oanda_account_id: str | None = Field(None, validation_alias=AliasChoices("OANDA_ACCOUNT_ID"))
    oanda_timeout_s: float = Field(15.0, gt=0)

    # --- Live switch: must be set *in addition* to OANDA_ENV=live ---
    live: bool = False

    # --- LLM reviewer ---
    anthropic_api_key: SecretStr | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )
    llm_model: str = Field("claude-opus-5-5", validation_alias=AliasChoices("ZIA_MODEL"))
    llm_enabled: bool = True
    llm_effort: Literal["low", "medium", "high"] = "medium"
    llm_timeout_s: float = Field(60.0, gt=0)
    llm_max_retries: int = Field(1, ge=0, le=3)
    llm_min_confidence: float = Field(0.6, ge=0, le=1)
    llm_context_candles: int = Field(30, ge=5, le=200)

    # --- Market ---
    instruments: str = "EUR_USD,GBP_USD,USD_JPY"
    timeframe: str = "H1"
    candle_count: int = Field(300, ge=50, le=5000)

    # --- Risk (development defaults; not claims about optimal values) ---
    risk_per_trade_pct: float = Field(1.0, gt=0, le=5)
    max_daily_loss_pct: float = Field(3.0, gt=0, le=50)
    max_drawdown_pct: float = Field(10.0, gt=0, le=100)
    max_open_trades: int = Field(3, ge=1, le=20)
    max_spread_pips: float = Field(3.0, gt=0)
    min_stop_pips: float = Field(5.0, gt=0)
    min_reward_risk: float = Field(1.0, gt=0)
    max_units: int = Field(1_000_000, ge=1)
    max_margin_utilization_pct: float = Field(50.0, gt=0, le=100)
    max_price_age_s: float = Field(120.0, gt=0)
    session_start_hour_utc: int | None = Field(None, ge=0, le=23)
    session_end_hour_utc: int | None = Field(None, ge=0, le=24)

    # --- Kill switch: env flag or presence of this file halts new trades ---
    kill_switch: bool = False
    kill_switch_file: Path = Path("ZIA_KILL")

    # --- Runtime ---
    db_path: Path = Path("zia.db")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    poll_delay_s: float = Field(15.0, ge=0)

    @field_validator("timeframe")
    @classmethod
    def _tf(cls, v: str) -> str:
        if v not in GRANULARITY_SECONDS:
            raise ValueError(f"timeframe must be one of {sorted(GRANULARITY_SECONDS)}")
        return v

    @field_validator("instruments")
    @classmethod
    def _instruments(cls, v: str) -> str:
        items = [s.strip().upper() for s in v.split(",") if s.strip()]
        if not items:
            raise ValueError("at least one instrument is required")
        for item in items:
            inst.split(item)
        return ",".join(items)

    @property
    def instrument_list(self) -> list[str]:
        return self.instruments.split(",")

    @property
    def strategy(self) -> StrategyParams:
        return StrategyParams()

    def secret_values(self) -> list[str]:
        out = []
        for s in (self.oanda_api_key, self.anthropic_api_key):
            if s is not None and s.get_secret_value():
                out.append(s.get_secret_value())
        return out

    def kill_switch_active(self) -> bool:
        return self.kill_switch or self.kill_switch_file.exists()


def resolve_environment(settings: Settings) -> TradingEnvironment:
    """Decide practice vs live. Anything inconsistent fails closed.

    Live requires OANDA_ENV=live AND ZIA_LIVE=true. ZIA_LIVE=true with a practice
    endpoint is treated as a misconfiguration rather than silently ignored.
    """
    if settings.oanda_env == "practice" and not settings.live:
        return TradingEnvironment.PRACTICE
    if settings.oanda_env == "live" and settings.live:
        return TradingEnvironment.LIVE
    if settings.oanda_env == "live":
        raise LiveTradingNotPermitted("OANDA_ENV=live but ZIA_LIVE is not true; refusing to start.")
    raise ConfigError("ZIA_LIVE=true while OANDA_ENV=practice is ambiguous; refusing to start.")


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]
