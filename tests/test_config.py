import pytest

from zia.config import (
    ConfigError,
    LiveTradingNotPermitted,
    Settings,
    load_settings,
    resolve_environment,
)
from zia.models import TradingEnvironment


def test_defaults_are_practice(monkeypatch):
    s = load_settings()
    assert s.oanda_env == "practice"
    assert s.live is False
    assert resolve_environment(s) is TradingEnvironment.PRACTICE
    assert s.instrument_list == ["EUR_USD", "GBP_USD", "USD_JPY"]
    assert s.timeframe == "H1"
    assert s.llm_model == "claude-opus-5-5"


def test_live_env_without_flag_refused(monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "live")
    with pytest.raises(LiveTradingNotPermitted):
        resolve_environment(load_settings())


def test_live_flag_without_live_env_is_ambiguous(monkeypatch):
    monkeypatch.setenv("ZIA_LIVE", "true")
    with pytest.raises(ConfigError):
        resolve_environment(load_settings())


def test_both_set_resolves_live(monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "live")
    monkeypatch.setenv("ZIA_LIVE", "true")
    assert resolve_environment(load_settings()) is TradingEnvironment.LIVE


def test_unknown_env_value_rejected(monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "production")
    with pytest.raises(ValueError):
        load_settings()


def test_env_vars_and_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "ZIA_INSTRUMENTS=eur_usd, usd_jpy\nZIA_RISK_PER_TRADE_PCT=0.5\nZIA_MODEL=some-model\n"
    )
    s = load_settings()
    assert s.instrument_list == ["EUR_USD", "USD_JPY"]
    assert s.risk_per_trade_pct == 0.5
    assert s.llm_model == "some-model"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ZIA_INSTRUMENTS", "EURUSD"),
        ("ZIA_TIMEFRAME", "H2"),
        ("ZIA_RISK_PER_TRADE_PCT", "25"),
        ("ZIA_MAX_OPEN_TRADES", "0"),
    ],
)
def test_invalid_values_rejected(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        load_settings()


def test_secrets_are_masked_in_repr(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "super-secret-oanda-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    s = Settings()
    assert "super-secret-oanda-token" not in repr(s)
    assert "sk-ant-secret-value" not in repr(s)
    assert set(s.secret_values()) == {"super-secret-oanda-token", "sk-ant-secret-value"}


def test_kill_switch_file(tmp_path):
    s = load_settings()
    assert not s.kill_switch_active()
    (tmp_path / "ZIA_KILL").write_text("")
    assert s.kill_switch_active()


def test_kill_switch_env(monkeypatch):
    monkeypatch.setenv("ZIA_KILL_SWITCH", "true")
    assert load_settings().kill_switch_active()
