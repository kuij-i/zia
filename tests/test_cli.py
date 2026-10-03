import pytest
from typer.testing import CliRunner

from tests.conftest import APPROVE, FakeReviewer
from zia import cli
from zia.broker.sim import SimBroker
from zia.data import synthetic_candles
from zia.journal import Journal
from zia.models import TradingEnvironment

runner = CliRunner()


@pytest.fixture
def sim_factory(monkeypatch):
    """Route CLI broker construction to a SimBroker; record what was asked for."""
    made = {}

    def build(settings, env):
        candles = synthetic_candles(400, seed=7)
        sim = SimBroker({"EUR_USD": candles}, environment=env)
        sim.advance_to(candles[-1].time + (candles[1].time - candles[0].time))
        made["broker"], made["env"] = sim, env
        return sim

    monkeypatch.setattr(cli, "build_broker", build)
    monkeypatch.setattr(cli, "build_reviewer", lambda s: FakeReviewer(APPROVE))
    monkeypatch.setenv("ZIA_INSTRUMENTS", "EUR_USD")
    monkeypatch.setenv("ZIA_LOG_JSON", "false")
    return made


def test_run_once_practice(sim_factory, tmp_path):
    res = runner.invoke(cli.app, ["run", "--once"])
    assert res.exit_code == 0, res.output
    assert "PRACTICE" in res.output
    assert sim_factory["env"] is TradingEnvironment.PRACTICE
    j = Journal(tmp_path / "zia.db")
    assert len(j.recent_evaluations()) == 1
    kinds = [r["kind"] for r in j.conn.execute("SELECT kind FROM events")]
    assert "run_start" in kinds


def test_run_refuses_live_without_flag(sim_factory, monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "live")
    res = runner.invoke(cli.app, ["run", "--once"])
    assert res.exit_code == 2
    assert "refus" in res.output.lower()
    assert "broker" not in sim_factory


def test_run_refuses_ambiguous_live_flag(sim_factory, monkeypatch):
    monkeypatch.setenv("ZIA_LIVE", "true")
    res = runner.invoke(cli.app, ["run", "--once"])
    assert res.exit_code == 2
    assert "broker" not in sim_factory


def test_run_live_requires_typed_confirmation(sim_factory, monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "live")
    monkeypatch.setenv("ZIA_LIVE", "true")
    res = runner.invoke(cli.app, ["run", "--once"], input="yes\n")
    assert res.exit_code == 1
    assert "LIVE" in res.output
    assert "broker" not in sim_factory


def test_run_without_credentials_fails_cleanly(monkeypatch):
    monkeypatch.setenv("ZIA_LOG_JSON", "false")
    res = runner.invoke(cli.app, ["run", "--once"])
    assert res.exit_code == 2
    assert "OANDA_API_KEY" in res.output


def test_status(sim_factory):
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0, res.output
    assert "PRACTICE" in res.output and "Kill switch: inactive" in res.output
    assert "SIM" in res.output


def test_close_all_requires_confirmation(sim_factory, tmp_path):
    res = runner.invoke(cli.app, ["close-all"], input="n\n")
    assert res.exit_code == 1
    assert "Nothing closed" in res.output


def test_close_all_practice_with_yes_is_journaled(sim_factory, tmp_path):
    res = runner.invoke(cli.app, ["close-all", "--yes"])
    assert res.exit_code == 0, res.output
    j = Journal(tmp_path / "zia.db")
    kinds = [r["kind"] for r in j.conn.execute("SELECT kind FROM events")]
    assert "close_all_requested" in kinds


def test_close_all_live_ignores_yes_and_needs_phrase(sim_factory, monkeypatch):
    monkeypatch.setenv("OANDA_ENV", "live")
    monkeypatch.setenv("ZIA_LIVE", "true")
    res = runner.invoke(cli.app, ["close-all", "--yes"], input="y\n")
    assert res.exit_code == 1
    assert "broker" not in sim_factory


def test_backtest_synthetic(monkeypatch):
    res = runner.invoke(cli.app, ["backtest", "--pair", "EUR_USD", "--synthetic", "1200"])
    assert res.exit_code == 0, res.output
    assert "BACKTEST" in res.output and "SYNTHETIC" in res.output
    assert "not evidence of future" in res.output.replace("\n", " ")


def test_backtest_requires_source():
    res = runner.invoke(cli.app, ["backtest"])
    assert res.exit_code == 2


def test_backtest_cost_options():
    res = runner.invoke(
        cli.app,
        ["backtest", "--synthetic", "1200", "--slippage-pips", "0.5", "--commission", "3.5"],
    )
    assert res.exit_code == 0, res.output
    assert "0.5 pips on entries" in res.output
    assert "3.50 per 100k" in res.output


def test_backtest_rejects_negative_costs():
    res = runner.invoke(cli.app, ["backtest", "--synthetic", "300", "--commission", "-1"])
    assert res.exit_code != 0


def test_backtest_swap_options():
    res = runner.invoke(
        cli.app,
        ["backtest", "--synthetic", "1200", "--swap-long", "-2.5", "--swap-short", "0.8"],
    )
    assert res.exit_code == 0, res.output
    assert "long -2.5% / short +0.8%" in res.output
