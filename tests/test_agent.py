import json
from datetime import timedelta

import pytest

from tests.conftest import APPROVE, REJECT, FakeReviewer, find_signal_series
from zia.agent import Agent
from zia.broker.base import BrokerError
from zia.broker.sim import SimBroker
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.journal import Journal
from zia.models import Side, TradingEnvironment
from zia.risk import RiskGovernor, RiskLimits

H = timedelta(hours=1)
P = TradingEnvironment.PRACTICE


@pytest.fixture
def setup(tmp_path):
    def _make(review=APPROVE, spread=1.0, kill=False, env=P, side=Side.BUY):
        prefix = find_signal_series(side)
        full = synthetic_candles(3000, seed=7)
        sim = SimBroker({"EUR_USD": full}, environment=env, spread_pips=spread)
        now = prefix[-1].time + H
        sim.advance_to(now)
        journal = Journal(tmp_path / "zia.db")
        reviewer = FakeReviewer(review)
        agent = Agent(
            broker=sim,
            reviewer=reviewer,
            governor=RiskGovernor(RiskLimits()),
            journal=journal,
            environment=P,
            instruments=["EUR_USD"],
            timeframe="H1",
            strategy_params=StrategyParams(),
            kill_switch=lambda: kill,
        )
        return agent, sim, journal, reviewer, now, full

    return _make


def test_signal_approved_and_filled(setup):
    agent, sim, journal, reviewer, now, _ = setup()
    [out] = agent.run_cycle(now)
    assert out.outcome == "filled", out.detail
    assert reviewer.calls == 1
    assert len(sim.trades) == 1
    t = sim.trades[0]
    assert t.stop_loss < t.open_price < t.take_profit

    row = journal.recent_evaluations(1)[0]
    assert row["outcome"] == "order_filled"
    assert row["llm_decision"] == "approve" and row["llm_rationale"] == "looks fine"
    assert row["risk_approved"] == 1
    assert row["units"] == t.units
    assert row["stop_loss"] == t.stop_loss and row["take_profit"] == t.take_profit
    assert json.loads(row["order_request_json"])["instrument"] == "EUR_USD"
    assert json.loads(row["signal_json"])["side"] == "buy"
    assert row["strategy_version"]
    assert row["trade_id"] == t.trade_id
    checks = json.loads(row["risk_checks_json"])
    assert all(c["passed"] for c in checks)
    [trade] = journal.trades(P.value)
    assert trade["state"] == "open"


def test_duplicate_candle_not_processed_twice(setup):
    agent, sim, journal, reviewer, now, _ = setup()
    agent.run_cycle(now)
    [out] = agent.run_cycle(now + timedelta(minutes=5))
    assert out.outcome == "duplicate"
    assert reviewer.calls == 1
    assert len(sim.trades) == 1
    assert len(journal.recent_evaluations(50)) == 1


def test_duplicate_prevention_survives_restart(setup, tmp_path):
    agent, sim, journal, reviewer, now, _ = setup()
    agent.run_cycle(now)
    journal.close()
    j2 = Journal(tmp_path / "zia.db")
    agent2 = Agent(
        broker=sim,
        reviewer=reviewer,
        governor=RiskGovernor(RiskLimits()),
        journal=j2,
        environment=P,
        instruments=["EUR_USD"],
        timeframe="H1",
        strategy_params=StrategyParams(),
    )
    [out] = agent2.run_cycle(now)
    assert out.outcome == "duplicate"
    assert len(sim.trades) == 1


def test_llm_rejection_blocks_trade(setup):
    agent, sim, journal, reviewer, now, _ = setup(review=REJECT)
    [out] = agent.run_cycle(now)
    assert out.outcome == "llm_rejected"
    assert sim.trades == []
    row = journal.recent_evaluations(1)[0]
    assert row["llm_decision"] == "reject" and row["risk_approved"] is None


def test_llm_approval_then_risk_rejection(setup):
    agent, sim, journal, reviewer, now, _ = setup(spread=10.0)
    [out] = agent.run_cycle(now)
    assert out.outcome == "risk_rejected"
    assert "max_spread" in out.detail
    assert reviewer.calls == 1
    assert sim.trades == []
    row = journal.recent_evaluations(1)[0]
    assert row["llm_decision"] == "approve"
    assert row["risk_approved"] == 0
    assert "max_spread" in row["risk_reason"]


def test_kill_switch_blocks_before_llm(setup):
    agent, sim, journal, reviewer, now, _ = setup(kill=True)
    [out] = agent.run_cycle(now)
    assert out.outcome == "kill_switch"
    assert reviewer.calls == 0
    assert sim.trades == []


def test_broker_order_failure_is_recorded(setup):
    agent, sim, journal, reviewer, now, _ = setup()
    sim.fail_next_order = "HTTP 503 upstream"
    [out] = agent.run_cycle(now)
    assert out.outcome == "error"
    row = journal.recent_evaluations(1)[0]
    assert row["execution_status"] == "error"
    assert "503" in row["error"]
    assert journal.trades() == []


def test_broker_data_failure_does_not_stop_other_instruments(setup, monkeypatch):
    agent, sim, journal, reviewer, now, _ = setup()
    agent.instruments = ["GBP_USD", "EUR_USD"]  # GBP_USD unknown to the sim broker
    outs = agent.run_cycle(now)
    assert [o.outcome for o in outs] == ["error", "filled"]
    ev = journal.conn.execute("SELECT * FROM events WHERE kind='cycle_error'").fetchall()
    assert len(ev) == 1


def test_environment_mismatch_halts(setup):
    agent, *_, now, _ = setup(env=TradingEnvironment.BACKTEST)
    with pytest.raises(RuntimeError):
        agent.run_cycle(now)


def test_trade_reconciled_after_close(setup):
    agent, sim, journal, reviewer, now, full = setup()
    agent.run_cycle(now)
    t = now
    for _ in range(500):
        t += H
        sim.advance_to(t)
        if not sim.trades[0].is_open:
            break
    assert not sim.trades[0].is_open
    agent.reconcile_trades(t)
    [row] = journal.trades(P.value)
    assert row["state"] == "closed"
    assert row["realized_pl"] == pytest.approx(sim.trades[0].realized_pl)
    assert journal.daily_pnl(P.value)[0]["trades"] == 1


def test_no_signal_is_journaled(setup):
    agent, sim, journal, reviewer, now, full = setup()
    agent.run_cycle(now)
    sim.advance_to(now + H)
    [out] = agent.run_cycle(now + H)
    assert out.outcome in ("no_signal", "risk_rejected")
    rows = journal.recent_evaluations(5)
    assert len(rows) == 2
    assert rows[0]["indicators_json"]


def test_next_wake_is_on_candle_boundary(setup):
    agent, *_, now, _ = setup()
    wake = agent.next_wake(now + timedelta(minutes=17), 15)
    assert wake == now + H + timedelta(seconds=15)


def test_broker_error_on_reconcile_is_logged(setup):
    agent, sim, journal, reviewer, now, _ = setup()
    agent.run_cycle(now)

    def boom(trade_id):
        raise BrokerError("down")

    sim.get_trade = boom
    agent.reconcile_trades(now)
    ev = journal.conn.execute("SELECT * FROM events WHERE kind='reconcile_error'").fetchall()
    assert len(ev) == 1


def test_daily_loss_baseline_set_without_signal(setup):
    agent, sim, journal, reviewer, now, _ = setup()
    day_key = f"day_start_equity:{P.value}:{(now - H).date().isoformat()}"
    agent.run_cycle(now - H)  # candle before the signal: no signal, but equity is marked
    assert journal.get_state(day_key) is not None
    assert journal.conn.execute("SELECT COUNT(*) FROM equity").fetchone()[0] == 1
