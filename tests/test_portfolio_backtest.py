"""Multi-pair backtests on one shared account."""

from datetime import timedelta
from itertools import pairwise

import pytest
from typer.testing import CliRunner

from zia import cli
from zia.backtest import run_backtest, run_portfolio_backtest
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.journal import Journal
from zia.risk import RiskLimits

PAIRS = {
    "EUR_USD": synthetic_candles(1500, start_price=1.10, seed=7),
    "GBP_USD": synthetic_candles(1500, start_price=1.27, seed=8),
    "USD_JPY": synthetic_candles(1500, start_price=150.0, seed=9),
}


@pytest.fixture(scope="module")
def portfolio():
    return run_portfolio_backtest(PAIRS, params=StrategyParams(), limits=RiskLimits())


def test_per_pair_breakdown_adds_up(portfolio):
    r = portfolio
    assert [p.instrument for p in r.per_pair] == list(PAIRS)
    assert r.instruments == list(PAIRS)
    assert sum(p.trades for p in r.per_pair) == r.trades
    assert sum(p.signals for p in r.per_pair) == r.signals
    assert sum(p.net_pnl for p in r.per_pair) == pytest.approx(r.net_pnl)
    assert r.candles == 4500
    assert all(p.trades > 0 for p in r.per_pair)
    rows = dict(r.rows())
    assert rows["Instruments"] == "EUR_USD,GBP_USD,USD_JPY"
    assert rows["Account"].startswith("one shared account")


def test_portfolio_is_deterministic(portfolio):
    again = run_portfolio_backtest(PAIRS, params=StrategyParams(), limits=RiskLimits())
    assert again == portfolio


def test_limits_apply_across_pairs():
    journal = Journal(":memory:")
    r = run_portfolio_backtest(
        PAIRS, params=StrategyParams(), limits=RiskLimits(max_open_trades=1), journal=journal
    )
    assert r.risk_rejections > 0
    reasons = [
        row["risk_reason"]
        for row in journal.conn.execute(
            "SELECT risk_reason FROM evaluations WHERE outcome='risk_rejected'"
        )
    ]
    assert any("max_open_trades" in reason for reason in reasons)
    # Never more than one trade open at a time across the whole account.
    trades = journal.trades("backtest")
    assert trades
    spans = sorted((t["open_time"], t["close_time"] or "9999") for t in trades)
    for (_, prev_close), (nxt_open, _) in pairwise(spans):
        assert nxt_open >= prev_close


def test_pairs_with_different_date_ranges():
    late_start = synthetic_candles(1500, start_price=1.27, seed=8)[500:]
    r = run_portfolio_backtest(
        {"EUR_USD": PAIRS["EUR_USD"], "GBP_USD": late_start},
        params=StrategyParams(),
        limits=RiskLimits(),
    )
    by_pair = {p.instrument: p for p in r.per_pair}
    assert by_pair["GBP_USD"].candles == 1000
    assert r.start == PAIRS["EUR_USD"][0].time
    assert r.end == PAIRS["EUR_USD"][-1].time


def test_single_pair_wrapper_matches_portfolio_of_one():
    candles = PAIRS["EUR_USD"]
    single = run_backtest("EUR_USD", candles, params=StrategyParams(), limits=RiskLimits())
    one = run_portfolio_backtest({"EUR_USD": candles}, params=StrategyParams(), limits=RiskLimits())
    assert single == one
    assert len(single.per_pair) == 1
    assert dict(single.rows())["Account"] == "single pair"


def test_cross_pair_converts_with_other_pairs_prices():
    eur_gbp = synthetic_candles(1500, start_price=0.86, seed=10)
    r = run_portfolio_backtest(
        {"EUR_GBP": eur_gbp, "GBP_USD": PAIRS["GBP_USD"]},
        params=StrategyParams(),
        limits=RiskLimits(),
    )
    by_pair = {p.instrument: p for p in r.per_pair}
    assert by_pair["EUR_GBP"].trades > 0  # sized via GBP_USD's price


def test_empty_input_rejected():
    with pytest.raises(ValueError):
        run_portfolio_backtest({}, params=StrategyParams(), limits=RiskLimits())


def test_step_is_respected():
    candles = synthetic_candles(400, seed=7, step=timedelta(hours=4))
    r = run_backtest(
        "EUR_USD", candles, params=StrategyParams(), limits=RiskLimits(),
        step=timedelta(hours=4), timeframe="H4",
    )  # fmt: skip
    assert r.candles == 400


# -- CLI ----------------------------------------------------------------------------------
runner = CliRunner()


def write_csv(path, candles):
    lines = ["time,open,high,low,close"] + [
        f"{c.time.isoformat()},{c.open},{c.high},{c.low},{c.close}" for c in candles
    ]
    path.write_text("\n".join(lines))
    return path


def test_cli_pairs_synthetic():
    res = runner.invoke(cli.app, ["backtest", "--synthetic", "800", "--pairs", "EUR_USD,USD_JPY"])
    assert res.exit_code == 0, res.output
    assert "Per pair (shared account)" in res.output
    assert "USD_JPY" in res.output


def test_cli_csv_per_pair(tmp_path):
    eur = write_csv(tmp_path / "eur.csv", PAIRS["EUR_USD"][:600])
    jpy = write_csv(tmp_path / "jpy.csv", PAIRS["USD_JPY"][:600])
    res = runner.invoke(cli.app, ["backtest", "--csv", f"EUR_USD={eur}", "--csv", f"usd_jpy={jpy}"])
    assert res.exit_code == 0, res.output
    assert "EUR_USD,USD_JPY" in res.output


def test_cli_single_bare_csv_still_works(tmp_path):
    eur = write_csv(tmp_path / "eur.csv", PAIRS["EUR_USD"][:600])
    res = runner.invoke(cli.app, ["backtest", "--pair", "EUR_USD", "--csv", str(eur)])
    assert res.exit_code == 0, res.output
    assert "Per pair" not in res.output


@pytest.mark.parametrize(
    "args",
    [
        ["--pairs", "EUR_USD,USD_JPY", "--csv", "only.csv"],  # bare path, several pairs
        ["--pairs", "EUR_USD,USD_JPY", "--csv", "EUR_USD=a.csv"],  # USD_JPY missing
        ["--pairs", "EUR_USD,EUR_USD", "--synthetic", "300"],  # duplicate
        ["--pairs", "EURUSD", "--synthetic", "300"],  # malformed
        ["--csv", "EUR_USD=a.csv", "--csv", "EUR_USD=b.csv"],  # two files, one pair
    ],
)
def test_cli_rejects_bad_pair_arguments(args, tmp_path):
    (tmp_path / "a.csv").write_text("time,open,high,low,close\n")
    res = runner.invoke(cli.app, ["backtest", *args])
    assert res.exit_code == 2, res.output


def test_cli_bare_csv_path_containing_equals(tmp_path):
    odd = tmp_path / "run=1"
    odd.mkdir()
    eur = write_csv(odd / "eur.csv", PAIRS["EUR_USD"][:600])
    res = runner.invoke(cli.app, ["backtest", "--pair", "EUR_USD", "--csv", str(eur)])
    assert res.exit_code == 0, res.output


def test_parse_csv_args_validates_pair_names():
    with pytest.raises(ValueError):
        cli._parse_csv_args(["EUR_XYZ1=a.csv", "USD_JPY=b.csv"], ["EUR_USD", "USD_JPY"])
    assert cli._parse_csv_args(["eur_usd=a=b.csv"], ["EUR_USD"]) == {"EUR_USD": cli.Path("a=b.csv")}
    # Left side isn't a pair name: with several pairs this is an error, not a pair "DATA/X".
    with pytest.raises(ValueError):
        cli._parse_csv_args(["data/x=1.csv", "USD_JPY=b.csv"], ["EUR_USD", "USD_JPY"])


def test_run_backtest_rejects_unknown_options():
    with pytest.raises(TypeError):
        run_backtest(
            "EUR_USD", PAIRS["EUR_USD"], params=StrategyParams(), limits=RiskLimits(), bogus=1
        )
