"""Walk-forward optimization."""

from dataclasses import replace
from datetime import timedelta

import pytest
from typer.testing import CliRunner

from zia import cli
from zia.backtest import run_portfolio_backtest
from zia.config import StrategyParams
from zia.data import synthetic_candles
from zia.risk import RiskLimits
from zia.walkforward import expand_grid, make_folds, parse_grid, score, walk_forward

CANDLES = synthetic_candles(1200, seed=7)
DATA = {"EUR_USD": CANDLES}
GRID = {"ema_fast": [10, 20]}
KW = dict(base_params=StrategyParams(), limits=RiskLimits(), train=600, test=200, min_trades=1)


@pytest.fixture(scope="module")
def wf():
    return walk_forward(DATA, grid=GRID, **KW)


# -- grid ---------------------------------------------------------------------------------
def test_parse_grid_types_and_order():
    g = parse_grid(["ema_fast=10, 20,10", "sl_atr_mult=1.5,2"])
    assert g == {"ema_fast": [10, 20], "sl_atr_mult": [1.5, 2.0]}
    assert isinstance(g["ema_fast"][0], int) and isinstance(g["sl_atr_mult"][1], float)


@pytest.mark.parametrize(
    "items",
    [
        ["ema_fast"],  # no '='
        ["nope=1,2"],  # unknown parameter
        ["ema_fast=ten"],  # bad value
        ["ema_fast=1", "ema_fast=2"],  # duplicate
        ["ema_fast="],  # no values
        [],  # empty grid
    ],
)
def test_parse_grid_rejects(items):
    with pytest.raises(ValueError):
        parse_grid(items)


def test_expand_grid_drops_invalid_combinations():
    combos = expand_grid(StrategyParams(), {"ema_fast": [10, 60], "ema_slow": [50, 100]})
    pairs = [(c.ema_fast, c.ema_slow) for c in combos]
    assert pairs == [(10, 50), (10, 100), (60, 100)]  # (60, 50) is invalid: fast >= slow


# -- folds --------------------------------------------------------------------------------
def test_make_folds_rolling_windows():
    times = [c.time for c in synthetic_candles(3000)]
    folds = make_folds(times, 1500, 500)
    assert len(folds) == 3
    for i, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(folds):
        assert tr_lo == times[i * 500]
        assert tr_hi == times[i * 500 + 1499]
        assert te_lo == times[i * 500 + 1500]
        assert te_hi == times[min(i * 500 + 1999, 2999)]


def test_make_folds_partial_last_window():
    times = [c.time for c in synthetic_candles(1800)]
    assert len(make_folds(times, 1000, 500)) == 2  # second test window has 300 >= 250
    times = [c.time for c in synthetic_candles(1700)]
    assert len(make_folds(times, 1000, 500)) == 1  # 200 < 250: dropped
    assert make_folds(times, 1700, 500) == []


# -- walk-forward -------------------------------------------------------------------------
def test_windows_are_out_of_sample_and_contiguous(wf):
    assert len(wf.folds) == 3
    for f in wf.folds:
        assert f.train_end < f.test_start
        assert f.best_params is not None and f.best_params["ema_fast"] in GRID["ema_fast"]
        assert f.test is not None and f.test.start == f.test_start
    for a, b in zip(wf.folds, wf.folds[1:], strict=False):
        assert b.test_start > a.test_end


def test_best_params_maximize_training_objective(wf):
    f = wf.folds[0]
    train = {"EUR_USD": [c for c in CANDLES if f.train_start <= c.time <= f.train_end]}
    results = {
        fast: run_portfolio_backtest(
            train, params=StrategyParams(ema_fast=fast), limits=RiskLimits()
        )
        for fast in GRID["ema_fast"]
    }
    # Only parameter sets with at least min_trades training trades compete.
    scores = {k: score(r) for k, r in results.items() if r.trades >= KW["min_trades"]}
    assert f.combos_qualified == len(scores)
    assert f.best_params["ema_fast"] == max(scores, key=scores.get)
    assert f.train_score == pytest.approx(max(scores.values()))


def test_no_lookahead_into_future_data(wf):
    """Changing prices after fold 1's test window must not change fold 1."""
    cut = wf.folds[0].test_end
    altered = [
        replace(c, open=c.open * 1.05, high=c.high * 1.05, low=c.low * 1.05, close=c.close * 1.05)
        if c.time > cut
        else c
        for c in CANDLES
    ]
    other = walk_forward({"EUR_USD": altered}, grid=GRID, **KW)
    assert other.folds[0] == wf.folds[0]


def test_oos_summary(wf):
    growth = 1.0
    for f in wf.traded_folds:
        growth *= 1 + f.test.return_pct / 100
    assert wf.oos_return_pct == pytest.approx((growth - 1) * 100)
    assert wf.oos_trades == sum(f.test.trades for f in wf.traded_folds)
    stability = wf.parameter_stability()
    assert sum(stability["ema_fast"].values()) == len(wf.folds)


def test_parallel_matches_serial(wf):
    assert walk_forward(DATA, grid=GRID, workers=2, **KW) == wf


def test_min_trades_can_leave_folds_untraded():
    r = walk_forward(DATA, grid=GRID, **{**KW, "min_trades": 10_000})
    assert r.traded_folds == []
    assert all(f.best_params is None and f.combos_qualified == 0 for f in r.folds)
    assert r.oos_return_pct == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"train": 100},  # shorter than warm-up
        {"train": 1200},  # no room for a test window
        {"test": 0},
        {"reviewer": object()},
    ],
)
def test_walk_forward_rejects(kw):
    with pytest.raises(ValueError):
        walk_forward(DATA, grid=GRID, **{**KW, **kw})


def test_trade_from_uses_history_only_for_warmup():
    start = CANDLES[600].time
    r = run_portfolio_backtest(DATA, params=StrategyParams(), limits=RiskLimits(), trade_from=start)
    assert r.start == start
    assert r.per_pair[0].candles == 600


# -- CLI ----------------------------------------------------------------------------------
runner = CliRunner()


def test_cli_walkforward_synthetic():
    res = runner.invoke(
        cli.app,
        ["walkforward", "--synthetic", "1200", "--train", "600", "--test", "200",
         "--grid", "ema_fast=10,20", "--min-trades", "1", "--workers", "1"],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    assert "Out-of-sample summary" in res.output
    assert "WALK-FORWARD" in res.output


@pytest.mark.parametrize(
    "args",
    [
        ["--grid", "bogus=1"],
        ["--train", "5000"],  # not enough data for one fold
    ],
)
def test_cli_walkforward_errors(args):
    res = runner.invoke(cli.app, ["walkforward", "--synthetic", "1200", "--workers", "1", *args])
    assert res.exit_code == 2, res.output


def test_step_passthrough_for_other_timeframes():
    h4 = synthetic_candles(1200, seed=3, step=timedelta(hours=4))
    r = walk_forward({"EUR_USD": h4}, grid=GRID, step=timedelta(hours=4), timeframe="H4", **KW)
    assert len(r.folds) == 3
