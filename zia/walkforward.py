"""Walk-forward optimization of the strategy parameters.

Rolling windows over the candle timeline: each fold grid-searches strategy parameters on a
training window, then trades the next (unseen) test window with the winner. Only the
test windows count as results, which shows how the optimize-then-trade *process* would
have done out of sample. Risk limits and cost assumptions stay fixed throughout.

Objective on each training window: return % divided by max drawdown % (drawdown floored at
1% so a tiny drawdown can't blow up the score). Parameter sets with fewer than
``min_trades`` trades are skipped. Ties go to the earlier grid entry (deterministic).
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from zia.backtest import BacktestResult, run_portfolio_backtest
from zia.config import StrategyParams
from zia.models import Candle
from zia.risk import RiskLimits

DRAWDOWN_FLOOR_PCT = 1.0


def parse_grid(items: Sequence[str]) -> dict[str, list[Any]]:
    """Parse ``name=v1,v2`` items into typed value lists for StrategyParams fields."""
    fields = StrategyParams.model_fields
    grid: dict[str, list[Any]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"grid entry {item!r} must look like name=v1,v2")
        name, raw = (part.strip() for part in item.split("=", 1))
        if name not in fields:
            raise ValueError(f"unknown strategy parameter {name!r}; choose from {sorted(fields)}")
        if name in grid:
            raise ValueError(f"{name} given twice in the grid")
        cast = int if fields[name].annotation is int else float
        try:
            values = [cast(v) for v in raw.split(",") if v.strip()]
        except ValueError as exc:
            raise ValueError(f"bad value in {item!r}: {exc}") from exc
        if not values:
            raise ValueError(f"no values given for {name}")
        grid[name] = list(dict.fromkeys(values))
    if not grid:
        raise ValueError("the grid is empty")
    return grid


def expand_grid(base: StrategyParams, grid: Mapping[str, Sequence[Any]]) -> list[StrategyParams]:
    """Every valid combination, in grid order. Invalid ones (e.g. fast >= slow) are dropped."""
    names = list(grid)
    out = []
    for values in itertools.product(*(grid[n] for n in names)):
        try:
            out.append(
                StrategyParams(**{**base.model_dump(), **dict(zip(names, values, strict=True))})
            )
        except ValidationError:
            continue
    return out


def score(result: BacktestResult) -> float:
    return result.return_pct / max(result.max_drawdown_pct, DRAWDOWN_FLOOR_PCT)


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    combos_tested: int
    combos_qualified: int
    best_params: dict[str, Any] | None  # only the grid parameters
    train_score: float | None
    train_return_pct: float | None
    test: BacktestResult | None  # None when no parameter set qualified


@dataclass(frozen=True)
class WalkForwardResult:
    instruments: list[str]
    grid: dict[str, list[Any]]
    train_candles: int
    test_candles: int
    min_trades: int
    folds: list[Fold] = field(default_factory=list)

    @property
    def traded_folds(self) -> list[Fold]:
        return [f for f in self.folds if f.test is not None]

    @property
    def oos_return_pct(self) -> float:
        """Test-window returns compounded in order (each fold restarts from equity)."""
        growth = 1.0
        for f in self.traded_folds:
            growth *= 1 + f.test.return_pct / 100
        return (growth - 1) * 100

    @property
    def worst_fold_drawdown_pct(self) -> float:
        return max((f.test.max_drawdown_pct for f in self.traded_folds), default=0.0)

    @property
    def oos_trades(self) -> int:
        return sum(f.test.trades for f in self.traded_folds)

    @property
    def oos_win_rate(self) -> float:
        wins = sum(f.test.wins for f in self.traded_folds)
        closed = wins + sum(f.test.losses for f in self.traded_folds)
        return wins / closed if closed else 0.0

    def parameter_stability(self) -> dict[str, Counter]:
        """How often each value was chosen, per grid parameter."""
        out: dict[str, Counter] = {name: Counter() for name in self.grid}
        for f in self.folds:
            for name, value in (f.best_params or {}).items():
                out[name][value] += 1
        return out


def make_folds(
    times: Sequence[datetime], train: int, test: int
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Rolling (train_start, train_end, test_start, test_end) by candle-open time.

    Windows are counted in timeline steps; the window slides forward by ``test`` each
    fold. A final, shorter test window is kept if it has at least half of ``test``.
    """
    times = sorted(set(times))
    folds = []
    start = 0
    while start + train < len(times):
        test_lo = start + train
        test_hi = min(test_lo + test, len(times))
        if test_hi - test_lo < max(1, test // 2):
            break
        folds.append((times[start], times[test_lo - 1], times[test_lo], times[test_hi - 1]))
        start += test
    return folds


def _window(
    candles_by_pair: Mapping[str, list[Candle]], lo: datetime, hi: datetime
) -> dict[str, list[Candle]]:
    return {p: [c for c in cs if lo <= c.time <= hi] for p, cs in candles_by_pair.items()}


def _run(args: tuple) -> BacktestResult | None:
    """Top-level so ProcessPoolExecutor can pickle it."""
    candles, params, limits, kwargs, trade_from = args
    if not any(candles.values()):
        return None
    return run_portfolio_backtest(
        candles, params=params, limits=limits, trade_from=trade_from, **kwargs
    )


def walk_forward(
    candles_by_pair: Mapping[str, list[Candle]],
    *,
    base_params: StrategyParams,
    grid: Mapping[str, Sequence[Any]],
    limits: RiskLimits,
    train: int = 2000,
    test: int = 500,
    min_trades: int = 5,
    workers: int = 1,
    **backtest_kwargs: Any,
) -> WalkForwardResult:
    """Rolling walk-forward. ``backtest_kwargs`` go to ``run_portfolio_backtest``
    (balance, spread_pips, slippage_pips, commission_per_100k, swap_*, step, timeframe)."""
    if train <= 0 or test <= 0:
        raise ValueError("train and test windows must be positive")
    if "reviewer" in backtest_kwargs:
        raise ValueError("walk-forward runs without LLM review")
    combos = expand_grid(base_params, grid)
    if not combos:
        raise ValueError("no valid parameter combination in the grid")
    longest_warmup = max(c.warmup for c in combos)
    if train <= longest_warmup:
        raise ValueError(
            f"train window ({train}) must exceed the strategy warm-up ({longest_warmup})"
        )
    times = [c.time for cs in candles_by_pair.values() for c in cs]
    folds = make_folds(times, train, test)
    if not folds:
        raise ValueError(f"need more than {train} candles for one fold (have {len(set(times))})")

    timeline = sorted(set(times))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    mapper = pool.map if pool else map
    result = WalkForwardResult(
        instruments=list(candles_by_pair),
        grid={k: list(v) for k, v in grid.items()},
        train_candles=train,
        test_candles=test,
        min_trades=min_trades,
    )
    try:
        for i, (tr_lo, tr_hi, te_lo, te_hi) in enumerate(folds):
            train_data = _window(candles_by_pair, tr_lo, tr_hi)
            jobs = [(train_data, p, limits, backtest_kwargs, None) for p in combos]
            scored = []
            for params, res in zip(combos, mapper(_run, jobs), strict=True):
                if res is not None and res.trades >= min_trades:
                    scored.append((score(res), params, res))
            if not scored:
                result.folds.append(
                    Fold(i, tr_lo, tr_hi, te_lo, te_hi, len(combos), 0, None, None, None, None)
                )
                continue
            best_score, best, best_res = max(scored, key=lambda s: s[0])  # first wins ties

            # Test with warm-up history immediately before the test window.
            idx = timeline.index(te_lo)
            warm_lo = timeline[max(0, idx - best.warmup)]
            test_data = _window(candles_by_pair, warm_lo, te_hi)
            test_res = _run((test_data, best, limits, backtest_kwargs, te_lo))
            result.folds.append(
                Fold(
                    index=i,
                    train_start=tr_lo,
                    train_end=tr_hi,
                    test_start=te_lo,
                    test_end=te_hi,
                    combos_tested=len(combos),
                    combos_qualified=len(scored),
                    best_params={k: getattr(best, k) for k in grid},
                    train_score=best_score,
                    train_return_pct=best_res.return_pct,
                    test=test_res,
                )
            )
    finally:
        if pool:
            pool.shutdown()
    return result
