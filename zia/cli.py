"""Command line interface: ``zia run | backtest | status | close-all``."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from zia.config import (
    GRANULARITY_SECONDS,
    ConfigError,
    Settings,
    load_settings,
    resolve_environment,
)
from zia.logging_setup import setup_logging
from zia.models import TradingEnvironment

app = typer.Typer(add_completion=False, help="Zia: risk-governed Forex trading agent.")
console = Console()
log = logging.getLogger("zia.cli")

LIVE_PHRASE = "I UNDERSTAND THIS USES REAL MONEY"


# -- factories (patched in tests) -------------------------------------------------------------
def build_broker(settings: Settings, env: TradingEnvironment):
    from zia.broker.oanda import OandaBroker

    if env not in (TradingEnvironment.PRACTICE, TradingEnvironment.LIVE):
        raise ConfigError(f"no broker for {env}")
    if settings.oanda_api_key is None or not settings.oanda_account_id:
        raise ConfigError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set (see .env.example)")
    return OandaBroker(
        environment=env,
        api_key=settings.oanda_api_key.get_secret_value(),
        account_id=settings.oanda_account_id,
        timeout_s=settings.oanda_timeout_s,
    )


def build_reviewer(settings: Settings):
    from zia.llm_reviewer import AnthropicReviewer, DisabledReviewer

    if not settings.llm_enabled:
        return DisabledReviewer()
    return AnthropicReviewer(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        effort=settings.llm_effort,
        context_candles=settings.llm_context_candles,
    )


def build_journal(settings: Settings):
    from zia.journal import Journal

    return Journal(settings.db_path)


# -- helpers ----------------------------------------------------------------------------------
def _setup() -> tuple[Settings, TradingEnvironment]:
    try:
        settings = load_settings()
        env = resolve_environment(settings)
    except (ConfigError, ValueError) as exc:
        console.print(f"[bold red]Configuration refused:[/] {exc}")
        raise typer.Exit(2) from exc
    setup_logging(settings.log_level, settings.log_json, settings.secret_values())
    return settings, env


def _banner(env: TradingEnvironment) -> None:
    if env is TradingEnvironment.LIVE:
        console.print("[bold white on red] MODE: LIVE — REAL MONEY [/]")
    else:
        console.print(f"[bold black on green] MODE: {env.value.upper()} [/]")


def _confirm_live(action: str) -> bool:
    console.print(f"[bold red]You are about to {action} against a LIVE account.[/]")
    typed = typer.prompt(f'Type "{LIVE_PHRASE}" to continue', default="", show_default=False)
    return typed.strip() == LIVE_PHRASE


# -- commands ---------------------------------------------------------------------------------
@app.command()
def run(
    once: bool = typer.Option(False, "--once", help="Run a single decision cycle and exit."),
) -> None:
    """Run the trading agent (PRACTICE by default)."""
    from zia.agent import Agent
    from zia.risk import RiskGovernor, RiskLimits

    settings, env = _setup()
    _banner(env)
    live_confirmed = False
    if env is TradingEnvironment.LIVE:
        live_confirmed = _confirm_live("run the trading agent")
        if not live_confirmed:
            console.print("Live confirmation not given. Exiting.")
            raise typer.Exit(1)
    if not settings.llm_enabled:
        console.print("[yellow]LLM review disabled (ZIA_LLM_ENABLED=false).[/]")
    elif settings.anthropic_api_key is None:
        console.print(
            "[yellow]ANTHROPIC_API_KEY is not set: unless the SDK finds other credentials, "
            "every signal will be rejected (fail closed).[/]"
        )

    try:
        broker = build_broker(settings, env)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc
    journal = build_journal(settings)
    agent = Agent(
        broker=broker,
        reviewer=build_reviewer(settings),
        governor=RiskGovernor(RiskLimits.from_settings(settings)),
        journal=journal,
        environment=env,
        instruments=settings.instrument_list,
        timeframe=settings.timeframe,
        strategy_params=settings.strategy,
        candle_count=settings.candle_count,
        live_confirmed=live_confirmed,
        kill_switch=settings.kill_switch_active,
    )
    journal.event(
        "run_start",
        f"run once={once}",
        environment=env.value,
        data={
            "instruments": settings.instrument_list,
            "timeframe": settings.timeframe,
            "model": settings.llm_model,
        },
    )
    try:
        if once:
            outcomes = agent.run_cycle()
            table = Table(title="Cycle result")
            for col in ("Instrument", "Outcome", "Detail"):
                table.add_column(col)
            for o in outcomes:
                table.add_row(o.instrument, o.outcome, o.detail[:120])
            console.print(table)
        else:
            agent.run_forever(settings.poll_delay_s)
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        broker.close()
        journal.close()


SYNTHETIC_START_PRICES = {"GBP_USD": 1.27}


def _parse_pairs(pair: str, pairs: str | None) -> list[str]:
    from zia import instruments as inst

    items = [p.strip().upper() for p in (pairs or pair).split(",") if p.strip()]
    for item in items:
        inst.split(item)  # raises ValueError for malformed names
    if len(set(items)) != len(items):
        raise ValueError("duplicate pair in --pairs")
    return items


def _parse_csv_args(csv: list[str], pairs: list[str]) -> dict[str, Path]:
    """Map pairs to CSV paths from repeated ``--csv PAIR=PATH`` (or one bare PATH).

    ``PAIR=`` is only recognised when the text before ``=`` is a pair name such as
    ``EUR_USD``, so a bare path that happens to contain ``=`` stays a path.
    """
    import re

    from zia import instruments as inst

    out: dict[str, Path] = {}
    for item in csv:
        head, sep, tail = item.partition("=")
        if sep and re.fullmatch(r"[A-Za-z]{3}_[A-Za-z]{3}", head.strip()):
            name, path = head.strip().upper(), tail
            inst.split(name)
        elif len(csv) == 1 and len(pairs) == 1:
            name, path = pairs[0], item
        else:
            raise ValueError(f"use --csv PAIR=PATH when backtesting several pairs (got {item!r})")
        if name in out:
            raise ValueError(f"two CSV files given for {name}")
        out[name] = Path(path)
    return out


def _load_candles(
    settings: Settings,
    pair: str,
    pairs: str | None,
    start: str | None,
    end: str | None,
    csv: list[str] | None,
    synthetic: int,
    seed: int,
    mode: str,
) -> tuple[dict[str, list], str]:
    """Load candles per pair from --synthetic, --csv or OANDA (--from). Exits on errors."""
    from zia import data

    try:
        pair_list = _parse_pairs(pair, pairs)
        csv_paths = _parse_csv_args(csv or [], pair_list)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc
    if csv_paths and pairs is None:
        pair_list = list(csv_paths)  # the CSV arguments name the pairs
    console.print(f"[bold black on cyan] MODE: {mode} (simulated, hypothetical) [/]")

    candles_by_pair: dict[str, list] = {}
    if synthetic:
        for i, p in enumerate(pair_list):
            start_price = 150.0 if p.endswith("JPY") else SYNTHETIC_START_PRICES.get(p, 1.10)
            candles_by_pair[p] = data.synthetic_candles(
                synthetic, start_price=start_price, seed=seed + i
            )
        source = f"SYNTHETIC random walk (seed={seed}) — not market data"
    elif csv_paths:
        missing = [p for p in pair_list if p not in csv_paths]
        if missing:
            console.print(f"[bold red]No --csv given for {', '.join(missing)}[/]")
            raise typer.Exit(2)
        for p in pair_list:
            candles_by_pair[p] = data.load_csv(csv_paths[p])
        source = "CSV " + ", ".join(f"{p}={csv_paths[p]}" for p in pair_list)
    elif start:
        try:
            env = resolve_environment(settings)
        except ConfigError as exc:
            console.print(f"[bold red]Configuration refused:[/] {exc}")
            raise typer.Exit(2) from exc
        try:
            broker = build_broker(settings, env)
        except ConfigError as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(2) from exc
        s = datetime.fromisoformat(start).replace(tzinfo=UTC)
        e = datetime.fromisoformat(end).replace(tzinfo=UTC) if end else None
        try:
            for p in pair_list:
                candles_by_pair[p] = broker.get_candles_range(p, settings.timeframe, s, e)
        finally:
            broker.close()
        source = f"OANDA {env.value} historical candles"
    else:
        console.print("Provide --from (OANDA), --csv PAIR=PATH or --synthetic N.")
        raise typer.Exit(2)

    empty = [p for p, c in candles_by_pair.items() if not c]
    if empty:
        console.print(f"No candles loaded for {', '.join(empty)}.")
        raise typer.Exit(1)
    return candles_by_pair, source


@app.command()
def backtest(
    pair: str = typer.Option("EUR_USD", "--pair", help="Single pair to backtest"),
    pairs: str | None = typer.Option(
        None, "--pairs", help="Comma-separated pairs sharing one account, e.g. EUR_USD,USD_JPY"
    ),
    start: str | None = typer.Option(None, "--from", help="Start date YYYY-MM-DD (OANDA data)"),
    end: str | None = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    csv: list[str] | None = typer.Option(
        None, "--csv", help="CSV (time,open,high,low,close) as PAIR=PATH; repeat per pair"
    ),
    synthetic: int = typer.Option(
        0, "--synthetic", help="Use N seeded random-walk candles (demo only, not market data)"
    ),
    seed: int = typer.Option(7, "--seed"),
    balance: float = typer.Option(10_000.0, "--balance"),
    spread_pips: float = typer.Option(1.0, "--spread-pips", min=0),
    slippage_pips: float = typer.Option(
        0.2, "--slippage-pips", min=0, help="Adverse pips on entries and stop-loss exits"
    ),
    commission: float = typer.Option(
        0.0, "--commission", min=0, help="Account currency per 100k units, per side"
    ),
    swap_long: float = typer.Option(
        0.0, "--swap-long", help="Annual % of notional for longs (negative = you pay)"
    ),
    swap_short: float = typer.Option(
        0.0, "--swap-short", help="Annual % of notional for shorts (negative = you pay)"
    ),
    llm: bool = typer.Option(False, "--llm", help="Call the LLM reviewer for every signal"),
) -> None:
    """Backtest the strategy + Risk Governor on historical candles (no LLM by default).

    Several pairs (--pairs, or repeated --csv PAIR=PATH) trade one shared account, so
    the Risk Governor's limits apply across the portfolio exactly as in ``zia run``.
    """
    from zia.backtest import run_portfolio_backtest
    from zia.risk import RiskLimits

    try:
        settings = load_settings()
    except ValueError as exc:
        console.print(f"[bold red]Configuration refused:[/] {exc}")
        raise typer.Exit(2) from exc
    setup_logging("WARNING", settings.log_json, settings.secret_values())
    candles_by_pair, source = _load_candles(
        settings, pair, pairs, start, end, csv, synthetic, seed, "BACKTEST"
    )

    reviewer = build_reviewer(settings) if llm else None
    result = run_portfolio_backtest(
        candles_by_pair,
        params=settings.strategy,
        limits=RiskLimits.from_settings(settings, require_llm_approval=llm),
        balance=balance,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_100k=commission,
        swap_long_pct=swap_long,
        swap_short_pct=swap_short,
        step=timedelta(seconds=GRANULARITY_SECONDS[settings.timeframe]),
        timeframe=settings.timeframe,
        reviewer=reviewer,
    )
    table = Table(title=f"Backtest — {source}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in result.rows():
        table.add_row(k, v)
    console.print(table)
    if len(result.per_pair) > 1:
        by_pair = Table(title="Per pair (shared account)")
        for col in ("Pair", "Signals", "Risk rej.", "Trades", "Win rate", "Net P&L", "Open"):
            by_pair.add_column(col, justify="left" if col == "Pair" else "right")
        for ps in result.per_pair:
            by_pair.add_row(
                ps.instrument,
                str(ps.signals),
                str(ps.risk_rejections),
                str(ps.trades),
                f"{ps.win_rate:.1%}",
                f"{ps.net_pnl:,.2f}",
                str(ps.open_at_end),
            )
        console.print(by_pair)
    console.print(
        "[dim]Backtest results are hypothetical, depend on the stated cost assumptions and are "
        "not evidence of future profitability.[/]"
    )


DEFAULT_GRID = ["ema_fast=10,20", "ema_slow=50,100"]


@app.command()
def walkforward(
    pair: str = typer.Option("EUR_USD", "--pair", help="Single pair"),
    pairs: str | None = typer.Option(
        None, "--pairs", help="Comma-separated pairs sharing one account"
    ),
    start: str | None = typer.Option(None, "--from", help="Start date YYYY-MM-DD (OANDA data)"),
    end: str | None = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    csv: list[str] | None = typer.Option(None, "--csv", help="PAIR=PATH; repeat per pair"),
    synthetic: int = typer.Option(
        0, "--synthetic", help="Use N seeded random-walk candles (demo only, not market data)"
    ),
    seed: int = typer.Option(7, "--seed"),
    grid: list[str] | None = typer.Option(
        None,
        "--grid",
        help="Strategy parameter values to search, e.g. ema_fast=10,20,30; repeat per "
        "parameter (default: ema_fast=10,20 ema_slow=50,100)",
    ),
    train: int = typer.Option(2000, "--train", min=1, help="Training window, in candles"),
    test: int = typer.Option(500, "--test", min=1, help="Test window and step, in candles"),
    min_trades: int = typer.Option(
        5, "--min-trades", min=0, help="Skip parameter sets with fewer training trades"
    ),
    workers: int = typer.Option(0, "--workers", min=0, help="Parallel processes (0 = one per CPU)"),
    balance: float = typer.Option(10_000.0, "--balance"),
    spread_pips: float = typer.Option(1.0, "--spread-pips", min=0),
    slippage_pips: float = typer.Option(0.2, "--slippage-pips", min=0),
    commission: float = typer.Option(0.0, "--commission", min=0),
    swap_long: float = typer.Option(0.0, "--swap-long"),
    swap_short: float = typer.Option(0.0, "--swap-short"),
) -> None:
    """Walk-forward optimization: grid-search strategy parameters on rolling training
    windows, then trade each following test window with the winner (no LLM).

    Only the out-of-sample test windows are reported as results.
    """
    import os

    from zia.risk import RiskLimits
    from zia.walkforward import parse_grid, walk_forward

    try:
        settings = load_settings()
    except ValueError as exc:
        console.print(f"[bold red]Configuration refused:[/] {exc}")
        raise typer.Exit(2) from exc
    setup_logging("WARNING", settings.log_json, settings.secret_values())
    try:
        search = parse_grid(grid or DEFAULT_GRID)
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc
    candles_by_pair, source = _load_candles(
        settings, pair, pairs, start, end, csv, synthetic, seed, "WALK-FORWARD"
    )
    try:
        result = walk_forward(
            candles_by_pair,
            base_params=settings.strategy,
            grid=search,
            limits=RiskLimits.from_settings(settings, require_llm_approval=False),
            train=train,
            test=test,
            min_trades=min_trades,
            workers=workers or os.cpu_count() or 1,
            balance=balance,
            spread_pips=spread_pips,
            slippage_pips=slippage_pips,
            commission_per_100k=commission,
            swap_long_pct=swap_long,
            swap_short_pct=swap_short,
            step=timedelta(seconds=GRANULARITY_SECONDS[settings.timeframe]),
            timeframe=settings.timeframe,
        )
    except ValueError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc

    folds = Table(title=f"Walk-forward folds — {source}")
    for col in (
        "Fold",
        "Test period",
        "Chosen parameters",
        "Train score",
        "Test return",
        "Test max DD",
        "Test trades",
    ):
        folds.add_column(col, justify="right" if col.startswith(("Train", "Test ")) else "left")
    for f in result.folds:
        chosen = (
            ", ".join(f"{k}={v:g}" for k, v in f.best_params.items())
            if f.best_params
            else f"none qualified (< {result.min_trades} trades)"
        )
        folds.add_row(
            str(f.index + 1),
            f"{f.test_start:%Y-%m-%d} -> {f.test_end:%Y-%m-%d}",
            chosen,
            f"{f.train_score:.2f}" if f.train_score is not None else "-",
            f"{f.test.return_pct:+.2f}%" if f.test else "not traded",
            f"{f.test.max_drawdown_pct:.2f}%" if f.test else "-",
            str(f.test.trades) if f.test else "0",
        )
    console.print(folds)

    summary = Table(title="Out-of-sample summary (test windows only)")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    stability = "; ".join(
        f"{name}: " + ", ".join(f"{v:g}x{n}" for v, n in counts.most_common())
        for name, counts in result.parameter_stability().items()
    )
    for k, v in [
        ("Instruments", ",".join(result.instruments)),
        ("Windows", f"train {result.train_candles} / test {result.test_candles} candles, rolling"),
        ("Objective", "return % / max drawdown % (drawdown floored at 1%)"),
        ("Parameter sets per fold", str(result.folds[0].combos_tested)),
        ("Folds traded / total", f"{len(result.traded_folds)} / {len(result.folds)}"),
        ("Compounded OOS return", f"{result.oos_return_pct:+.2f}%"),
        ("Worst fold drawdown", f"{result.worst_fold_drawdown_pct:.2f}%"),
        ("OOS trades / win rate", f"{result.oos_trades} / {result.oos_win_rate:.1%}"),
        ("Chosen values (value x folds)", stability),
    ]:
        summary.add_row(k, v)
    console.print(summary)
    console.print(
        "[dim]Walk-forward results are hypothetical. They test the optimization process on "
        "past data, not future profitability; parameters that change a lot between folds "
        "suggest the edge is not stable.[/]"
    )


@app.command()
def status() -> None:
    """Show mode, account, open positions and recent journal entries."""
    settings, env = _setup()
    _banner(env)
    console.print(f"Kill switch: {'ACTIVE' if settings.kill_switch_active() else 'inactive'}")
    console.print(
        f"Instruments: {settings.instruments}  Timeframe: {settings.timeframe}  "
        f"LLM: {settings.llm_model if settings.llm_enabled else 'disabled'}"
    )

    try:
        broker = build_broker(settings, env)
    except ConfigError as exc:
        console.print(f"[yellow]Broker not configured: {exc}[/]")
        broker = None
    if broker is not None:
        try:
            acct = broker.get_account()
            console.print(
                f"Account {acct.account_id}: balance {acct.balance:,.2f} {acct.currency}, "
                f"NAV {acct.nav:,.2f}, margin available {acct.margin_available:,.2f}"
            )
            positions = broker.get_open_positions()
            t = Table(title="Open positions")
            for col in ("Instrument", "Units", "Avg price", "Unrealized P&L"):
                t.add_column(col)
            for p in positions:
                t.add_row(
                    p.instrument, f"{p.units:,.0f}", f"{p.average_price}", f"{p.unrealized_pl:,.2f}"
                )
            console.print(t)
        except Exception as exc:
            console.print(f"[red]Broker error: {exc}[/]")
        finally:
            broker.close()

    journal = build_journal(settings)
    try:
        t = Table(title="Recent evaluations")
        for col in ("Time", "Env", "Instrument", "Candle", "Outcome", "Note"):
            t.add_column(col)
        for r in journal.recent_evaluations(10):
            t.add_row(
                r["ts"][:19],
                r["environment"],
                r["instrument"],
                (r["candle_time"] or "")[:16],
                r["outcome"],
                (r["note"] or "")[:60],
            )
        console.print(t)
    finally:
        journal.close()


@app.command("close-all")
def close_all(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation in PRACTICE mode."),
) -> None:
    """Close every open position. Requires confirmation; live requires the typed phrase."""
    settings, env = _setup()
    _banner(env)
    if env is TradingEnvironment.LIVE:
        if not _confirm_live("close ALL positions"):
            console.print("Live confirmation not given. Nothing closed.")
            raise typer.Exit(1)
    elif not yes and not typer.confirm(f"Close ALL open positions on the {env.value} account?"):
        console.print("Aborted. Nothing closed.")
        raise typer.Exit(1)

    try:
        broker = build_broker(settings, env)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(2) from exc
    journal = build_journal(settings)
    try:
        positions = broker.get_open_positions()
        journal.event(
            "close_all_requested",
            f"{len(positions)} open positions",
            level="WARNING",
            environment=env.value,
            data=[p.instrument for p in positions],
        )
        failures = 0
        for p in positions:
            result = broker.close_position(p.instrument)
            journal.event(
                "close_position",
                f"{p.instrument}: {result.status}",
                level="WARNING",
                environment=env.value,
                data={
                    "instrument": p.instrument,
                    "status": result.status,
                    "error": result.error,
                    "response": result.raw,
                },
            )
            console.print(f"{p.instrument}: {result.status} {result.error or ''}")
            failures += not result.filled
        if not positions:
            console.print("No open positions.")
        if failures:
            raise typer.Exit(1)
    finally:
        broker.close()
        journal.close()


if __name__ == "__main__":  # pragma: no cover
    app()
