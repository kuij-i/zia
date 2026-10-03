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


@app.command()
def backtest(
    pair: str = typer.Option("EUR_USD", "--pair"),
    start: str | None = typer.Option(None, "--from", help="Start date YYYY-MM-DD (OANDA data)"),
    end: str | None = typer.Option(None, "--to", help="End date YYYY-MM-DD"),
    csv: Path | None = typer.Option(None, "--csv", help="CSV with time,open,high,low,close"),
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
    llm: bool = typer.Option(False, "--llm", help="Call the LLM reviewer for every signal"),
) -> None:
    """Backtest the strategy + Risk Governor on historical candles (no LLM by default)."""
    from zia import data
    from zia.backtest import run_backtest
    from zia.risk import RiskLimits

    try:
        settings = load_settings()
    except ValueError as exc:
        console.print(f"[bold red]Configuration refused:[/] {exc}")
        raise typer.Exit(2) from exc
    setup_logging("WARNING", settings.log_json, settings.secret_values())
    pair = pair.upper()
    console.print("[bold black on cyan] MODE: BACKTEST (simulated, hypothetical) [/]")

    if synthetic:
        candles = data.synthetic_candles(
            synthetic, start_price=150.0 if pair.endswith("JPY") else 1.10, seed=seed
        )
        source = f"SYNTHETIC random walk (seed={seed}) — not market data"
    elif csv:
        candles = data.load_csv(csv)
        source = f"CSV {csv}"
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
            candles = broker.get_candles_range(pair, settings.timeframe, s, e)
        finally:
            broker.close()
        source = f"OANDA {env.value} historical candles"
    else:
        console.print("Provide --from (OANDA), --csv PATH or --synthetic N.")
        raise typer.Exit(2)

    if not candles:
        console.print("No candles loaded.")
        raise typer.Exit(1)

    reviewer = build_reviewer(settings) if llm else None
    result = run_backtest(
        pair,
        candles,
        params=settings.strategy,
        limits=RiskLimits.from_settings(settings, require_llm_approval=llm),
        balance=balance,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        commission_per_100k=commission,
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
    console.print(
        "[dim]Backtest results are hypothetical, depend on the stated cost assumptions and are "
        "not evidence of future profitability.[/]"
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
