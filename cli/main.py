"""vnquant command-line interface (typer + rich).

Commands
--------
    vnquant research    Run the alpha research loop and print IC / FDR / DSR table.
    vnquant backtest    Run the event-driven backtest and print performance summary.
    vnquant audit       Run the data-leakage audit checks on synthetic data.
    vnquant pipeline    Run the full end-to-end pipeline.
    vnquant ask         Ask the RAG research assistant a grounded question.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vnquant.alpha.research import research
from vnquant.backtest.engine import run_backtest
from vnquant.data.synthetic import generate_panel, to_wide
from vnquant.default_config import Config
from vnquant.pipeline import run_pipeline

app = typer.Typer(
    add_completion=False,
    help="vnquant — institutional-grade quant research engine (showcase edition).",
)
console = Console()


def _banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]vnquant[/bold cyan] — Vietnam Equity Quant Engine\n"
            "[dim]showcase edition · synthetic data · textbook factors[/dim]",
            border_style="cyan",
        )
    )


@app.command(name="research")
def research_cmd(seed: int = typer.Option(42, help="RNG seed for synthetic data")):
    """Run the alpha research loop with multiple-testing correction."""
    _banner()
    cfg = Config()
    cfg.data.seed = seed
    panel, sym2sector = generate_panel(cfg.data)
    reports = research(
        to_wide(panel, "close"), to_wide(panel, "volume"),
        to_wide(panel, "high"), to_wide(panel, "low"),
        sym2sector, cfg.alpha,
    )
    table = Table(title="Alpha Research — IC / Multiple-Testing")
    for col in ["Factor", "mean IC", "ICIR", "t-stat", "p-value", "FDR pass", "DSR prob"]:
        table.add_column(col, justify="right")
    for name, rep in sorted(reports.items(), key=lambda kv: -abs(kv[1].ic.mean_ic)):
        ic = rep.ic
        table.add_row(
            name, f"{ic.mean_ic:+.4f}", f"{ic.icir:+.3f}", f"{ic.t_stat:+.2f}",
            f"{ic.p_value:.3f}",
            "[green]yes[/green]" if rep.survives_fdr else "[red]no[/red]",
            f"{rep.deflated_sr_prob:.2f}",
        )
    console.print(table)


@app.command()
def backtest(seed: int = typer.Option(42, help="RNG seed for synthetic data")):
    """Run the event-driven backtest and print a performance summary."""
    _banner()
    cfg = Config()
    cfg.data.seed = seed
    panel, sym2sector = generate_panel(cfg.data)
    result = run_backtest(panel, sym2sector, cfg)
    table = Table(title="Backtest Summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.summary().items():
        table.add_row(k, str(v))
    console.print(table)


@app.command()
def audit(seed: int = typer.Option(42)):
    """Run the data-leakage audit checks."""
    _banner()
    out = run_pipeline(_cfg(seed))
    color = {"OK": "green", "WARNING": "yellow", "CRITICAL": "red"}[out.audit_severity.value]
    console.print(f"Audit severity: [{color}]{out.audit_severity.value}[/{color}]")


@app.command()
def pipeline(seed: int = typer.Option(42)):
    """Run the full end-to-end pipeline (data -> audit -> alpha -> risk -> backtest)."""
    _banner()
    out = run_pipeline(_cfg(seed))
    console.print(f"Audit severity : {out.audit_severity.value}")
    console.print(f"Risk shrinkage : {out.risk_shrinkage:.3f}")
    n_pass = sum(r.survives_fdr for r in out.factor_reports.values())
    console.print(f"Factors passing FDR: {n_pass}/{len(out.factor_reports)}")
    table = Table(title="Backtest Summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in out.backtest.summary().items():
        table.add_row(k, str(v))
    console.print(table)


def _cfg(seed: int) -> Config:
    cfg = Config()
    cfg.data.seed = seed
    return cfg


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question for the research assistant"),
    seed: int = typer.Option(42, help="RNG seed for the pipeline run to ground answers on"),
    backend: str = typer.Option("offline", help="RAG backend: 'offline' or 'openai'"),
    ground: bool = typer.Option(
        True, help="Run the pipeline first so answers are grounded on live run facts"
    ),
):
    """Ask the retrieval-augmented research assistant a grounded question.

    Indexes the engine's methodology notes plus (optionally) live pipeline run-facts, then
    retrieves and answers ONLY from that context — abstaining when support is too weak.
    """
    from vnquant.assistant import build_assistant  # noqa: PLC0415

    _banner()
    cfg = _cfg(seed)
    cfg.assistant.backend = backend
    out = run_pipeline(cfg) if ground else None
    assistant = build_assistant(out, cfg.assistant)
    answer = assistant.ask(question)

    body = answer.text
    if answer.sources and not answer.abstained:
        body += "\n\n[dim]sources: " + ", ".join(answer.sources) + "[/dim]"
    border = "yellow" if answer.abstained else "green"
    console.print(
        Panel(
            body,
            title=f"vnquant assistant · {backend} · {assistant.n_chunks} chunks indexed",
            border_style=border,
        )
    )


if __name__ == "__main__":
    app()
