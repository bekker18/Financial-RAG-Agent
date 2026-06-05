from typing import List

import typer
from rich.console import Console
from rich.markdown import Markdown

from financial_agent.agent import FinancialResearchAgent
from financial_agent.ingest import ingest_sec_filings

app = typer.Typer()
console = Console()


@app.command()
def ingest_sec(
    ticker: str = typer.Option(..., "--ticker", "-t"),
    form: List[str] = typer.Option(["10-K", "10-Q"], "--form", "-f"),
    limit: int = typer.Option(4, "--limit", "-l"),
    reset: bool = typer.Option(False, "--reset"),
):
    """
    Download SEC filings and index them into Chroma.
    """

    ingest_sec_filings(
        ticker=ticker,
        forms=form,
        limit=limit,
        reset=reset,
    )


@app.command()
def ask(
    question: str = typer.Argument(...),
):
    """
    Ask the financial research agent a question.
    """

    agent = FinancialResearchAgent()
    answer = agent.answer(question)
    console.print(Markdown(answer))


if __name__ == "__main__":
    app()
