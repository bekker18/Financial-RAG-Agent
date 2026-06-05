from pathlib import Path

from bs4 import BeautifulSoup
from rich.console import Console

from financial_agent.chunking import chunk_text
from financial_agent.config import get_settings
from financial_agent.sec_client import SecClient
from financial_agent.vector_store import FinancialVectorStore

console = Console()


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return text


def ingest_sec_filings(
    ticker: str,
    forms: list[str],
    limit: int,
    reset: bool = False,
) -> None:
    """
    ingest_sec_filings("AAPL", ["10-K"], 2)
    Downloads 2 latest Apple 10-K filings, converts HTML to text, chunks text, embeds each chunk with Gemini, stores chunks in Chroma.
    """
    settings = get_settings()

    sec = SecClient()
    store = FinancialVectorStore()

    if reset:
        console.print("[yellow]Reseting vector database... [/yellow]")
        store.reset()

    filings = sec.get_recent_filings(
        ticker=ticker,
        forms=forms,
        limit=limit,
    )

    if not filings:
        console.print(f"[red]No filings found for {ticker}.[/red]")
        return

    for filing in filings:
        console.print(
            f"[cyan]Downloading {filing.ticker} {filing.form}"
            f"{filing.filing_date}...[/cyan]"
        )

        path = sec.download_filing(filing)

        html = Path(path).read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(html)

        metadata = filing.__dict__.copy()
        metadata["source_file"] = str(path)

        chunks = chunk_text(
            text=text,
            metadata=metadata,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        console.print(f"Created {len(chunks)} chunks.")

        store.add_chunks(chunks)

        console.print("[green]Stored in Chroma.[/green]")

    console.print("[bold green]Ingestion complete.[/bold green]")
