"""Command line interface: `uv run kap <command>`.

Typer turns each decorated function into a sub-command; function arguments
become CLI arguments/options automatically (with --help for free).
"""

from __future__ import annotations

import logging
import statistics

import typer
from rich.console import Console
from rich.table import Table as RichTable

from kaprag.config import get_settings
from kaprag.logging_setup import setup_logging
from kaprag.utils.turkish import tr_upper

app = typer.Typer(help="BIST Report Analyst: hybrid-search RAG over KAP reports.",
                  no_args_is_help=True)
console = Console()
log = logging.getLogger("kaprag.cli")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    setup_logging("DEBUG" if verbose else None)


# --------------------------------------------------------------------- inspect
@app.command()
def inspect(ticker: str, year: int,
            samples: int = typer.Option(3, help="Sample chunks to print per type")) -> None:
    """Parse all PDFs of TICKER/YEAR and print parsing statistics + sample chunks."""
    from kaprag.ingestion.loader import DocumentLoader

    settings = get_settings()
    loader = DocumentLoader(settings)
    ticker = tr_upper(ticker)
    paths = [p for p in loader.discover(settings.paths.raw_dir / ticker)
             if p.stem.startswith(str(year))]
    if not paths:
        console.print(f"[red]No PDFs found under {settings.paths.raw_dir / ticker} for {year}.")
        raise typer.Exit(1)

    for path in paths:
        doc = loader.load(path)
        p, chunks = doc.parsed, doc.chunks
        text_chunks = [c for c in chunks if c.chunk_type == "text"]
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        lengths = [loader.counter.count(c.text) for c in chunks] or [0]

        stats = RichTable(title=f"{path.relative_to(settings.paths.raw_dir)}", show_header=False)
        stats.add_row("company", f"{doc.meta.company_name} ({doc.meta.ticker})")
        stats.add_row("year / period / type", f"{doc.meta.year} / {doc.meta.period} / {doc.meta.doc_type}")
        stats.add_row("pages", str(p.n_pages))
        stats.add_row("scanned pages", _fmt_pages(p.scanned_pages))
        stats.add_row("garbled pages", _fmt_pages(p.garbled_pages))
        stats.add_row("OCR'd pages", _fmt_pages(p.ocr_pages))
        stats.add_row("sections found", str(len(p.sections)))
        stats.add_row("tables found", f"{len(p.tables)} (layout-based: "
                      f"{sum(t.engine == 'layout' for t in p.tables)})")
        stats.add_row("chunks", f"{len(chunks)} (text {len(text_chunks)}, table {len(table_chunks)})")
        stats.add_row("avg / max chunk tokens", f"{statistics.mean(lengths):.0f} / {max(lengths)}")
        stats.add_row("removed header/footer", "; ".join(p.removed_header_footer[:6]))
        console.print(stats)
        console.print("[bold]First sections:[/] " + " · ".join(p.sections[:12]))
        for kind, group in (("text", text_chunks), ("table", table_chunks)):
            step = max(1, len(group) // (samples + 1))
            for c in group[step::step][:samples]:
                console.rule(f"{c.chunk_id} | {kind} | s.{c.page_start}-{c.page_end} | {c.section_title[:60]}")
                console.print(c.text[:900] + (" ..." if len(c.text) > 900 else ""), markup=False)


# ---------------------------------------------------------------- indexing
@app.command()
def ingest(
    path: str = typer.Argument(None, help="A PDF or a folder (default: data/raw/)"),
    ticker: str = typer.Option(None, help="Override ticker from the path"),
    year: int = typer.Option(None, help="Override year from the path"),
    doc_type: str = typer.Option(None, help="faaliyet_raporu | finansal_rapor | ozel_durum | other"),
    period: str = typer.Option(None, help="FY, Q1..Q4 or a date"),
    force: bool = typer.Option(False, help="Re-index even if unchanged"),
) -> None:
    """Parse, chunk, embed and index PDFs (unchanged documents are skipped)."""
    from pathlib import Path

    from kaprag.indexing.indexer import Indexer

    settings = get_settings()
    indexer = Indexer(settings)
    root = Path(path) if path else settings.paths.raw_dir
    paths = indexer.loader.discover(root) if root.is_dir() else [root]
    if not paths:
        console.print(f"[red]No PDFs found in {root}. Put files in data/raw/<TICKER>/<YEAR>_<doc_type>.pdf")
        raise typer.Exit(1)
    results = indexer.ingest(paths, tr_upper(ticker) if ticker else None, year, doc_type, period, force)
    table = RichTable("file", "doc", "status", "chunks", "sec", "note")
    for r in results:
        color = {"indexed": "green", "skipped": "cyan", "empty": "yellow"}.get(r.status, "red")
        table.add_row(Path(r.source_file).name, r.doc_key, f"[{color}]{r.status}", str(r.n_chunks),
                      f"{r.seconds:.0f}", r.message)
    console.print(table)
    console.print(f"Index now holds {len(indexer.store)} chunks ({indexer.dense.count()} vectors).")


@app.command(name="list")
def list_docs() -> None:
    """List indexed documents with chunk counts."""
    from kaprag.indexing.indexer import Indexer

    docs = Indexer(get_settings()).list_documents()
    if not docs:
        console.print("[yellow]Index is empty. Run `uv run kap ingest`.")
        return
    table = RichTable("ticker", "year", "doc_type", "period", "chunks", "pages", "tables", "file", "indexed_at")
    for d in docs:
        table.add_row(d["ticker"], str(d["year"]), d["doc_type"], d["period"], str(d["n_chunks"]),
                      str(d["n_pages"]), str(d["n_tables"]), d["source_file"], d["indexed_at"])
    console.print(table)


@app.command()
def remove(ticker: str, year: int, doc_type: str) -> None:
    """Remove a document (all its chunks) from the index."""
    from kaprag.indexing.indexer import Indexer

    removed = Indexer(get_settings()).remove(tr_upper(ticker), year, doc_type)
    console.print(f"Removed: {removed}" if removed else "[yellow]Nothing matched.")


# --------------------------------------------------------------------- ask
@app.command()
def ask(
    question: str,
    mode: str = typer.Option(None, help="dense | bm25 | hybrid | hybrid_rerank"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Only show retrieval results"),
    show_chunks: int = typer.Option(6, help="How many retrieved chunks to print"),
) -> None:
    """Ask a question from the command line."""
    from kaprag.pipeline import RAGPipeline

    resp = RAGPipeline().ask(question, mode=mode, generate=not no_llm)
    pq = resp.parsed_query
    console.print(f"[bold]Filters:[/] tickers={pq['tickers']} years={pq['years']} periods={pq['periods']} "
                  f"doc_types={pq['doc_types']} comparison={pq['is_comparison']} "
                  f"financial={pq['is_financial']}  mode={resp.retrieval_mode}")
    for note in pq["notes"]:
        console.print(f"[yellow]note:[/] {note}")
    table = RichTable("#", "source", "type", "section", "dense", "bm25", "fused", "rerank")
    for i, c in enumerate(resp.chunks[:show_chunks], 1):
        table.add_row(str(i), f"{c.citation} {c.group}", c.chunk_type, c.section_title[:35],
                      _rank(c.dense_rank, c.dense_score), _rank(c.bm25_rank, c.bm25_score),
                      _rank(c.fused_rank, c.fused_score), f"{c.rerank_score:.3f}" if c.rerank_score is not None else "-")
    console.print(table)
    if resp.answer:
        console.rule("Answer")
        console.print(resp.answer, markup=False)
        if resp.unverified_numbers:
            console.print(f"[red]Unverified numbers:[/] {resp.unverified_numbers}")
        else:
            console.print(f"[green]All {len(resp.numbers)} numbers verified against the context.")
    console.print(f"[dim]timings (ms): {resp.timings_ms}")


def _rank(rank: int | None, score: float | None) -> str:
    return "-" if rank is None else f"#{rank} ({score:.3f})"


def _fmt_pages(pages: list[int]) -> str:
    if not pages:
        return "-"
    shown = ", ".join(map(str, pages[:15]))
    return f"{len(pages)}: {shown}{' ...' if len(pages) > 15 else ''}"


if __name__ == "__main__":
    app()
