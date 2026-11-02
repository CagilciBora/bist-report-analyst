"""Evaluate retrieval strategies and answer quality on eval/eval_set.jsonl.

Usage:
  uv run python eval/run_eval.py                       # retrieval grid + generation
  uv run python eval/run_eval.py --no-generation       # retrieval only (fast)
  uv run python eval/run_eval.py --limit 5             # smoke test

Metrics
  Hit@k   share of answerable questions with >=1 relevant chunk in the top k
          (relevant = same ticker/year/doc_type and the expected page lies in
          the chunk's page range)
  MRR     mean of 1/rank of the first relevant chunk (0 if none in the top 6)
  Filter accuracy   parser extracted exactly the expected tickers / years
  Numeric faithfulness  share of expected numbers that appear in the answer
  Unverified rate   share of numbers in answers NOT found in the context
  Unanswerable handling  share of unanswerable questions answered with "not in the documents"
  Latency   mean and p95 per stage
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaprag.config import get_settings  # noqa: E402
from kaprag.indexing.bm25_index import BM25Index  # noqa: E402
from kaprag.logging_setup import setup_logging  # noqa: E402
from kaprag.pipeline import RAGPipeline  # noqa: E402
from kaprag.utils.numbers import extract_numbers, parse_tr_number  # noqa: E402

KS = (1, 3, 6)
REFUSAL_MARKERS = ("yer almıyor", "bulunmuyor", "bulunmamaktadır", "yer almamaktadır", "mevcut değil",
                   "bilgi yok", "not in the documents", "not contain", "does not contain", "not available",
                   "not provided", "no information", "bulunamadı", "içermiyor", "içermemektedir")


def load_eval(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk, expected: list[dict]) -> bool:
    return any(chunk.ticker == e["ticker"] and chunk.year == e["year"] and chunk.doc_type == e["doc_type"]
               and chunk.page_start <= e["page"] <= chunk.page_end for e in expected)


def p95(xs: list[float]) -> float:
    return float(np.percentile(xs, 95)) if xs else 0.0


def number_in_answer(expected: str, answer: str) -> bool:
    """Exact substring, or the same value (any formatting / unit scale) among the answer's numbers."""
    if expected in answer:
        return True
    target = parse_tr_number(expected)
    if target is None:
        return False
    for n in extract_numbers(answer, skip_years=False):
        if target.is_percent != n.is_percent:
            continue
        # compare at the precision of the expected string, allowing a different unit scale
        for scale in (1.0, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9):
            a, b = abs(n.absolute), abs(target.absolute) * scale
            if b and abs(a - b) <= max(0.5 * 10 ** (-target.decimals) * target.scale * scale, 1e-9 * b):
                return True
    return False


# ------------------------------------------------------------------ retrieval
def run_retrieval(pipe: RAGPipeline, items: list[dict], mode: str, boost: bool) -> dict:
    per_q, timings = [], {}
    for it in items:
        if it["type"] == "unanswerable":
            continue
        _, chunks, extra = pipe.retrieve(it["question"], mode=mode, boost_tables=None if boost else False)
        for k, v in extra["timings"].items():
            timings.setdefault(k, []).append(v * 1000)
        ranks = [i for i, rc in enumerate(chunks, 1) if is_relevant(rc.chunk, it["expected_pages"])]
        first = ranks[0] if ranks else None
        sides = {(e["ticker"], e["year"]) for e in it["expected_pages"]}
        covered = {(rc.chunk.ticker, rc.chunk.year) for rc in chunks if is_relevant(rc.chunk, it["expected_pages"])}
        per_q.append({"id": it["id"], "type": it["type"], "first_relevant_rank": first,
                      "side_coverage": len(covered & sides) / len(sides) if sides else 0.0})
    out = {f"hit@{k}": statistics.mean(q["first_relevant_rank"] is not None and q["first_relevant_rank"] <= k
                                       for q in per_q) for k in KS}
    out["mrr"] = statistics.mean(1 / q["first_relevant_rank"] if q["first_relevant_rank"] else 0 for q in per_q)
    comp = [q["side_coverage"] for q in per_q if q["type"] == "comparison"]
    out["comparison_side_coverage"] = statistics.mean(comp) if comp else None
    by_type = {}
    for t in sorted({q["type"] for q in per_q}):
        qs = [q for q in per_q if q["type"] == t]
        by_type[t] = {"n": len(qs), "hit@6": statistics.mean(q["first_relevant_rank"] is not None for q in qs)}
    out["by_type"] = by_type
    out["latency_ms"] = {k: {"mean": statistics.mean(v), "p95": p95(v)} for k, v in timings.items()}
    out["per_question"] = per_q
    return out


def filter_accuracy(pipe: RAGPipeline, items: list[dict]) -> dict:
    t_ok = y_ok = both = 0
    misses = []
    for it in items:
        p = pipe.parse(it["question"])
        tk = set(p.tickers) == set(it["tickers"])
        yr = set(p.years) == set(it["years"])
        t_ok += tk
        y_ok += yr
        both += tk and yr
        if not (tk and yr):
            misses.append({"id": it["id"], "parsed": [p.tickers, p.years], "expected": [it["tickers"], it["years"]]})
    n = len(items)
    return {"ticker_acc": t_ok / n, "year_acc": y_ok / n, "both_acc": both / n, "misses": misses}


# ----------------------------------------------------------------- generation
def run_generation(pipe: RAGPipeline, items: list[dict], mode: str) -> dict:
    rows, timings = [], {}
    for i, it in enumerate(items, 1):
        t = time.perf_counter()
        resp = pipe.ask(it["question"], mode=mode)
        print(f"  [{i}/{len(items)}] {it['id']} {time.perf_counter() - t:.0f}s", flush=True)
        for k, v in resp.timings_ms.items():
            timings.setdefault(k, []).append(v)
        ans = resp.answer
        refused = any(m in ans.lower() for m in REFUSAL_MARKERS)
        found = [n for n in it["expected_numbers"] if number_in_answer(n, ans)]
        rows.append({
            "id": it["id"], "type": it["type"], "answer": ans, "error": resp.error,
            "expected_numbers": it["expected_numbers"], "found_numbers": found,
            "n_numbers": len(resp.numbers), "unverified": resp.unverified_numbers,
            "refused": refused, "citations": resp.citations,
            "chunk_ids": [c.chunk_id for c in resp.chunks],  # lets eval/judge.py see the same context
        })
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unans = [r for r in rows if r["type"] == "unanswerable"]
    n_exp = sum(len(r["expected_numbers"]) for r in answerable)
    n_nums = sum(r["n_numbers"] for r in rows)
    return {
        "numeric_faithfulness": sum(len(r["found_numbers"]) for r in answerable) / n_exp if n_exp else None,
        "unverified_rate": sum(len(r["unverified"]) for r in rows) / n_nums if n_nums else 0.0,
        "unanswerable_handling": statistics.mean(r["refused"] for r in unans) if unans else None,
        "false_refusal_rate": statistics.mean(r["refused"] for r in answerable) if answerable else None,
        "citation_rate": statistics.mean(bool(r["citations"]) for r in answerable) if answerable else None,
        "errors": sum(bool(r["error"]) for r in rows),
        "latency_ms": {k: {"mean": statistics.mean(v), "p95": p95(v)} for k, v in timings.items()},
        "rows": rows,
    }


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "eval_set.jsonl"))
    ap.add_argument("--modes", default="dense,bm25,hybrid,hybrid_rerank")
    ap.add_argument("--no-generation", action="store_true")
    ap.add_argument("--gen-mode", default="hybrid_rerank")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    setup_logging("WARNING")
    items = load_eval(Path(args.eval_set))
    if args.limit:
        items = items[: args.limit]
    settings = get_settings()
    pipe = RAGPipeline(settings)

    # Rerank scores don't depend on the table boost: cache them so the boost
    # on/off runs don't pay for the cross-encoder twice.
    rr = pipe.retriever.reranker
    raw_score, cache = rr.score, {}

    def cached_score(question: str, passages: list[str]) -> list[float]:
        missing = [p for p in passages if (question, p) not in cache]
        if missing:
            cache.update({(question, p): s for p, s in zip(missing, raw_score(question, missing), strict=True)})
        return [cache[(question, p)] for p in passages]

    rr.score = cached_score

    bm25_plain = pipe.bm25
    bm25_stem = BM25Index(pipe.store, settings.bm25.model_copy(update={"stemming": True}),
                          settings.paths.index_dir / "bm25_stem")

    results = {"eval_set": args.eval_set, "n_questions": len(items),
               "embedding_model": settings.embedding.model_name, "reranker": settings.retrieval.reranker_model,
               "llm": f"{pipe.llm.name}/{pipe.llm.model}", "prompt_version": pipe.prompts.version,
               "filter_accuracy": filter_accuracy(pipe, items), "retrieval": {}}

    for mode in args.modes.split(","):
        stem_options = [False] if mode == "dense" else [False, True]
        for stem in stem_options:
            for boost in (False, True):
                name = f"{mode}|stem={'on' if stem else 'off'}|boost={'on' if boost else 'off'}"
                pipe.retriever.bm25 = bm25_stem if stem else bm25_plain
                t = time.perf_counter()
                results["retrieval"][name] = run_retrieval(pipe, items, mode, boost)
                r = results["retrieval"][name]
                print(f"{name:40s} hit@1 {r['hit@1']:.2f} hit@3 {r['hit@3']:.2f} hit@6 {r['hit@6']:.2f} "
                      f"mrr {r['mrr']:.3f}  ({time.perf_counter() - t:.0f}s)", flush=True)
    pipe.retriever.bm25 = bm25_plain

    if not args.no_generation:
        print(f"Generation with {results['llm']} in mode {args.gen_mode} ...", flush=True)
        results["generation"] = {"mode": args.gen_mode, "table_boost": settings.retrieval.table_boost_enabled,
                                 **run_generation(pipe, items, args.gen_mode)}

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"results_{stamp}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(results)
    (out_dir / "latest.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def render_markdown(r: dict) -> str:
    fa = r["filter_accuracy"]
    lines = [f"Eval set: {r['n_questions']} questions · embedding {r['embedding_model']} · "
             f"reranker {r['reranker']}", "",
             "| Config | Hit@1 | Hit@3 | Hit@6 | MRR | Comparison side coverage | Retrieval p95 (ms) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in r["retrieval"].items():
        lat = sum(v["p95"] for k, v in m["latency_ms"].items())
        cov = "–" if m["comparison_side_coverage"] is None else f"{m['comparison_side_coverage']:.2f}"
        lines.append(f"| {name} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@6']:.2f} | {m['mrr']:.3f} | "
                     f"{cov} | {lat:.0f} |")
    lines += ["", f"Filter accuracy: tickers {fa['ticker_acc']:.2f} · years {fa['year_acc']:.2f} · "
                  f"both {fa['both_acc']:.2f}"]
    g = r.get("generation")
    if g:
        def pct(x):
            return "–" if x is None else f"{x:.2f}"
        lines += ["", f"Generation ({r['llm']}, mode {g['mode']}, prompt v{r['prompt_version']}):", "",
                  "| Numeric faithfulness | Unverified-number rate | Unanswerable handling | False refusals | "
                  "Answers with citations | Errors |", "|---|---|---|---|---|---|",
                  f"| {pct(g['numeric_faithfulness'])} | {pct(g['unverified_rate'])} | "
                  f"{pct(g['unanswerable_handling'])} | {pct(g['false_refusal_rate'])} | "
                  f"{pct(g['citation_rate'])} | {g['errors']} |", "",
                  "| Stage | mean ms | p95 ms |", "|---|---|---|"]
        lines += [f"| {k} | {v['mean']:.0f} | {v['p95']:.0f} |" for k, v in g["latency_ms"].items()]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
