"""LLM-as-a-judge: Claude reviews the eval set and grades the RAG answers.

Two jobs:
  uv run python eval/judge.py --review-dataset   # check the 40 eval questions themselves
  uv run python eval/judge.py                    # grade answers in eval/results/results_*.json

Why a judge? The deterministic metrics (Hit@k, number matching, refusal
keywords) are objective but blind to meaning: they cannot grade a risk
summary, cannot tell that "55.614 TL" is the wrong line item, and count
"180,4 milyar TL ... bilgi belgelerde yok" as a refusal. A strong LLM with a
fixed rubric can.

How we keep the judge honest without a human in the loop:
  * it grades against the GOLD evidence pages from the eval set, never its own knowledge
  * a stronger model judges than the one that answered (Claude Opus vs Qwen 3B)
  * verdicts must match a JSON Schema (no free-form parsing)
  * its verdicts are cross-checked against the deterministic metrics; the
    agreement rate and every disagreement are reported, so drift is visible

Backend: `llm.judge_backend` in config.yaml (default `claude_code` = your
logged-in Claude subscription through the Claude Code CLI; no API key needed).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaprag.chunking.models import Chunk  # noqa: E402
from kaprag.config import get_settings  # noqa: E402
from kaprag.generation.llm import make_llm  # noqa: E402
from kaprag.indexing.bm25_index import BM25Index  # noqa: E402
from kaprag.indexing.chunk_store import ChunkStore  # noqa: E402
from kaprag.indexing.filters import SearchFilter  # noqa: E402
from kaprag.logging_setup import setup_logging  # noqa: E402

MAX_CHARS = 3000  # per chunk shown to the judge

# --------------------------------------------------------------------- schemas
ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_correct", "faithfulness", "unsupported_claims", "citations",
                 "refused", "refusal_appropriate", "score", "explanation"],
    "properties": {
        "answer_correct": {"type": "string",
                           "enum": ["correct", "partially_correct", "incorrect"]},
        "faithfulness": {"type": "number", "minimum": 0, "maximum": 1},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "string", "enum": ["all_supported", "some_supported", "none_supported",
                                                 "no_citations"]},
        "refused": {"type": "boolean"},
        "refusal_appropriate": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "explanation": {"type": "string"},
    },
}

DATASET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "answerable_from_evidence", "expected_numbers_correct",
                 "metadata_correct", "question_clear", "issues", "suggested_fix"],
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "needs_fix", "drop"]},
        "answerable_from_evidence": {"type": "boolean"},
        "expected_numbers_correct": {"type": "boolean"},
        "metadata_correct": {"type": "boolean"},
        "question_clear": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggested_fix": {"type": "string"},
    },
}

ANSWER_SYSTEM = """You are a strict evaluator of a retrieval-augmented QA system over Turkish company \
reports (annual reports and audited financial statements of Borsa Istanbul companies).
Judge ONLY with the evidence given in the prompt. Never use your own knowledge of these companies.

Fields:
- answer_correct: compare the ANSWER with the REFERENCE EVIDENCE and EXPECTED NUMBERS.
  "correct" = the facts the question asks for are right. Equivalent formatting/units count as right
  (1,2 milyar TL = 1.200 milyon TL = 1.200.000 bin TL). A right figure for the wrong line item, year
  or company is wrong. "partially_correct" = some asked facts right, some missing or wrong.
  For an UNANSWERABLE question, "correct" = the answer says the information is not in the documents
  and invents nothing.
- faithfulness: share (0-1) of the answer's factual claims that are supported by the RETRIEVED
  CONTEXT (what the system actually saw). Disclaimer lines are not claims. A pure refusal = 1.0.
- unsupported_claims: the claims that the retrieved context does not support (short quotes).
- citations: whether citation tags like [THYAO 2025 finansal_rapor s.11] point to retrieved context
  entries that support the sentence they follow.
- refused: the answer states that the (main) information is not available.
- refusal_appropriate: true if it refused and the retrieved context really lacks the answer, or if it
  did not refuse and the context contains the answer.
- score: 1 (useless/wrong) to 5 (correct, complete, faithful, well cited).
- explanation: at most two sentences."""

DATASET_SYSTEM = """You review test items for the evaluation set of a RAG system over Turkish company \
reports. Each item was written by someone reading the REFERENCE EVIDENCE shown with it.
Judge ONLY with the evidence given. Fields:
- answerable_from_evidence: the evidence contains the answer. For items of type "unanswerable" you get
  the CLOSEST PASSAGES from the index instead; set false if none of them answers the question
  (that is what an unanswerable item needs).
- expected_numbers_correct: every expected number is the right answer to the question (not a
  neighbouring figure, other year or other line item) and appears in the evidence. true if the list is empty
  and the question is not numeric.
- metadata_correct: "tickers" are the companies the question is about and "years" are the fiscal
  years it asks about (relative phrases like "son iki yıl" / "geçen yıl" are relative to 2025).
- question_clear: a careful analyst would understand it the same way and there is one right answer.
- verdict: "ok", "needs_fix" (fixable problem, describe the fix) or "drop" (cannot be fixed).
- issues: short list of problems; suggested_fix: concrete corrected field values, or "" if ok."""


# --------------------------------------------------------------------- helpers
def fmt_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(none)"
    return "\n\n---\n\n".join(f"{c.citation} ({c.chunk_type}, s.{c.page_start}-{c.page_end})\n"
                              f"{c.text[:MAX_CHARS]}" for c in chunks)


def gold_chunks(store: ChunkStore, item: dict, per_page: int = 4, limit: int = 14) -> list[Chunk]:
    """Chunks covering the expected pages of an eval item (the reference evidence).

    Every expected page gets its best chunks, so the judge never misses a page.
    "Best" = contains an expected number; we also match on the digits alone,
    because the source may write "18,1* milyar TL" where the eval says "18,1 milyar TL".
    """
    cores = [m.group(0) for n in item["expected_numbers"] if (m := re.search(r"\d[\d.,]*", n))]
    # Text questions have no numbers: use the stored evidence quote and the question's words.
    quote = re.sub(r"\s+", " ", item.get("evidence", ""))[:60].strip()
    words = {w for w in re.findall(r"\w{4,}", item["question"].lower())}

    def relevance(c: Chunk) -> float:
        text = re.sub(r"\s+", " ", c.text)
        return (10 * bool(quote and quote in text)
                + 2 * sum(n in text for n in item["expected_numbers"]) + sum(k in text for k in cores)
                + 0.1 * sum(w in text.lower() for w in words))

    picked: list[Chunk] = []
    for e in item["expected_pages"]:
        on_page = [c for c in store.chunks.values()
                   if c.ticker == e["ticker"] and c.year == e["year"] and c.doc_type == e["doc_type"]
                   and c.page_start <= e["page"] <= c.page_end]
        on_page.sort(key=lambda c: -relevance(c))
        picked += [c for c in on_page[:per_page] if c not in picked]
    return picked[:limit]


def run_parallel(fn, items: list, workers: int) -> list:
    done = 0

    def wrapped(x):
        nonlocal done
        try:
            return fn(x)
        except Exception as exc:  # one failed call must not kill the whole run
            return {"error": str(exc)[:300]}
        finally:
            done += 1
            print(f"  judged {done}/{len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(wrapped, items))


# ------------------------------------------------------------- dataset review
def review_dataset(llm, items: list[dict], store: ChunkStore, workers: int) -> dict:
    bm25 = BM25Index(store, get_settings().bm25, get_settings().paths.index_dir / "bm25")

    def review(it: dict) -> dict:
        if it["type"] == "unanswerable":
            hits = bm25.search(it["question"], SearchFilter(), 8)
            evidence = [store.get(cid) for cid, _ in hits]
            label = "CLOSEST PASSAGES IN THE INDEX (keyword search, all companies)"
        else:
            evidence = gold_chunks(store, it)
            label = "REFERENCE EVIDENCE (the expected pages)"
        user = (f"ITEM\n{json.dumps({k: it[k] for k in ('id', 'question', 'type', 'tickers', 'years', 'expected_pages', 'expected_numbers')}, ensure_ascii=False)}\n\n"
                f"{label}\n{fmt_chunks(evidence)}")
        return llm.generate_json(DATASET_SYSTEM, user, DATASET_SCHEMA)

    verdicts = run_parallel(review, items, workers)
    stamp = time.strftime("%Y-%m-%d")
    for it, v in zip(items, verdicts, strict=True):
        it["llm_review"] = {**v, "model": f"{llm.name}/{llm.model}", "date": stamp}
    counts = {k: sum(v.get("verdict") == k for v in verdicts) for k in ("ok", "needs_fix", "drop")}
    counts["errors"] = sum("error" in v for v in verdicts)
    return {"counts": counts, "items": [{"id": it["id"], **it["llm_review"]} for it in items]}


# ------------------------------------------------------------ answer grading
def contexts_for(rows: list[dict], items: dict, gen: dict) -> dict[str, list[str]]:
    """Chunk ids each answer saw. Old result files lack them: re-run retrieval with the same settings."""
    if all(r.get("chunk_ids") for r in rows):
        return {r["id"]: r["chunk_ids"] for r in rows}
    import shutil
    import tempfile

    from kaprag.indexing.dense_index import DenseIndex
    from kaprag.pipeline import RAGPipeline

    boost = gen.get("table_boost", True)  # files written before this field existed used boost=on
    print(f"Result file has no chunk ids: re-running retrieval (mode {gen['mode']}, table boost {boost}) ...",
          flush=True)
    # Embedded Qdrant allows one process per folder and the API may be running,
    # so we search a throwaway copy of the vector index instead of the original.
    settings = get_settings()
    tmp = Path(tempfile.mkdtemp(prefix="kap_judge_"))
    shutil.copytree(settings.paths.index_dir / "qdrant", tmp / "qdrant")
    pipe = RAGPipeline(settings, dense=DenseIndex(tmp / "qdrant"))
    pipe.retriever.cfg.table_boost_enabled = boost
    out = {}
    for i, r in enumerate(rows, 1):
        _, chunks, _ = pipe.retrieve(items[r["id"]]["question"], mode=gen["mode"])
        out[r["id"]] = [rc.chunk.chunk_id for rc in chunks]
        print(f"  retrieved {i}/{len(rows)}", flush=True)
    pipe.dense.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def judge_answers(llm, results: dict, items: dict, store: ChunkStore, workers: int) -> dict:
    gen = results["generation"]
    rows = gen["rows"]
    ctx = contexts_for(rows, items, gen)

    def grade(r: dict) -> dict:
        it = items[r["id"]]
        retrieved = [store.get(cid) for cid in ctx[r["id"]] if cid in store.chunks]
        user = (f"QUESTION ({it['type']}): {it['question']}\n"
                f"EXPECTED NUMBERS: {it['expected_numbers'] or '-'}\n\n"
                f"REFERENCE EVIDENCE (gold pages):\n{fmt_chunks(gold_chunks(store, it))}\n\n"
                f"RETRIEVED CONTEXT (what the system saw):\n{fmt_chunks(retrieved)}\n\n"
                f"ANSWER TO GRADE:\n{r['answer']}")
        return llm.generate_json(ANSWER_SYSTEM, user, ANSWER_SCHEMA)

    verdicts = run_parallel(grade, rows, workers)
    graded = [{"id": r["id"], "type": r["type"], **v} for r, v in zip(rows, verdicts, strict=True)]
    ok = [g for g in graded if "error" not in g]
    credit = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

    def mean(xs):
        xs = list(xs)
        return statistics.mean(xs) if xs else None

    metrics = {
        "n": len(graded), "errors": len(graded) - len(ok),
        "accuracy": mean(credit[g["answer_correct"]] for g in ok),
        "faithfulness": mean(g["faithfulness"] for g in ok),
        "citations_all_supported": mean(g["citations"] == "all_supported" for g in ok),
        "refusal_appropriate": mean(g["refusal_appropriate"] for g in ok),
        "mean_score": mean(g["score"] for g in ok),
        "by_type": {t: {"n": len(gs), "accuracy": mean(credit[g["answer_correct"]] for g in gs),
                        "faithfulness": mean(g["faithfulness"] for g in gs)}
                    for t in sorted({g["type"] for g in ok})
                    for gs in [[g for g in ok if g["type"] == t]]},
    }

    # Automatic calibration: does the judge agree with the deterministic metrics?
    by_id = {r["id"]: r for r in rows}
    num_pairs, ref_pairs = [], []
    for g in ok:
        r = by_id[g["id"]]
        if r["expected_numbers"]:
            det = len(r["found_numbers"]) == len(r["expected_numbers"])
            num_pairs.append((g["id"], det, g["answer_correct"] == "correct"))
        ref_pairs.append((g["id"], r["refused"], g["refused"]))
    metrics["calibration"] = {
        "numbers_agreement": mean(a == b for _, a, b in num_pairs),
        "numbers_disagreements": [{"id": i, "deterministic_all_found": a, "judge_correct": b}
                                  for i, a, b in num_pairs if a != b],
        "refusal_agreement": mean(a == b for _, a, b in ref_pairs),
        "refusal_disagreements": [{"id": i, "keyword_refused": a, "judge_refused": b}
                                  for i, a, b in ref_pairs if a != b],
    }
    return {"source_results": results.get("_path"), "judge": f"{llm.name}/{llm.model}",
            "generator": results.get("llm"), "metrics": metrics, "graded": graded}


def render_md(j: dict) -> str:
    m, c = j["metrics"], j["metrics"]["calibration"]

    def f(x):
        return "–" if x is None else f"{x:.2f}"

    lines = [f"Judge: {j['judge']} · graded answers of {j['generator']} · {m['n']} questions "
             f"({m['errors']} judge errors)", "",
             "| Accuracy | Faithfulness | Citations fully supported | Refusal appropriate | Mean score (1-5) |",
             "|---|---|---|---|---|",
             f"| {f(m['accuracy'])} | {f(m['faithfulness'])} | {f(m['citations_all_supported'])} | "
             f"{f(m['refusal_appropriate'])} | {f(m['mean_score'])} |", "",
             "| Type | n | Accuracy | Faithfulness |", "|---|---|---|---|"]
    lines += [f"| {t} | {v['n']} | {f(v['accuracy'])} | {f(v['faithfulness'])} |" for t, v in m["by_type"].items()]
    lines += ["", f"Judge vs deterministic metrics: numbers agreement {f(c['numbers_agreement'])} "
                  f"({len(c['numbers_disagreements'])} disagreements), refusal agreement "
                  f"{f(c['refusal_agreement'])} ({len(c['refusal_disagreements'])} disagreements)."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dataset", action="store_true", help="review eval/eval_set.jsonl itself")
    ap.add_argument("--results", help="results JSON to grade (default: newest eval/results/results_*.json)")
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "eval_set.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    setup_logging("WARNING")
    settings = get_settings()
    llm = make_llm(settings.llm, settings.llm.judge_backend)
    llm.check()
    workers = settings.llm.judge_workers
    store = ChunkStore(settings.paths.index_dir / "chunks.jsonl")
    eval_path = Path(args.eval_set)
    items = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.review_dataset:
        todo = items[: args.limit] if args.limit else items
        print(f"Reviewing {len(todo)} eval items with {llm.name}/{llm.model} ...", flush=True)
        report = review_dataset(llm, todo, store, workers)
        eval_path.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
        (out_dir / f"dataset_review_{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
        print(json.dumps(report["counts"]))
        for r in report["items"]:
            if r.get("verdict") != "ok":
                print(f"- {r['id']} {r.get('verdict')}: {'; '.join(r.get('issues', []))} -> {r.get('suggested_fix', '')}")
        return

    path = Path(args.results or sorted(glob.glob(str(out_dir / "results_*.json")))[-1])
    results = json.loads(path.read_text(encoding="utf-8"))
    results["_path"] = str(path.relative_to(ROOT))
    if "generation" not in results:
        sys.exit(f"{path} has no generation results (run eval/run_eval.py without --no-generation).")
    if args.limit:
        results["generation"]["rows"] = results["generation"]["rows"][: args.limit]
    item_map = {it["id"]: it for it in items}
    print(f"Grading {len(results['generation']['rows'])} answers from {path.name} with {llm.name}/{llm.model} ...",
          flush=True)
    report = judge_answers(llm, results, item_map, store, workers)
    (out_dir / f"judge_{stamp}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_md(report)
    (out_dir / "judge_latest.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
