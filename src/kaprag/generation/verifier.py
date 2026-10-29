"""NumberVerifier: is every number in the answer backed by the retrieved context?

Why: LLMs are fluent but sloppy with digits. They round, swap columns
(2024 vs 2025), mix up "bin TL" and "milyon TL", or invent a plausible
figure. We parse every number in the answer and look for it in the context.

Matching allows for formatting and unit scale:
  answer "1,2 milyar TL"  == context "1.200.000 bin TL"   (same absolute value)
  answer "955,5 milyar TL" == context "955.472" in a "milyon TL" table
      (tolerance = half a unit of the answer's last digit: 0,05 milyar)
A context number without a unit takes the unit the chunk itself declares
(e.g. a caption "Milyon Türk Lirası" or "(milyon TL)"); only if the chunk
declares no single unit are all scales (TL, bin, milyon, milyar) tried.
This is what catches "955.472 bin TL" when the table says milyon TL.

Shown calculations "(a − b) / b = %c" are re-computed: c is accepted if the
arithmetic is right; a and b are still checked against the context.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import asdict, dataclass

from kaprag.utils.numbers import ParsedNumber, _to_float, extract_numbers

_SCALES = (1.0, 1e3, 1e6, 1e9)
_UNIT_RE = re.compile(r"\b(bin|milyon|milyar)\s+(tl|türk lirası|try|abd doları|usd|avro|eur)",
                      re.IGNORECASE)
_SCALE_OF = {"bin": 1e3, "milyon": 1e6, "milyar": 1e9}
# Numbered list markers ("1. ", "2) ") at the start of a line are not facts.
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d{1,2}[.)]\s")
# A calculation: something with operators, then "=", then a result.
_CALC_RE = re.compile(r"(?P<expr>[\d.,%()\s+\-−–*/×x÷]{3,}?)\s*=\s*(?P<result>[-−%]?\s?\d[\d.,]*\s?%?)")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


@dataclass
class NumberCheck:
    raw: str
    value: float
    status: str          # verified | calculated | unverified
    start: int
    end: int
    source_chunk: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_eval(expr: str) -> float | None:
    """Evaluate + - * / and parentheses only (never Python's eval on model output)."""
    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError("unsupported")

    try:
        return ev(ast.parse(expr.strip(), mode="eval"))
    except Exception:
        return None


class NumberVerifier:
    def __init__(self, rel_tol: float = 1e-9) -> None:
        self.rel_tol = rel_tol

    # ------------------------------------------------------------ context
    @staticmethod
    def _context_values(chunks: dict[str, str]) -> list[tuple[float, bool, str]]:
        """All candidate absolute values found in the context: (value, is_percent, chunk_id)."""
        out = []
        for cid, text in chunks.items():
            declared = {_SCALE_OF[m.group(1).lower()] for m in _UNIT_RE.finditer(text)}
            # one declared unit -> unitless numbers are in that unit (or plain counts)
            scales = (1.0, *declared) if len(declared) == 1 else _SCALES
            for n in extract_numbers(text, skip_years=False):
                v = abs(n.value)
                if n.scale != 1.0:
                    out.append((v * n.scale, n.is_percent, cid))
                else:
                    out.extend((v * s, n.is_percent, cid) for s in scales)
        return out

    @staticmethod
    def _tolerance(n: ParsedNumber) -> float:
        # Half a unit of the last written digit, in absolute terms.
        return 0.5 * 10 ** (-n.decimals) * n.scale

    def _find(self, n: ParsedNumber, ctx: list[tuple[float, bool, str]]) -> str | None:
        target, tol = abs(n.absolute), self._tolerance(n)
        best = None
        for value, _is_pct, cid in ctx:
            if abs(value - target) <= tol + self.rel_tol * target:
                # prefer an exact match over a rounded one
                if abs(value - target) <= self.rel_tol * max(1.0, target):
                    return cid
                best = best or cid
        return best

    # -------------------------------------------------------- calculation
    def _calculations(self, answer: str) -> list[tuple[int, int, bool]]:
        """Return (result_start, result_end, is_correct) for every shown calculation."""
        out = []
        for m in _CALC_RE.finditer(answer):
            expr = m.group("expr")
            if not re.search(r"[+\-−–*/×x÷]", expr.strip().lstrip("(-−")):
                continue  # just "x = y", no arithmetic
            py = expr.replace("×", "*").replace("x", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
            py = re.sub(r"%\s?", "", py)
            py = re.sub(r"\d[\d.,]*", lambda t: repr(_to_float(t.group(0).rstrip(".,"))[0]), py)
            value = _safe_eval(py)
            result = m.group("result")
            res_num = _to_float(re.sub(r"[^\d.,]", "", result).rstrip(".,"))[0]
            if "-" in result or "−" in result:
                res_num = -res_num
            ok = False
            if value is not None:
                for candidate in (value, value * 100):  # ratio shown as a percentage
                    decimals = len(re.sub(r".*[.,]", "", result)) if re.search(r"[.,]\d", result) else 0
                    if abs(candidate - res_num) <= 0.5 * 10 ** (-decimals) + 1e-9:
                        ok = True
            out.append((m.start("result"), m.end("result"), ok))
        return out

    # ------------------------------------------------------------- public
    def verify(self, answer: str, context_chunks: dict[str, str]) -> list[NumberCheck]:
        """Check every number in `answer` against {chunk_id: text} context."""
        masked = _LIST_MARKER_RE.sub(lambda m: " " * len(m.group(0)), answer)
        ctx = self._context_values(context_chunks)
        calcs = self._calculations(masked)
        checks: list[NumberCheck] = []
        for n in extract_numbers(masked):
            calc = next((c for c in calcs if c[0] <= n.start < c[1] or n.start <= c[0] < n.end), None)
            source = self._find(n, ctx)
            if source:
                checks.append(NumberCheck(n.raw, n.value, "verified", n.start, n.end, source))
            elif calc and calc[2]:
                checks.append(NumberCheck(n.raw, n.value, "calculated", n.start, n.end,
                                          note="result of a shown calculation; arithmetic checks out"))
            else:
                note = "calculation does not add up" if calc else "not found in retrieved context"
                checks.append(NumberCheck(n.raw, n.value, "unverified", n.start, n.end, note=note))
        return checks
