"""PromptBuilder: fills the versioned template prompts/answer_tr.md.

The prompt lives in a file (not in code) so it can be versioned, diffed and
reviewed like any other artifact; the version string travels with every
answer, which makes evaluation results reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from kaprag.retrieval.hybrid import RetrievedChunk

DISCLAIMER_TR = "Not: Bu yanıt yalnızca bilgilendirme amaçlı bir analizdir, yatırım tavsiyesi değildir."
DISCLAIMER_EN = "Note: This answer is analysis only and is not investment advice."

# No re.IGNORECASE here: case-folding makes "İ" match a plain "i", so every
# English sentence would look Turkish (the dotted-I problem again).
_TR_LETTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_TR_WORDS = re.compile(r"\b(ve|nedir|ne kadar|nasil|hangi|kac|mi|mu)\b")


def detect_language(text: str) -> str:
    """Tiny heuristic: Turkish letters or common Turkish words -> 'tr', else 'en'."""
    return "tr" if _TR_LETTERS.search(text) or _TR_WORDS.search(text.lower()) else "en"


def disclaimer_for(question: str) -> str:
    return DISCLAIMER_TR if detect_language(question) == "tr" else DISCLAIMER_EN


@dataclass
class BuiltPrompt:
    system: str
    user: str
    version: str


class PromptBuilder:
    def __init__(self, template_path: Path) -> None:
        raw = template_path.read_text(encoding="utf-8")
        m = re.search(r"prompt_version:\s*([\w.]+)", raw)
        self.version = m.group(1) if m else "unversioned"
        system_part, user_part = raw.split("## USER", 1)
        self.system_template = system_part.split("## SYSTEM", 1)[1].strip()
        self.user_template = user_part.strip()

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Number each chunk and prefix it with the exact citation tag the model must use."""
        blocks = []
        for i, rc in enumerate(chunks, 1):
            c = rc.chunk
            kind = "tablo" if c.chunk_type == "table" else "metin"
            pages = f"s.{c.page_start}" + (f"-{c.page_end}" if c.page_end != c.page_start else "")
            head = f"[{i}] Kaynak: {c.citation} | {c.company_name} | {pages} | {kind}"
            if c.section_title:
                head += f" | Bölüm: {c.section_title}"
            blocks.append(f"{head}\n{c.text}")
        return "\n\n---\n\n".join(blocks)

    def build(self, question: str, chunks: list[RetrievedChunk], is_comparison: bool) -> BuiltPrompt:
        disclaimer = disclaimer_for(question)
        note = ""
        if is_comparison:
            groups = sorted({rc.group for rc in chunks if rc.group})
            note = ("Bu bir karşılaştırma sorusudur. Bağlam şu gruplar için ayrı ayrı getirildi: "
                    f"{', '.join(groups)}. Her grubu ayrı değerlendir ve sonunda tabloyu ekle.\n\n")
        system = self.system_template.replace("{disclaimer}", disclaimer)
        user = (self.user_template.replace("{context}", self.format_context(chunks) or "(boş)")
                .replace("{comparison_note}", note).replace("{question}", question))
        return BuiltPrompt(system, user, self.version)
