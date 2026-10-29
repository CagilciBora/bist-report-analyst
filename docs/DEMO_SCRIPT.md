# Demo Script (under 5 minutes)

**Setup before the interview** (models load in ~20 s, keep both running):
```bash
ollama serve &                                    # local LLM
uv run uvicorn kaprag.api.main:app --port 8000 &  # API
uv run streamlit run ui/app.py                    # UI → http://localhost:8501
```
Start in mode **hybrid_rerank** with "Filtreleri elle belirle" off.

---

### 1. Factual number (≈45 s)
**Ask:** `THY'nin 2025 yılı hasılatı ne kadar?`

Point out:
- The caption under the answer: the parser found `THYAO` from the alias **"THY"** and the year `2025` by itself, with no dropdowns needed.
- The number is underlined green = **verified** against the retrieved context. Hover to see the status.
- The citation (in our test run `[THYAO 2025 faaliyet_raporu s.39]`, "24,1 milyar ABD doları") → open *Kaynaklar* and show the per-retriever ranks: this chunk came in from **dense** (#5) while BM25 missed it; the reranker scored it 0.93.
- Honest point if asked: the income statement (955.472 milyon TL, finansal_rapor s.11) is the TL figure. A 3B model picks one source; this is why the answer shows sources instead of hiding them.

### 2. Table lookup (≈45 s)
**Ask:** `THY'nin 31 Aralık 2025 itibarıyla toplam varlıkları ne kadar?`

Point out:
- The balance sheet is **borderless** in the PDF; generic table extractors collapse it. Our layout extractor rebuilt rows and columns from word coordinates (`src/kaprag/ingestion/table_extractor.py`).
- In *Kaynaklar*, compare the **BM25 rank** and **dense rank** of the table chunk. Exact line-item words favour BM25, and RRF combines both.
- **Live switch:** set the mode to `dense`, ask again, and show the ranking change. Then go back to `hybrid_rerank`.

### 3. Risk summary (≈60 s)
**Ask:** `BİM'in 2025 raporunda iklimle ilgili hangi riskler anlatılıyor? Kısaca özetle.`

Point out:
- **Section titles** come from font-size/bold detection, not hard-coded per company.
- Several cited pages from the same section; every bullet has its own citation.
- Disclaimer line at the end, always present (enforced in code, not left to the model).

### 4. Two-company comparison (≈75 s)
**Ask:** `ASELSAN ile BİM'in 2025 hasılatını karşılaştır.`

Point out:
- `karşılaştırma=True` in the caption: the question was split into **one sub-query per (company, year)** with an **equal budget**, so BİM's 700 chunks cannot crowd out ASELSAN. The `group` label in *Kaynaklar* shows which sub-query found each chunk.
- The answer ends with the Markdown comparison table (company, year, metric, value, source).
- The units differ (ASELSAN reports "Milyar TL" in narrative text, BİM a table in "milyon TL"). The verifier still matches them because it compares absolute values.

### 5. Unanswerable (≈30 s)
**Ask:** `Pegasus'un 2025 yılı yolcu sayısı kaç?`

Point out:
- `PGSUS` is recognised from `companies.csv`, but no Pegasus document is indexed. The filter returns **zero chunks**, so the system answers "Bu bilgi sağlanan belgelerde yer almıyor" **without calling the LLM**. Nothing can be hallucinated.
- Contrast: remove the company name (`2025 yolcu sayısı kaç?`). The caption warns *"tüm belgelerde arandı"* (searched all documents).

---

**Close (15 s):** open `eval/results/latest.md` or the README table: "Here are the measured numbers per retrieval mode. Hybrid and reranking are not assumptions, I measured them."
