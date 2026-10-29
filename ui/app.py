"""Streamlit frontend. Talks to the FastAPI backend over HTTP.

Run:  uv run streamlit run ui/app.py      (API must be running on KAP_API_URL)
The UI holds no RAG logic; it only renders what POST /ask returns, so the
same answers can be reproduced with curl or the CLI.
"""

from __future__ import annotations

import html
import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("KAP_API_URL", "http://127.0.0.1:8000")
MODES = ["hybrid_rerank", "hybrid", "dense", "bm25"]
DOC_TYPES = ["faaliyet_raporu", "finansal_rapor", "ozel_durum", "other"]

st.set_page_config(page_title="BIST Rapor Analisti", page_icon="📊", layout="wide")


@st.cache_data(ttl=60)
def get(path: str):
    return httpx.get(f"{API}{path}", timeout=30).json()


def highlight_numbers(answer: str, numbers: list[dict]) -> str:
    """Wrap unverified numbers in a red <mark>, verified ones in a subtle green underline."""
    out, pos = [], 0
    for n in sorted(numbers, key=lambda n: n["start"]):
        if n["start"] < pos:
            continue
        out.append(html.escape(answer[pos:n["start"]], quote=False))
        text = html.escape(answer[n["start"]:n["end"]], quote=False)
        if n["status"] == "unverified":
            out.append(f'<mark style="background:#ffb3b3" title="{html.escape(n["note"])}">{text} ⚠</mark>')
        else:
            out.append(f'<span style="border-bottom:2px solid #5cb85c" title="{n["status"]}">{text}</span>')
        pos = n["end"]
    out.append(html.escape(answer[pos:], quote=False))
    return "".join(out)


def fmt(rank, score) -> str:
    return "–" if rank is None else f"#{rank} ({score:.3f})"


def render_answer(resp: dict) -> None:
    st.markdown(highlight_numbers(resp["answer"], resp["numbers"]), unsafe_allow_html=True)
    if resp.get("error"):
        st.error(resp["error"])
    nums = resp["numbers"]
    unver = resp["unverified_numbers"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Sayılar (numbers)", len(nums))
    c2.metric("Doğrulanmamış (unverified)", len(unver))
    c3.metric("Mod", resp["retrieval_mode"])
    if unver:
        st.warning("Kaynakta bulunamayan sayılar: " + ", ".join(unver))
    pq = resp["parsed_query"]
    st.caption(f"Filtreler: şirket={pq['tickers'] or 'hepsi'} · yıl={pq['years'] or 'hepsi'} · "
               f"karşılaştırma={pq['is_comparison']} · finansal={pq['is_financial']}"
               + (" · ⚠ tüm belgelerde arandı" if resp["searched_all_documents"] else ""))
    for note in pq["notes"]:
        st.caption(f"ℹ {note}")

    with st.expander(f"Kaynaklar ({len(resp['chunks'])} parça)"):
        for i, c in enumerate(resp["chunks"], 1):
            st.markdown(f"**{i}. {c['citation']}** {c['group']} · {c['company_name']} · "
                        f"s.{c['page_start']}-{c['page_end']} · *{c['section_title'][:80]}* · `{c['chunk_type']}`")
            st.caption(f"dense {fmt(c['dense_rank'], c['dense_score'])} · bm25 {fmt(c['bm25_rank'], c['bm25_score'])}"
                       f" · fused {fmt(c['fused_rank'], c['fused_score'])} · rerank "
                       f"{'–' if c['rerank_score'] is None else round(c['rerank_score'], 3)}")
            if c["chunk_type"] == "table":
                st.markdown(c["text"])  # Markdown table renders as a real table
            else:
                st.text(c["text"][:1500])
            st.divider()
    with st.expander("Süreler (timings, ms)"):
        st.bar_chart(pd.Series(resp["timings_ms"]).drop("total", errors="ignore"))
        st.caption(f"Toplam: {resp['timings_ms'].get('total', 0):.0f} ms · LLM: {resp['llm_backend']}/"
                   f"{resp['llm_model']} · prompt v{resp['prompt_version']}")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 BIST Rapor Analisti")
try:
    health = get("/health")
    companies = get("/companies")
    docs = get("/documents")
except Exception as exc:  # API not running
    st.error(f"API'ye bağlanılamadı ({API}). Önce: `uv run uvicorn kaprag.api.main:app`\n\n{exc}")
    st.stop()

st.sidebar.caption(f"{health['indexed_documents']} belge · {health['indexed_chunks']} parça · "
                   f"LLM {health['llm_model']} {'✅' if health['llm_ready'] else '❌'}")
if not health["llm_ready"]:
    st.sidebar.warning(health["llm_message"])

mode = st.sidebar.selectbox("Arama modu (retrieval mode)", MODES, help="Mülakat demosu için modları karşılaştırın")
use_filters = st.sidebar.toggle("Filtreleri elle belirle", value=False,
                                help="Açıkken aşağıdaki filtreler sorudan çıkarılan filtrelerin yerine geçer")
indexed = sorted({d["ticker"] for d in docs})
years = sorted({d["year"] for d in docs}) or [2024, 2025]
options = indexed + [c["ticker"] for c in companies if c["ticker"] not in indexed]
sel_tickers = st.sidebar.multiselect("Şirket", options, disabled=not use_filters)
yr = st.sidebar.slider("Yıl aralığı", min(years) - 1, max(years), (min(years), max(years)), disabled=not use_filters)
sel_types = st.sidebar.multiselect("Belge türü", DOC_TYPES, disabled=not use_filters)
st.sidebar.divider()
st.sidebar.caption("⚠ Yalnızca analiz amaçlıdır, yatırım tavsiyesi değildir.")

tab_chat, tab_docs = st.tabs(["Sohbet", "Belgeler / Yükle"])

# --------------------------------------------------------------------- chat
with tab_chat:
    if "history" not in st.session_state:
        st.session_state.history = []
    for q, resp in st.session_state.history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            render_answer(resp)
    question = st.chat_input("Örn: THY'nin 2025 hasılatı ne kadar?")
    if question:
        with st.chat_message("user"):
            st.write(question)
        payload = {"question": question, "retrieval_mode": mode}
        if use_filters:
            payload["filters"] = {"tickers": sel_tickers or None, "year_from": yr[0], "year_to": yr[1],
                                  "doc_types": sel_types or None}
        with st.chat_message("assistant"), st.spinner("Aranıyor ve yanıt üretiliyor..."):
            r = httpx.post(f"{API}/ask", json=payload, timeout=900)
            if r.status_code != 200:
                st.error(r.text)
            else:
                resp = r.json()
                st.session_state.history.append((question, resp))
                render_answer(resp)

# --------------------------------------------------------------- documents
with tab_docs:
    st.subheader("İndekslenmiş belgeler")
    if docs:
        st.dataframe(pd.DataFrame(docs)[["ticker", "year", "doc_type", "period", "n_chunks", "n_pages",
                                         "source_file", "indexed_at"]], hide_index=True)
    st.subheader("Yeni PDF yükle")
    with st.form("upload"):
        f = st.file_uploader("PDF", type=["pdf"])
        c1, c2, c3 = st.columns(3)
        t = c1.text_input("Ticker", "ASELS")
        y = c2.number_input("Yıl", 2000, 2100, 2025)
        dt = c3.selectbox("Belge türü", DOC_TYPES)
        if st.form_submit_button("Yükle ve indeksle") and f:
            with st.spinner("Ayrıştırılıyor ve gömülüyor (birkaç dakika sürebilir)..."):
                r = httpx.post(f"{API}/ingest", files={"file": (f.name, f.getvalue(), "application/pdf")},
                               data={"ticker": t, "year": int(y), "doc_type": dt}, timeout=3600)
            st.write(r.json())
            get.clear()
