"""
RAG Assistant — dark navy UI.
Tabs: Chat | Dashboard
Dashboard: KPI cards, query log, in-app evaluation (Run button + progress + chart).
"""

from __future__ import annotations

import json
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.config import RAGConfig
from src.engine import RAGQueryEngine
from src.ingestor import DocumentIngestor
from src.observability.query_logger import QueryLogger

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background: #07111f !important;
    color: #d0e4f7 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.block-container { padding-top: 0.6rem !important; padding-bottom: 0.5rem !important; max-width: 100% !important; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="collapsedControl"], [data-testid="stSidebarHeader"], [data-testid="stSidebarCollapsedControl"] { display: none !important; visibility: hidden !important; }
[data-testid="stSidebar"] { display: flex !important; visibility: visible !important; background: #0a1929 !important; border-right: 1px solid #132d50 !important; min-width: 18rem !important; width: 18rem !important; transform: none !important; }
[data-testid="stSidebar"] > div:first-child { padding: 0.8rem 0.7rem 1rem !important; }

/* Buttons */
.stButton > button {
    background: #0e2240 !important; color: #90b8e0 !important;
    border: 1px solid #163660 !important; border-radius: 5px !important;
    font-size: 0.78rem !important; padding: 0.25rem 0.5rem !important;
    height: auto !important; line-height: 1.4 !important; transition: all 0.14s !important;
}
.stButton > button:hover { background: #163660 !important; border-color: #2f6db5 !important; color: #c8dff5 !important; }
.stButton > button:focus { box-shadow: none !important; }
.accent-btn .stButton > button { background: #154a9e !important; border-color: #1a5abf !important; color: #fff !important; font-weight: 500 !important; }
.accent-btn .stButton > button:hover { background: #1a5abf !important; }
.run-btn .stButton > button { background: #0f4c1a !important; border-color: #1a7a2e !important; color: #4ade80 !important; font-weight: 600 !important; font-size: 0.82rem !important; }
.run-btn .stButton > button:hover { background: #1a7a2e !important; color: #86efac !important; }
.del-btn .stButton > button { background: transparent !important; border-color: #4a1919 !important; color: #c87070 !important; font-size: 0.72rem !important; padding: 0.15rem 0.35rem !important; }
.del-btn .stButton > button:hover { background: #3d0e0e !important; border-color: #e05252 !important; color: #f87171 !important; }
.ghost-btn .stButton > button { background: transparent !important; border-color: #163660 !important; color: #4d7aaa !important; }
.ghost-btn .stButton > button:hover { background: #0e2240 !important; color: #90b8e0 !important; }

/* Tabs */
[data-testid="stTabs"] > div:first-child { background: transparent !important; border-bottom: 1px solid #132d50 !important; padding: 0 !important; gap: 0 !important; }
button[role="tab"] { background: transparent !important; color: #4d7aaa !important; border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; padding: 0.3rem 0.9rem !important; font-size: 0.82rem !important; margin: 0 !important; }
button[role="tab"][aria-selected="true"] { color: #4da6ff !important; border-bottom-color: #2b7fff !important; font-weight: 500 !important; }
button[role="tab"]:hover { color: #7ec0f5 !important; }

/* Chat */
[data-testid="stChatInput"] textarea { background: #0a1929 !important; color: #d0e4f7 !important; border: 1px solid #163660 !important; border-radius: 8px !important; font-size: 0.88rem !important; }
[data-testid="stChatInputSubmitButton"] button { background: #154a9e !important; color: #fff !important; border: none !important; }
.msg-user { display: flex; justify-content: flex-end; margin: 0.4rem 0; }
.bubble-user { background: #163a6e; color: #d0e4f7; padding: 0.5rem 0.8rem; border-radius: 14px 14px 3px 14px; max-width: 70%; font-size: 0.87rem; line-height: 1.55; word-wrap: break-word; }
.msg-ai { display: flex; flex-direction: column; align-items: flex-start; margin: 0.4rem 0; }
.bubble-ai { background: #0c1e35; color: #d0e4f7; padding: 0.5rem 0.8rem; border-radius: 14px 14px 14px 3px; max-width: 70%; font-size: 0.87rem; line-height: 1.55; border: 1px solid #132d50; word-wrap: break-word; }
.msg-src { font-size: 0.68rem; color: #3a6090; margin-top: 0.22rem; max-width: 70%; cursor: pointer; }
.msg-src summary { list-style: none; user-select: none; color: #3a6090; }
.msg-src summary::-webkit-details-marker { display: none; }
.msg-src[open] summary { color: #4d7aaa; margin-bottom: 0.15rem; }
.src-item { padding: 0.08rem 0.6rem; color: #3a6090; }

/* Metric card */
.mcard { background: #0c1e35; border: 1px solid #132d50; border-radius: 9px; padding: 0.75rem 0.9rem; text-align: center; }
.mcard-val { font-size: 1.5rem; font-weight: 700; color: #2b7fff; line-height: 1.1; }
.mcard-val.green { color: #22c55e !important; }
.mcard-lbl { font-size: 0.65rem; color: #3a6090; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 0.07em; }

/* Section header */
.shdr { font-size: 0.64rem; font-weight: 600; color: #3a6090; text-transform: uppercase; letter-spacing: 0.09em; margin: 0.6rem 0 0.35rem; }

/* Doc row */
.doc-row { background: #0c1e35; border: 1px solid #132d50; border-radius: 6px; padding: 0.3rem 0.5rem; margin-bottom: 0.25rem; }
.doc-name { font-size: 0.74rem; color: #90b8e0; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-meta { font-size: 0.63rem; color: #3a6090; }

hr { border-color: #132d50 !important; margin: 0.5rem 0 !important; }

/* File uploader */
[data-testid="stFileUploader"] { background: #0c1e35 !important; border: 1px dashed #163660 !important; border-radius: 7px !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div, [data-testid="stFileUploaderDropzoneInstructions"] small { color: #3a6090 !important; }

/* Multiselect */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div { background: #0c1e35 !important; border-color: #163660 !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background: #163a6e !important; color: #90b8e0 !important; }
[data-testid="stMultiSelect"] span { color: #90b8e0 !important; font-size: 0.8rem !important; }
[data-testid="stMultiSelect"] label { color: #3a6090 !important; font-size: 0.72rem !important; }

/* Checkbox */
.stCheckbox label p, .stCheckbox label span { color: #90b8e0 !important; font-size: 0.79rem !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid #132d50 !important; border-radius: 7px !important; }

/* Alerts */
[data-testid="stAlert"] > div { border-radius: 6px !important; font-size: 0.82rem !important; }
.stSuccess > div { background: #041f12 !important; color: #4ade80 !important; border: 1px solid #14532d !important; }
.stError > div { background: #1f0404 !important; color: #f87171 !important; border: 1px solid #7f1d1d !important; }
.stInfo > div { background: #050f1f !important; color: #7ec0f5 !important; border: 1px solid #132d50 !important; }
.stWarning > div { background: #1c1500 !important; color: #fbbf24 !important; border: 1px solid #854d0e !important; }

/* Status box */
[data-testid="stStatus"] { background: #0c1e35 !important; border: 1px solid #132d50 !important; border-radius: 8px !important; }
[data-testid="stStatus"] > div { color: #d0e4f7 !important; }

/* Spinner */
.stSpinner > div { color: #2b7fff !important; }
code { background: #0c1e35 !important; color: #7ec0f5 !important; border: 1px solid #132d50 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE  &  LOGGER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_engine():
    try:
        config = RAGConfig.from_yaml("config.yaml")
        engine = RAGQueryEngine(config=config, use_hybrid=True, use_reranking=False)
        qlogger = QueryLogger(db_path=config.sqlite_path)
        return engine, qlogger, None
    except Exception as exc:
        return None, None, str(exc)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    for k, v in [("messages", []), ("scope", []), ("_prev_scope", None)]:
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _doc_chunk_count(engine, filename: str) -> int:
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        return engine.client.count(
            collection_name=engine.config.collection_name,
            count_filter=Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=filename))]),
            exact=False,
        ).count
    except Exception:
        return 0


def _total_chunks(engine) -> int:
    try:
        return engine.client.count(collection_name=engine.config.collection_name).count
    except Exception:
        return 0


def _delete_doc(engine, filename: str) -> None:
    from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
    engine.client.delete(
        collection_name=engine.config.collection_name,
        points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=filename))])),
    )
    pdf = Path(engine.config.data_dir) / filename
    if pdf.exists():
        pdf.unlink()
    engine.refresh()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(engine):
    sb = st.sidebar
    sb.markdown(
        "<div style='font-size:0.95rem;font-weight:700;color:#4da6ff;padding:0 0 0.8rem;letter-spacing:-0.01em'>🔷 RAG Assistant</div>",
        unsafe_allow_html=True,
    )
    sb.markdown('<div class="shdr">Knowledge Base</div>', unsafe_allow_html=True)

    docs: List[str] = engine.get_available_documents() if engine else []

    if not docs:
        sb.markdown("<div style='font-size:0.75rem;color:#3a6090;padding:0.3rem 0 0.6rem'>No documents indexed.</div>", unsafe_allow_html=True)
    else:
        to_delete: Optional[str] = None
        for doc in docs:
            count = _doc_chunk_count(engine, doc)
            short = doc[:26] + ("…" if len(doc) > 26 else "")
            col_name, col_del = sb.columns([5, 1], gap="small")
            col_name.markdown(
                f'<div class="doc-row"><div class="doc-name" title="{doc}">{short}</div><div class="doc-meta">{count:,} chunks</div></div>',
                unsafe_allow_html=True,
            )
            with col_del:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"del_{doc}", help=f"Delete {doc}"):
                    to_delete = doc
                st.markdown("</div>", unsafe_allow_html=True)

        if to_delete:
            with st.spinner(f"Deleting {to_delete}…"):
                try:
                    _delete_doc(engine, to_delete)
                    st.session_state.pop(f"cb_{to_delete}", None)
                    st.rerun()
                except Exception as exc:
                    sb.error(str(exc))

    if docs:
        sb.markdown('<div class="shdr">Query scope</div>', unsafe_allow_html=True)
        selected = []
        for doc in docs:
            short = doc[:28] + ("…" if len(doc) > 28 else "")
            if sb.checkbox(short, value=st.session_state.get(f"cb_{doc}", True), key=f"cb_{doc}"):
                selected.append(doc)
        # None = no filter (all docs); pass list only when subset selected
        scope_filter = None if not selected or len(selected) == len(docs) else selected
        if engine and scope_filter != st.session_state._prev_scope:
            engine.update_document_filter(scope_filter)
            st.session_state._prev_scope = scope_filter

    sb.markdown("---")
    sb.markdown('<div class="shdr">Add Documents</div>', unsafe_allow_html=True)
    uploaded = sb.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded:
        st.markdown('<div class="accent-btn">', unsafe_allow_html=True)
        if sb.button("⬆  Ingest", use_container_width=True):
            with st.spinner("Indexing…"):
                try:
                    config = RAGConfig.from_yaml("config.yaml")
                    Path(config.data_dir).mkdir(exist_ok=True)
                    for f in uploaded:
                        (Path(config.data_dir) / f.name).write_bytes(f.getbuffer())
                    shared = getattr(engine, "client", None) if engine else None
                    ingestor = DocumentIngestor(config=config, existing_client=shared)
                    result = ingestor.ingest(recreate=False)
                    if engine:
                        engine.refresh()
                    sb.success(f"{result['chunks_created']} chunks in {result['duration_seconds']:.1f}s")
                    st.rerun()
                except Exception as exc:
                    sb.error(str(exc))
        st.markdown("</div>", unsafe_allow_html=True)

    sb.markdown("---")
    st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
    if sb.button("🗑  Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def render_chat(engine, qlogger):
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user"><div class="bubble-user">{_esc(msg["content"])}</div></div>', unsafe_allow_html=True)
        else:
            src_html = ""
            if msg.get("sources"):
                seen, items = set(), []
                for s in msg["sources"]:
                    fn = s["file_name"]
                    pg = s.get("page")
                    lbl = f"{fn} · page {pg}" if pg and pg != "N/A" else fn
                    if lbl not in seen:
                        seen.add(lbl)
                        items.append(lbl)
                if items:
                    n = len(items)
                    label = f"{n} source{'s' if n > 1 else ''}"
                    rows = "".join(f'<div class="src-item">📄 {_esc(it)}</div>' for it in items)
                    src_html = f'<details class="msg-src"><summary>{label} ▾</summary>{rows}</details>'
            st.markdown(f'<div class="msg-ai"><div class="bubble-ai">{_esc(msg["content"])}</div>{src_html}</div>', unsafe_allow_html=True)

    if engine is None:
        st.info("Build the vector index first: `python src/ingestor.py --data-dir data --db-path db --recreate`")
        return

    user_input = st.chat_input("Ask about your documents…")
    if user_input and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": q})
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.spinner("Thinking…"):
            try:
                result = engine.query(q, conversation_history=history)
                answer, sources = result["answer"], result.get("sources", [])
                if qlogger:
                    qlogger.log_query(q, result)
            except Exception as exc:
                answer, sources = f"Error: {exc}", []
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_BUILTIN_Q = Path("data/test_questions.json")
_AUTOGEN_Q = Path("data/eval_questions_autogen.json")


def _latest_hit_rate() -> Optional[float]:
    """Return hit_rate_at_k from the most recent experiment run, or None."""
    runs_file = Path("results/experiments.jsonl")
    if not runs_file.exists():
        return None
    last_val = None
    with runs_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    run = json.loads(line)
                    v = run.get("metrics", {}).get("hit_rate_at_k")
                    if v is not None:
                        last_val = v
                except json.JSONDecodeError:
                    pass
    return last_val


def _load_eval_questions() -> Optional[List[Dict]]:
    """Return parsed test cases list, preferring built-in over auto-generated."""
    for path in (_BUILTIN_Q, _AUTOGEN_Q):
        if path.exists():
            try:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                if data:
                    return data
            except Exception:
                pass
    return None


def _generate_questions_with_llm(engine) -> bool:
    """
    Sample nodes from the index, ask the LLM to create one Q/A per node,
    and save results to data/eval_questions_autogen.json.
    Returns True on success.
    """
    nodes = engine._all_nodes
    if not nodes:
        st.warning("No nodes loaded in engine — ingest documents first.")
        return False

    # Sample up to 8 nodes across all documents
    sample = random.sample(nodes, min(8, len(nodes)))
    questions: List[Dict] = []

    progress = st.empty()
    for i, node in enumerate(sample):
        chunk = node.text[:600].strip()
        if len(chunk) < 80:
            continue
        fname = node.metadata.get("file_name", "doc")
        prompt = (
            "Generate one factual question and its answer based on this text.\n"
            "Return ONLY valid JSON (no markdown, no explanation):\n"
            '{"question": "...", "ground_truth": "...", "relevant_keywords": ["word1", "word2", "word3"]}\n\n'
            f"Text:\n{chunk}"
        )
        progress.markdown(f"<div style='font-size:0.78rem;color:#4d7aaa'>Generating question {i+1}/{len(sample)}…</div>", unsafe_allow_html=True)
        try:
            resp = engine.llm.complete(prompt)
            raw = resp.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```$", "", raw.strip()).strip()
            obj = json.loads(raw)
            if "question" in obj and "ground_truth" in obj:
                questions.append({
                    "id": f"auto_{i:03d}",
                    "question": obj["question"],
                    "ground_truth": obj["ground_truth"],
                    "relevant_keywords": obj.get("relevant_keywords", []),
                    "category": "auto",
                    "source_document": fname,
                })
        except Exception:
            pass  # Skip nodes where LLM output can't be parsed

    progress.empty()

    if not questions:
        st.error("Could not generate questions — is LM Studio running with a model loaded?")
        return False

    _AUTOGEN_Q.parent.mkdir(exist_ok=True)
    with _AUTOGEN_Q.open("w", encoding="utf-8") as fh:
        json.dump(questions, fh, indent=2, ensure_ascii=False)
    st.success(f"Generated {len(questions)} evaluation questions → {_AUTOGEN_Q}")
    return True


def _run_eval_in_ui(engine, k: int = 5, with_ragas: bool = False) -> None:
    """
    Run the full evaluation pipeline with real-time progress shown via st.status().
    Saves results to results/experiments.jsonl on completion.
    """
    from src.evaluation.benchmark import evaluate_retrieval, load_test_cases, TestCase
    from src.evaluation.ragas_eval import evaluate_dataset
    from src.experiment_tracker import ExperimentTracker
    from llama_index.core.retrievers import VectorIndexRetriever

    with st.status("Running evaluation…", expanded=True) as status:

        # ── Stage 1: Load questions ────────────────────────────────────────
        st.write("📋 Loading evaluation questions…")
        raw_cases = _load_eval_questions()
        if not raw_cases:
            status.update(label="❌ No evaluation questions found", state="error")
            st.write("Generate questions first using the button below.")
            return

        test_cases = [TestCase(**tc) for tc in raw_cases]
        q_source = "built-in" if _BUILTIN_Q.exists() else "auto-generated"
        st.write(f"✅ {len(test_cases)} questions loaded ({q_source})")

        # ── Stage 2: Retrieve chunks ───────────────────────────────────────
        st.write(f"🔍 Retrieving top-{k} chunks for each question…")

        def retriever_fn(question: str) -> List[str]:
            try:
                r = VectorIndexRetriever(index=engine.index, similarity_top_k=k)
                return [n.node.text for n in r.retrieve(question)]
            except Exception:
                return []

        # ── Stage 3: Compute retrieval metrics ────────────────────────────
        retrieval_metrics = evaluate_retrieval(test_cases, retriever_fn, k=k)
        hr  = retrieval_metrics.get("hit_rate_at_k", 0.0)
        mrr = retrieval_metrics.get("mrr", 0.0)
        ndcg = retrieval_metrics.get(f"ndcg_at_{k}", 0.0)
        st.write(f"✅ **Hit Rate@{k}:** {hr:.3f}  ·  **MRR:** {mrr:.3f}  ·  **NDCG@{k}:** {ndcg:.3f}")

        # ── Stage 4+5: RAGAS (optional) ───────────────────────────────────
        ragas_metrics: Dict[str, Any] = {}
        if with_ragas:
            st.write(f"🤖 Generating answers ({len(test_cases)} LLM calls)…")
            samples: List[Dict] = []
            for i, tc in enumerate(test_cases):
                st.write(f"  [{i+1}/{len(test_cases)}] {tc.question[:60]}…")
                try:
                    result = engine.query(tc.question)
                    if "error" not in result.get("metadata", {}):
                        samples.append({
                            "question": tc.question,
                            "answer": result["answer"],
                            "contexts": [s["text_preview"] for s in result["sources"]],
                            "ground_truth": tc.ground_truth,
                        })
                except Exception as exc:
                    st.write(f"  ⚠ Skipped: {exc}")

            if samples:
                st.write("📐 Scoring answer quality…")
                ragas_metrics = evaluate_dataset(
                    samples, embed_fn=engine.embed_model.get_text_embedding
                )
                faith = ragas_metrics.get("faithfulness", 0.0)
                rel   = ragas_metrics.get("answer_relevancy", 0.0)
                prec  = ragas_metrics.get("context_precision", 0.0)
                rec   = ragas_metrics.get("context_recall", 0.0)
                st.write(f"✅ **Faithfulness:** {faith:.3f}  ·  **Relevancy:** {rel:.3f}  ·  **Ctx Prec:** {prec:.3f}  ·  **Ctx Rec:** {rec:.3f}")
            else:
                st.write("⚠ No answers were generated — RAGAS skipped.")

        # ── Stage 6: Save ─────────────────────────────────────────────────
        st.write("💾 Saving results…")
        cfg = engine.config
        exp_config = {
            "chunking_strategy": cfg.chunking_strategy,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "vector_top_k": cfg.vector_top_k,
            "bm25_top_k": cfg.bm25_top_k,
            "k": k,
            "with_ragas": with_ragas,
            "source": "ui",
        }
        all_metrics: Dict[str, Any] = {**retrieval_metrics}
        if ragas_metrics and "error" not in ragas_metrics:
            for key in ("answer_relevancy", "faithfulness", "context_precision", "context_recall", "n_samples"):
                if key in ragas_metrics:
                    all_metrics[key] = ragas_metrics[key]

        tracker = ExperimentTracker(results_dir=cfg.results_dir)
        run_id = tracker.log_run(config=exp_config, metrics=all_metrics, tags={"source": "ui"})

        status.update(label=f"✅ Evaluation complete — run {run_id}", state="complete")


def _load_all_runs() -> List[Dict]:
    """Load all experiment runs from experiments.jsonl, newest first."""
    runs_file = Path("results/experiments.jsonl")
    if not runs_file.exists():
        return []
    runs = []
    with runs_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(runs))


def _render_eval_table(runs: List[Dict]) -> None:
    """Show a styled results table. Best row highlighted in green."""
    import pandas as pd

    rows = []
    for run in runs:
        cfg = run.get("config", {})
        met = run.get("metrics", {})
        k   = met.get("k", cfg.get("k", 5))
        rows.append({
            "Run":      run["run_id"][-8:],
            "Tag":      run.get("tags", {}).get("tag", ""),
            "Strategy": cfg.get("chunking_strategy", "?"),
            "Chunk":    cfg.get("chunk_size", "?"),
            f"Hit@{k}": met.get("hit_rate_at_k"),
            "MRR":      met.get("mrr"),
            f"NDCG@{k}": met.get(f"ndcg_at_{k}"),
            "Faithful": met.get("faithfulness"),
            "Ctx Rec":  met.get("context_recall"),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("Run", "Tag", "Strategy", "Chunk")]

    # Find best run (highest hit_rate or first numeric column)
    hr_col = next((c for c in df.columns if c.startswith("Hit@")), None)

    def _style(row):
        is_best = hr_col and row[hr_col] is not None and row[hr_col] == df[hr_col].max()
        bg = "#0a2e1a" if is_best else ""
        return [f"background-color: {bg}"] * len(row)

    styled = (
        df.style
        .format({c: "{:.4f}" for c in num_cols if c in df.columns}, na_rep="—")
        .apply(_style, axis=1)
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_eval_chart(runs: List[Dict]) -> None:
    """Bar chart comparing Hit Rate and MRR across runs. Best run in green."""
    if len(runs) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    display = runs[:10]  # cap at 10 runs
    display.reverse()    # chronological order left → right

    labels, hit_rates, mrrs = [], [], []
    for run in display:
        tag = run.get("tags", {}).get("tag", "")
        labels.append(tag or run["run_id"][-8:])
        met = run.get("metrics", {})
        hit_rates.append(float(met.get("hit_rate_at_k") or 0))
        mrrs.append(float(met.get("mrr") or 0))

    best_idx = hit_rates.index(max(hit_rates)) if hit_rates else -1
    hr_colors  = ["#22c55e" if i == best_idx else "#2b7fff"  for i in range(len(labels))]
    mrr_colors = ["#16a34a" if i == best_idx else "#1a4f9e"  for i in range(len(labels))]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5.5, len(labels) * 1.1), 3.2))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0c1e35")

    b1 = ax.bar(x - width / 2, hit_rates, width, color=hr_colors,  label="Hit Rate@k", zorder=3)
    b2 = ax.bar(x + width / 2, mrrs,      width, color=mrr_colors, label="MRR",         zorder=3)

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7, color="#90b8e0")
    ax.tick_params(axis="y", colors="#90b8e0", labelsize=7)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    for spine in ax.spines.values():
        spine.set_color("#132d50")
    ax.grid(axis="y", color="#132d50", linewidth=0.5, zorder=0)

    for bar in (*b1, *b2):
        h = bar.get_height()
        if h > 0.02:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6, color="#d0e4f7")

    ax.legend(fontsize=7.5, facecolor="#0c1e35", edgecolor="#132d50",
              labelcolor="#90b8e0", loc="upper left")
    ax.set_ylabel("Score", fontsize=7.5, color="#90b8e0")
    ax.set_title("Hit Rate & MRR across runs  (🟢 = best)", fontsize=8,
                 color="#90b8e0", pad=6)

    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD v2 — GitHub dark, 4-tab layout
# ══════════════════════════════════════════════════════════════════════════════

_MONO = "font-family:'IBM Plex Mono',monospace"
_DC = {
    "card": "#161b22", "border": "#21262d", "txt": "#c9d1d9", "muted": "#8b949e",
    "head": "#f0f6fc", "green": "#3fb950", "orange": "#f0883e", "red": "#f85149",
    "blue": "#58a6ff", "purple": "#a371f7",
}


def _dbadge(label: str, color: str) -> str:
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
            f'border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;{_MONO}">{label}</span>')


def _dbar(value: float, color: str) -> str:
    pct = min(100.0, max(0.0, value * 100))
    return (f'<div style="background:#1a2035;border-radius:4px;height:6px;width:100%;overflow:hidden">'
            f'<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px"></div></div>')


def _dcolor(v: Optional[float]) -> str:
    if v is None:
        return _DC["muted"]
    return _DC["green"] if v >= 0.8 else (_DC["orange"] if v >= 0.5 else _DC["red"])


def _dfmt(v: Optional[float]) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _run_label(run: Dict, emb_model: str = "") -> str:
    cfg = run.get("config", {})
    chunk = cfg.get("chunk_size", "?")
    top_k = cfg.get("vector_top_k", cfg.get("k", "?"))
    model_short = emb_model.split("/")[-1] if emb_model else "model"
    return f"{model_short} · chunk={chunk} · top_k={top_k}"


def _run_date(run: Dict) -> str:
    ts = run.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return ts[:10]


def _dcard(content: str, border_color: str = "") -> str:
    bc = border_color if border_color else _DC["border"]
    return (f'<div style="background:{_DC["card"]};border:1px solid {bc};border-radius:8px;'
            f'padding:20px;margin-bottom:16px;{_MONO}">{content}</div>')


def _dsec(label: str, color: str = "") -> str:
    c = color if color else _DC["muted"]
    return (f'<div style="font-size:11px;font-weight:600;color:{c};letter-spacing:0.1em;'
            f'text-transform:uppercase;margin-bottom:12px">{label}</div>')


def _clear_dashboard_data(qlogger) -> None:
    if qlogger:
        try:
            with qlogger._connect() as conn:
                conn.execute("DELETE FROM queries")
        except Exception:
            pass
    runs_file = Path("results/experiments.jsonl")
    if runs_file.exists():
        runs_file.unlink()
    for f in Path("results").glob("run_*.json"):
        try:
            f.unlink()
        except Exception:
            pass


def render_dashboard(engine, qlogger):
    # ── Collect live data ─────────────────────────────────────────────────────
    emb_model   = engine.config.embedding_model if engine else ""
    model_short = emb_model.split("/")[-1] if emb_model else "unknown"
    use_hybrid  = getattr(engine, "use_hybrid", False) if engine else False
    device      = engine.config.embedding_device.upper() if engine else "CPU"
    total_docs  = len(engine.get_available_documents()) if engine else 0
    total_chunks = _total_chunks(engine) if engine else 0

    summary: Dict = {"total_queries": 0, "avg_latency_ms": 0.0,
                     "min_latency_ms": 0.0, "max_latency_ms": 0.0, "top_10_questions": []}
    if qlogger:
        try:
            summary = qlogger.get_summary()
        except Exception:
            pass

    recent_qs: List[Dict] = []
    if qlogger:
        try:
            recent_qs = qlogger.get_recent_queries(limit=20)
        except Exception:
            pass

    all_runs   = _load_all_runs()
    test_cases = _load_eval_questions() or []
    test_by_id = {tc["id"]: tc for tc in test_cases}

    # ── Header bar ────────────────────────────────────────────────────────────
    retrieval_label = "Hybrid" if use_hybrid else "Vector"
    col_hdr, col_btn = st.columns([9, 1])
    with col_hdr:
        st.markdown(
            f'<div style="background:{_DC["card"]};border:1px solid {_DC["border"]};border-radius:8px;'
            f'padding:12px 20px;display:flex;align-items:center;justify-content:space-between;'
            f'margin-bottom:0;{_MONO}">'
            f'<div style="display:flex;align-items:center;gap:12px">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:{_DC["green"]}"></div>'
            f'<span style="font-size:14px;font-weight:600;color:{_DC["head"]}">RAG Assistant</span>'
            f'<span style="color:{_DC["muted"]};font-size:12px">/ dashboard</span></div>'
            f'<div style="display:flex;gap:8px">'
            f'{_dbadge(model_short, _DC["blue"])}'
            f'{_dbadge(retrieval_label, _DC["purple"])}'
            f'{_dbadge(device, _DC["green"])}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
    with col_btn:
        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="del-btn">', unsafe_allow_html=True)
        clear_clicked = st.button("🗑 Clear All", key="btn_clear_all", use_container_width=True,
                                   help="Clear all query logs and evaluation runs")
        st.markdown("</div>", unsafe_allow_html=True)

    if clear_clicked:
        st.session_state.dash_confirm_clear = True

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Confirmation dialog ───────────────────────────────────────────────────
    if st.session_state.get("dash_confirm_clear", False):
        st.warning(
            "**Clear all dashboard data?**  This includes query history, evaluation runs, and all "
            "metrics. System config will not be affected."
        )
        conf_col, canc_col, _ = st.columns([1, 1, 6])
        with conf_col:
            if st.button("✓ Confirm", key="btn_clear_confirm", type="primary",
                         use_container_width=True):
                _clear_dashboard_data(qlogger)
                st.session_state.dash_confirm_clear = False
                st.session_state.pop("dash_sel_run", None)
                st.toast("Dashboard cleared", icon="✓")
                st.rerun()
        with canc_col:
            if st.button("✕ Cancel", key="btn_clear_cancel", use_container_width=True):
                st.session_state.dash_confirm_clear = False
                st.rerun()

    # ── 4 sub-tabs ────────────────────────────────────────────────────────────
    tab_ov, tab_ev, tab_qr, tab_cmp = st.tabs(["overview", "evaluation", "queries", "compare"])

    # ═══════════════════════════ OVERVIEW ════════════════════════════════════
    with tab_ov:
        latest_run = all_runs[0] if all_runs else None
        best_hr    = max((r.get("metrics", {}).get("hit_rate_at_k", 0) for r in all_runs), default=None) if all_runs else None
        avg_lat    = summary.get("avg_latency_ms", 0.0)
        total_q    = summary.get("total_queries", 0)

        # KPI cards
        kpis = [
            ("Total Queries",  str(total_q),             _DC["txt"],    None),
            ("Avg Latency",    f"{avg_lat:.0f}ms",        _DC["orange"] if avg_lat > 5000 else _DC["txt"],
             "⚠ slow" if avg_lat > 5000 else None),
            ("Documents",      str(total_docs),           _DC["txt"],    None),
            ("Chunks Indexed", str(total_chunks),         _DC["txt"],    None),
            ("Best Hit Rate",  _dfmt(best_hr),            _dcolor(best_hr), None),
        ]
        kpi_cols = st.columns(5, gap="small")
        for col, (lbl, val, clr, note) in zip(kpi_cols, kpis):
            note_html = f'<div style="font-size:11px;color:{_DC["orange"]};margin-top:4px">{note}</div>' if note else ""
            col.markdown(
                f'<div style="background:{_DC["card"]};border:1px solid {_DC["border"]};'
                f'border-radius:8px;padding:16px 20px;{_MONO}">'
                f'<div style="font-size:28px;font-weight:600;color:{clr}">{val}</div>'
                f'<div style="font-size:11px;color:{_DC["muted"]};margin-top:4px;'
                f'text-transform:uppercase;letter-spacing:0.05em">{lbl}</div>'
                f'{note_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # System Health bars
        if latest_run:
            met = latest_run.get("metrics", {})
            health = [("Hit Rate@K", met.get("hit_rate_at_k")), ("MRR", met.get("mrr")),
                      ("Faithfulness", met.get("faithfulness")), ("Context Recall", met.get("context_recall"))]
            visible = [(l, v) for l, v in health if v is not None]
            if visible:
                def _hcell(label, val):
                    color   = _dcolor(val)
                    verdict = "Good" if val >= 0.8 else ("Needs work" if val >= 0.5 else "Poor")
                    return (f'<div style="display:flex;flex-direction:column;gap:4px">'
                            f'<div style="display:flex;justify-content:space-between;font-size:13px">'
                            f'<span style="color:{_DC["muted"]}">{label}</span>'
                            f'<span style="color:{color};font-weight:600">{val:.2f}</span></div>'
                            f'{_dbar(val, color)}'
                            f'<span style="font-size:11px;color:{color}">{verdict}</span></div>')

                grid = ",".join(["1fr"] * len(visible))
                cells = "".join(_hcell(l, v) for l, v in visible)
                st.markdown(
                    _dcard(f'{_dsec("System Health")}'
                           f'<div style="display:grid;grid-template-columns:{grid};gap:20px">{cells}</div>'),
                    unsafe_allow_html=True,
                )

        # Issues Detected
        issues: List[tuple] = []
        if avg_lat > 5000:
            issues.append(("HIGH", f"Avg latency {avg_lat:.0f}ms — consider caching frequent queries or reducing top_k."))
        if latest_run:
            met = latest_run.get("metrics", {})
            faith   = met.get("faithfulness")
            ctx_rec = met.get("context_recall")
            if faith is not None and faith < 0.6:
                issues.append(("MED", f"Faithfulness = {faith:.2f} — LLM may generate content not grounded in retrieved chunks."))
            if ctx_rec is not None and ctx_rec < 0.5:
                issues.append(("MED", f"Context Recall = {ctx_rec:.2f} — relevant chunks are being missed."))
        failed_q = sum(1 for q in recent_qs if q.get("num_sources", 1) == 0)
        if failed_q:
            issues.append(("HIGH", f"{failed_q} recent quer{'ies' if failed_q > 1 else 'y'} returned 0 sources."))
        if total_chunks < 50:
            issues.append(("LOW", f"Only {total_chunks} chunks indexed — small knowledge base limits coverage."))

        sev_color = {"HIGH": _DC["red"], "MED": _DC["orange"], "LOW": _DC["muted"]}
        if issues:
            rows_html = "".join(
                f'<div style="display:flex;gap:12px;padding:8px 12px;background:#1c2128;'
                f'border-radius:6px;font-size:12px;align-items:center">'
                f'{_dbadge(sev, sev_color[sev])}'
                f'<span style="color:{_DC["txt"]}">{msg}</span></div>'
                for sev, msg in issues
            )
            st.markdown(
                _dcard(f'{_dsec("⚠ Issues Detected", _DC["orange"])}'
                       f'<div style="display:flex;flex-direction:column;gap:8px">{rows_html}</div>',
                       _DC["orange"] + "44"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _dcard(f'{_dsec("✓ No Issues Detected", _DC["green"])}'
                       f'<span style="font-size:13px;color:{_DC["txt"]}">System looks healthy.</span>',
                       _DC["green"] + "44"),
                unsafe_allow_html=True,
            )

    # ═══════════════════════════ EVALUATION ══════════════════════════════════
    with tab_ev:
        # Run / Generate controls (always visible)
        hc, kc, rc = st.columns([4, 1, 1], gap="small")
        with hc:
            raw_cases = _load_eval_questions()
            src_label = ""
            if raw_cases:
                src = "built-in" if _BUILTIN_Q.exists() else "auto-generated"
                src_label = f'<span style="font-size:11px;color:{_DC["muted"]}">  ({len(raw_cases)} questions, {src})</span>'
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;color:{_DC["muted"]};'
                f'letter-spacing:0.1em;text-transform:uppercase;padding-top:8px;{_MONO}">'
                f'Run Evaluation{src_label}</div>',
                unsafe_allow_html=True,
            )
        with kc:
            with_ragas = st.checkbox("+RAGAS", value=False, key="eval_with_ragas",
                                     help="Include answer quality metrics (needs LM Studio)")
        with rc:
            st.markdown('<div class="run-btn">', unsafe_allow_html=True)
            run_clicked = st.button("▶ Run", key="btn_run_eval", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if not raw_cases:
            st.markdown(
                f'<div style="font-size:0.75rem;color:{_DC["muted"]};margin-bottom:0.3rem">No evaluation questions found.</div>',
                unsafe_allow_html=True,
            )
            gc, _ = st.columns([2, 3])
            with gc:
                if engine and engine._all_nodes:
                    if st.button("⚡ Generate from documents (needs LM Studio)", key="btn_gen_q"):
                        with st.spinner("Generating…"):
                            ok = _generate_questions_with_llm(engine)
                        if ok:
                            st.rerun()
                else:
                    st.markdown(f'<div style="font-size:0.72rem;color:{_DC["muted"]}">Ingest documents first.</div>', unsafe_allow_html=True)

        if run_clicked and engine:
            if not raw_cases:
                st.warning("Generate or provide evaluation questions first.")
            else:
                _run_eval_in_ui(engine, k=5, with_ragas=with_ragas)
                st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if not all_runs:
            st.markdown(
                f'<div style="color:{_DC["muted"]};font-size:13px;padding:12px 0;{_MONO}">'
                f'No runs yet — click ▶ Run above.</div>',
                unsafe_allow_html=True,
            )
        else:
            # Runs table
            col_tmpl = "2fr 1fr 1fr 1fr 1fr 1fr"
            def _th(t):
                return f'<span style="font-size:11px;color:{_DC["muted"]};letter-spacing:0.05em">{t}</span>'

            hdr = (f'<div style="display:grid;grid-template-columns:{col_tmpl};gap:8px;'
                   f'padding:8px 12px;{_MONO}">'
                   + _th("Run / Config") + _th("Hit@K") + _th("MRR") + _th("NDCG")
                   + _th("Faithful") + _th("Ctx Rec") + '</div>')

            if "dash_sel_run" not in st.session_state:
                st.session_state.dash_sel_run = 0

            rows_html = hdr
            for i, run in enumerate(all_runs):
                met  = run.get("metrics", {})
                k_   = met.get("k", run.get("config", {}).get("k", 5))
                vals = [met.get("hit_rate_at_k"), met.get("mrr"), met.get(f"ndcg_at_{k_}"),
                        met.get("faithfulness"), met.get("context_recall")]
                is_sel  = st.session_state.dash_sel_run == i
                bg_row  = "#1c2128" if is_sel else "transparent"
                brd_row = _DC["blue"] + "44" if is_sel else "transparent"
                metric_cells = "".join(
                    f'<span style="color:{_dcolor(v)};font-weight:600">{_dfmt(v)}</span>'
                    for v in vals
                )
                rows_html += (
                    f'<div style="display:grid;grid-template-columns:{col_tmpl};gap:8px;'
                    f'padding:10px 12px;border-radius:6px;font-size:13px;background:{bg_row};'
                    f'border:1px solid {brd_row};margin-bottom:2px;{_MONO}">'
                    f'<div><div style="color:{_DC["head"]}">{_run_label(run, emb_model)}</div>'
                    f'<div style="color:{_DC["muted"]};font-size:11px">{_run_date(run)}</div></div>'
                    f'{metric_cells}</div>'
                )

            st.markdown(
                _dcard(f'{_dsec("Evaluation Runs — select to inspect")}{rows_html}'),
                unsafe_allow_html=True,
            )

            sel = st.selectbox(
                "Select run",
                options=range(len(all_runs)),
                format_func=lambda i: f"{_run_label(all_runs[i], emb_model)}  [{_run_date(all_runs[i])}]",
                index=min(st.session_state.dash_sel_run, len(all_runs) - 1),
                key="dash_run_sel",
                label_visibility="collapsed",
            )
            st.session_state.dash_sel_run = sel

            # Per-question breakdown
            per_q = all_runs[sel].get("metrics", {}).get("per_question", [])
            if per_q:
                pq_hdr = (
                    f'<div style="display:grid;grid-template-columns:2fr 2fr 60px 70px;gap:8px;'
                    f'padding:8px 12px;font-size:11px;color:{_DC["muted"]};{_MONO}">'
                    f'<span>Question</span><span>Ground Truth</span><span>Rank</span><span>Hit</span></div>'
                )
                pq_rows = ""
                for item in per_q:
                    qid     = item.get("id", "")
                    tc      = test_by_id.get(qid, {})
                    q_text  = item.get("question", tc.get("question", qid))
                    gt      = tc.get("ground_truth", "—")
                    gt_s    = gt[:80] + ("…" if len(gt) > 80 else "")
                    hit     = item.get("hit", False)
                    rr      = item.get("rr", 0)
                    rank    = round(1 / rr) if rr and rr > 0 else None
                    pq_rows += (
                        f'<div style="display:grid;grid-template-columns:2fr 2fr 60px 70px;gap:8px;'
                        f'padding:8px 12px;font-size:12px;border-bottom:1px solid {_DC["border"]};'
                        f'align-items:start;{_MONO}">'
                        f'<span style="color:{_DC["txt"]}">{q_text}</span>'
                        f'<span style="color:{_DC["muted"]}">{gt_s}</span>'
                        f'<span style="color:{_DC["muted"]}">{rank if rank else "—"}</span>'
                        f'{_dbadge("✓ HIT", _DC["green"]) if hit else _dbadge("✗ MISS", _DC["red"])}'
                        f'</div>'
                    )
                sel_label = _run_label(all_runs[sel], emb_model)
                st.markdown(
                    _dcard(f'{_dsec(f"Question Breakdown — {sel_label}")}{pq_hdr}{pq_rows}'),
                    unsafe_allow_html=True,
                )

    # ═══════════════════════════ QUERIES ═════════════════════════════════════
    with tab_qr:
        if not recent_qs:
            st.markdown(
                f'<div style="color:{_DC["muted"]};font-size:13px;padding:20px 0;{_MONO}">'
                f'No queries logged yet — ask something in Chat.</div>',
                unsafe_allow_html=True,
            )
        else:
            def _lat_clr(ms: float) -> str:
                return _DC["red"] if ms > 7000 else (_DC["orange"] if ms > 5000 else _DC["green"])

            col_tmpl = "70px 1fr 90px 50px 80px 70px"
            q_hdr = (
                f'<div style="display:grid;grid-template-columns:{col_tmpl};gap:8px;'
                f'padding:8px 12px;font-size:11px;color:{_DC["muted"]};{_MONO}">'
                f'<span>Time</span><span>Question</span><span>Latency</span>'
                f'<span>Src</span><span>Mode</span><span>Status</span></div>'
            )
            q_rows = ""
            for q in recent_qs:
                ts = q.get("timestamp", "")
                try:
                    t_str = datetime.fromisoformat(ts).strftime("%H:%M")
                except Exception:
                    t_str = ts[11:16]
                question = q.get("question", "")
                q_s      = question[:65] + ("…" if len(question) > 65 else "")
                lat      = q.get("latency_ms", 0)
                num_src  = q.get("num_sources", 0)
                mode     = q.get("retrieval_mode", "?")
                passed   = num_src > 0
                q_rows += (
                    f'<div style="display:grid;grid-template-columns:{col_tmpl};gap:8px;'
                    f'padding:8px 12px;font-size:12px;border-bottom:1px solid {_DC["border"]};'
                    f'align-items:center;{_MONO}">'
                    f'<span style="color:{_DC["muted"]}">{t_str}</span>'
                    f'<span style="color:{_DC["txt"]};overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{q_s}</span>'
                    f'<span style="color:{_lat_clr(lat)}">{lat:.0f}ms</span>'
                    f'<span style="color:{_DC["muted"]}">{num_src}</span>'
                    f'{_dbadge(mode, _DC["purple"])}'
                    f'{_dbadge("ok", _DC["green"]) if passed else _dbadge("fail", _DC["red"])}'
                    f'</div>'
                )
            st.markdown(
                _dcard(f'{_dsec("Recent Queries")}{q_hdr}{q_rows}'),
                unsafe_allow_html=True,
            )

    # ═══════════════════════════ COMPARE ═════════════════════════════════════
    with tab_cmp:
        if not all_runs:
            st.markdown(
                f'<div style="color:{_DC["muted"]};font-size:13px;padding:20px 0;{_MONO}">'
                f'No runs to compare yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            compare_metrics = [
                ("hit_rate_at_k", "Hit Rate"),
                ("mrr",           "MRR"),
                ("faithfulness",  "Faithfulness"),
                ("context_recall","Context Recall"),
            ]
            cmp_inner = _dsec("Model Comparison")
            for met_key, met_label in compare_metrics:
                runs_with_val = [(r, r.get("metrics", {}).get(met_key)) for r in all_runs
                                 if r.get("metrics", {}).get(met_key) is not None]
                if not runs_with_val:
                    continue
                cmp_inner += f'<div style="margin-bottom:20px"><div style="font-size:12px;color:{_DC["muted"]};margin-bottom:8px">{met_label}</div>'
                for run, v in runs_with_val:
                    color = _dcolor(v)
                    lbl   = _run_label(run, emb_model)
                    cmp_inner += (
                        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
                        f'<span style="font-size:11px;color:{_DC["muted"]};width:220px;flex-shrink:0;'
                        f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{lbl}</span>'
                        f'<div style="flex:1">{_dbar(v, color)}</div>'
                        f'<span style="font-size:12px;font-weight:600;width:40px;text-align:right;color:{color}">{v:.2f}</span>'
                        f'</div>'
                    )
                cmp_inner += '</div>'
            st.markdown(_dcard(cmp_inner), unsafe_allow_html=True)

            # Recommendation
            best_run  = max(all_runs, key=lambda r: r.get("metrics", {}).get("hit_rate_at_k", 0))
            best_met  = best_run.get("metrics", {})
            best_hr_v = best_met.get("hit_rate_at_k")
            best_mrr  = best_met.get("mrr")
            best_faith = best_met.get("faithfulness")

            steps = []
            if best_faith is None or best_faith < 0.6:
                steps.append("Tighten system prompt to enforce source citations")
            if best_faith is not None and best_faith < 0.8:
                steps.append("Add cross-encoder reranker for better relevance scoring")
            steps.append("Consider bge-m3 for multilingual query support")
            steps_html = "".join(f'<div>— {s}</div>' for s in steps)

            st.markdown(
                _dcard(
                    f'{_dsec("✓ Recommendation", _DC["green"])}'
                    f'<div style="font-size:13px;color:{_DC["txt"]};line-height:1.8">'
                    f'Best config: <strong style="color:{_DC["head"]}">{_run_label(best_run, emb_model)}</strong><br>'
                    f'Hit Rate: <strong style="color:{_DC["green"]}">{_dfmt(best_hr_v)}</strong>'
                    f' · MRR: <strong style="color:{_DC["green"]}">{_dfmt(best_mrr)}</strong><br><br>'
                    f'<span style="color:{_DC["muted"]}">Next steps to improve Faithfulness ({_dfmt(best_faith)}):<br>'
                    f'{steps_html}</span></div>',
                    _DC["green"] + "44",
                ),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _init()
    engine, qlogger, error = load_engine()
    render_sidebar(engine)

    tab_chat, tab_dash = st.tabs(["💬  Chat", "📊  Dashboard"])

    with tab_chat:
        if error and engine is None:
            st.error(f"Engine failed to initialise: {error}")
            st.code("python src/ingestor.py --data-dir data --db-path db --recreate")
        else:
            render_chat(engine, qlogger)

    with tab_dash:
        render_dashboard(engine, qlogger)


if __name__ == "__main__":
    main()
