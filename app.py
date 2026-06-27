"""
app.py — ResearchXpert Streamlit UI (no eval metrics)
Run: streamlit run app.py
"""

import base64
import streamlit as st
from rag import RAGEngine

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchXpert",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo helper ────────────────────────────────────────────────────────────────
def get_logo_b64() -> str:
    try:
        with open("logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

LOGO_B64 = get_logo_b64()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background: #0a0a0f;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
/* ===== Banner Logo ===== */

.rx-header {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0.5rem 0 1.5rem 0;
    margin-bottom: 10px;
}

.rx-logo {
    width: 100%;
    max-width: 1200px;
    height: 180px;
    object-fit: contain;
    display: block;
}
.card {
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}

.bubble-user {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 16px 16px 3px 16px;
    padding: 0.85rem 1.1rem; margin: 0.6rem 0 0.6rem 18%;
    color: #fff; font-size: 0.93rem; line-height: 1.55;
    box-shadow: 0 4px 20px rgba(79,70,229,0.25);
}
.bubble-ai {
    background: #13131f; border: 1px solid #1e1e30;
    border-radius: 16px 16px 16px 3px;
    padding: 0.85rem 1.1rem; margin: 0.6rem 18% 0.6rem 0;
    color: #e2e8f0; font-size: 0.93rem; line-height: 1.6;
}
.bubble-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em; color: #475569; margin: 0.2rem 4px; }
.bubble-label.right { text-align: right; color: #6366f1; }
.bubble-label.left  { text-align: left;  color: #38bdf8; }

.pdf-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1a2e; border: 1px solid #2a2a45;
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.78rem; color: #a78bfa; margin: 3px 3px 3px 0;
}

.stTextInput > div > div > input {
    background: #13131f !important; color: #e2e8f0 !important;
    border: 1px solid #1e1e30 !important; border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: #fff !important; border: none !important; border-radius: 9px !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
}
[data-testid="stFileUploader"] {
    background: #0f0f1a !important; border: 2px dashed #2d2d50 !important; border-radius: 12px !important;
}
[data-testid="stMetricValue"] { color: #818cf8 !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.78rem !important; }
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: #1e1e30; }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "engine":       None,
        "chat_history": [],
        "model":        "llama-3.3-70b-versatile",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    groq_api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...",
                                  help="Get a free key at console.groq.com")
    jina_api_key = st.text_input("🔑 Jina API Key", type="password", placeholder="jina_...",
                                  help="Free 1M tokens/month — get key at jina.ai")

    model_choice = st.selectbox("🤖 Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], index=0)
    st.session_state["model"] = model_choice

    st.divider()
    st.markdown("### 📤 Upload PDFs (max 3)")

    uploaded_files = st.file_uploader("Drop PDFs here", type=["pdf"],
                                       accept_multiple_files=True,
                                       label_visibility="collapsed")

    if uploaded_files and len(uploaded_files) > 3:
        st.warning("⚠️ Only the first 3 PDFs will be processed.")
        uploaded_files = uploaded_files[:3]

    if uploaded_files and groq_api_key and jina_api_key:
        if st.button("⚡ Process PDFs", use_container_width=True):
            engine = RAGEngine(groq_api_key=groq_api_key, jina_api_key=jina_api_key, model_name=model_choice)
            st.session_state["engine"]       = engine
            st.session_state["chat_history"] = []
            for uf in uploaded_files:
                with st.spinner(f"Embedding {uf.name}…"):
                    try:
                        meta = engine.add_pdf(uf.read(), uf.name)
                        sections_preview = ", ".join(meta.sections_found[:5])
                        if len(meta.sections_found) > 5:
                            sections_preview += f" +{len(meta.sections_found)-5} more"
                        st.success(f"✅ {meta.name} — {meta.pages}p / {meta.chunks} chunks")
                        st.caption(f"📑 Sections: {sections_preview}")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")
    elif uploaded_files and not (groq_api_key and jina_api_key):
        st.info("Enter both API keys above first.")

    st.divider()

    engine = st.session_state.get("engine")
    if engine and engine.ready:
        st.markdown("**Loaded documents**")
        for p in engine.pdfs:
            st.markdown(f"<div class='pdf-pill'>📄 {p.name} <span style='color:#475569'>({p.pages}p)</span></div>",
                        unsafe_allow_html=True)
        if st.button("🗑️ Remove all PDFs", use_container_width=True):
            engine.reset()
            st.session_state["chat_history"] = []
            st.rerun()

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    st.divider()
    st.markdown("""
    <div style='color:#334155;font-size:0.75rem;line-height:1.7;'>
        <b style='color:#475569;'>Embeddings</b><br>
        jina-embeddings-v3 via Jina AI<br>(cloud — zero local RAM)<br><br>
        <b style='color:#475569;'>LLM</b><br>Llama 3.3 70B via Groq<br><br>
        <b style='color:#475569;'>Stack</b><br>LangChain · FAISS · Groq + Jina
    </div>""", unsafe_allow_html=True)


# ── Main Header ────────────────────────────────────────────────────────────────
# ── Large Banner Header ───────────────────────────────────────────────────────

if LOGO_B64:
    st.markdown(
        f"""
        <div class="rx-header">
            <img src="data:image/png;base64,{LOGO_B64}" class="rx-logo">
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="rx-header">
            <h1 style="
                font-size:3rem;
                color:#818cf8;
                text-align:center;
                margin:0;
            ">
                🔬 ResearchXpert
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
engine = st.session_state.get("engine")
c1, c2, c3, c4 = st.columns(4)
c1.metric("PDFs loaded",    f"{engine.pdf_count if engine else 0} / 3")
c2.metric("Chunks indexed", engine.total_chunks if engine else "—")
c3.metric("Model",          model_choice.split("-")[0].upper())
c4.metric("Q&A turns",      len(st.session_state["chat_history"]))

st.divider()


# ── Chat ───────────────────────────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    if not (engine and engine.ready):
        st.markdown("""
        <div class='card' style='text-align:center;padding:2.5rem 1rem;'>
            <div style='font-size:2.8rem;'>📚</div>
            <p style='color:#a78bfa;font-weight:700;font-size:1.1rem;margin:0.5rem 0 0.2rem;'>Upload up to 3 PDFs to get started</p>
            <p style='color:#334155;font-size:0.88rem;'>Add your API keys → upload PDFs → click Process PDFs</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='card' style='text-align:center;padding:2rem 1rem;'>
            <div style='font-size:2.4rem;'>✅</div>
            <p style='color:#34d399;font-weight:700;font-size:1rem;margin:0.4rem 0 0.2rem;'>Documents ready — ask anything</p>
        </div>""", unsafe_allow_html=True)
else:
    for turn in st.session_state["chat_history"]:
        st.markdown("<div class='bubble-label right'>You</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble-user'>{turn['question']}</div>", unsafe_allow_html=True)
        st.markdown("<div class='bubble-label left'>🔬 ResearchXpert</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='bubble-ai'>{turn['answer']}</div>", unsafe_allow_html=True)

        if turn.get("sources"):
            with st.expander("📎 Retrieved chunks", expanded=False):
                for i, doc in enumerate(turn["sources"], 1):
                    src = doc.metadata.get("source_file", "unknown")
                    pg  = doc.metadata.get("page", "?")
                    sec = doc.metadata.get("section", "")
                    st.markdown(f"**Chunk {i}** · `{src}` · page {pg}" + (f" · *{sec}*" if sec else ""))
                    st.text(doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""))

        st.markdown("<br>", unsafe_allow_html=True)

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
with st.form("q_form", clear_on_submit=True):
    qcol, bcol = st.columns([5, 1])
    with qcol:
        question = st.text_input("question",
                                  placeholder="What is the main contribution of this paper?",
                                  label_visibility="collapsed")
    with bcol:
        submitted = st.form_submit_button("Ask ➤", use_container_width=True)

if submitted:
    engine = st.session_state.get("engine")
    if not groq_api_key or not jina_api_key:
        st.warning("⚠️ Enter both API keys in the sidebar.")
    elif not (engine and engine.ready):
        st.warning("⚠️ Upload and process at least one PDF first.")
    elif not question.strip():
        st.warning("⚠️ Please type a question.")
    else:
        with st.spinner("🔍 Retrieving & generating…"):
            try:
                ans, docs = engine.answer(question)
                st.session_state["chat_history"].append({
                    "question": question, "answer": ans, "sources": docs,
                })
                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")

# ── Suggestions ────────────────────────────────────────────────────────────────
engine = st.session_state.get("engine")
if engine and engine.ready and not st.session_state["chat_history"]:
    st.markdown("#### 💡 Try asking")
    suggestions = [
        "Summarize the abstract",
        "What methodology was used?",
        "What are the key findings?",
        "Give an overview of the proposed model",
        "What are the limitations?",
        "What future work is suggested?",
    ]
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(s, key=f"sug_{i}", use_container_width=True):
                if not groq_api_key or not jina_api_key:
                    st.warning("⚠️ Enter both API keys first.")
                else:
                    with st.spinner("🔍 Thinking…"):
                        try:
                            ans, docs = engine.answer(s)
                            st.session_state["chat_history"].append({
                                "question": s, "answer": ans, "sources": docs,
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
