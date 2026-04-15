import streamlit as st
import os
from pdf_utils import load_and_chunk_pdfs
from embeddings import EmbeddingManager
from retrieval import HybridRetriever
from qa import QASystem
import tempfile

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Neural Knowledge Engine",
    page_icon="🧠",
    layout="wide"
)

# =========================
# ULTRA UI (HUD + APPLE + ANIMATIONS)
# =========================
st.markdown("""
<style>

/* GLOBAL */
html, body {
    background: radial-gradient(circle at top, #0a0f2c, #020617);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ANIMATED BACKGROUND GLOW */
body::before {
    content: "";
    position: fixed;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59,130,246,0.15), rgba(239,68,68,0.1));
    animation: pulseBG 8s infinite alternate;
    z-index: -1;
}

@keyframes pulseBG {
    0% { transform: translate(-25%, -25%) scale(1); }
    100% { transform: translate(-30%, -30%) scale(1.2); }
}

/* TITLE */
.title-glow {
    font-size: 2.8rem;
    font-weight: 900;
    text-align: center;
    color: #60a5fa;
    letter-spacing: 1px;
    text-shadow:
        0 0 10px #3b82f6,
        0 0 20px #1d4ed8,
        0 0 30px #ef4444;
    animation: glowPulse 2s infinite alternate;
}

@keyframes glowPulse {
    from { text-shadow: 0 0 10px #3b82f6; }
    to { text-shadow: 0 0 25px #ef4444; }
}

/* GLASS CARD */
.glass {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(16px);
    border-radius: 18px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}

/* HOVER GLOW */
.glass:hover {
    transform: translateY(-6px);
    box-shadow:
        0 0 20px #3b82f6,
        0 0 40px #ef4444;
}

/* CHAT BUBBLE */
[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 14px;
    padding: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* INPUT */
textarea {
    background: rgba(2,6,23,0.9) !important;
    color: #fff !important;
    border-radius: 12px !important;
    border: 1px solid #1e293b !important;
}

/* INPUT FOCUS ANIMATION */
textarea:focus {
    border: 1px solid #ef4444 !important;
    box-shadow:
        0 0 10px #ef4444,
        0 0 20px #3b82f6 !important;
}

/* BUTTON */
button[kind="primary"] {
    background: linear-gradient(90deg, #3b82f6, #ef4444) !important;
    border-radius: 12px !important;
    font-weight: bold;
}

/* BUTTON HOVER */
button:hover {
    box-shadow:
        0 0 15px #3b82f6,
        0 0 25px #ef4444 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0a0f2c);
}

/* HUD LINES */
.hud {
    position: relative;
}

.hud::before, .hud::after {
    content: "";
    position: absolute;
    border: 1px solid rgba(59,130,246,0.3);
    width: 20px;
    height: 20px;
}

.hud::before {
    top: 0;
    left: 0;
    border-right: none;
    border-bottom: none;
}

.hud::after {
    bottom: 0;
    right: 0;
    border-left: none;
    border-top: none;
}

/* SCROLLBAR */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#3b82f6, #ef4444);
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title-glow">🧠 NEURAL KNOWLEDGE ENGINE</div>', unsafe_allow_html=True)
st.markdown("### ⚡ Hybrid Intelligence • Retrieval • Reasoning")

# =========================
# SESSION STATE
# =========================
for key in ["documents","embedding_manager","retriever","qa_system","chat_history","processed_files"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history","processed_files"] else None

# =========================
# API KEYS
# =========================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Add GROQ_API_KEY")
    st.stop()

qdrant_api_key = st.secrets.get("QDRANT_API_KEY", None)
qdrant_endpoint = st.secrets.get("QDRANT_ENDPOINT", None)
use_qdrant = bool(qdrant_api_key and qdrant_endpoint)

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown("## 📁 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Drop files", type="pdf", accept_multiple_files=True)

# =========================
# PROCESSING
# =========================
if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if current_files != st.session_state.processed_files:
        with st.spinner("⚡ Neural Indexing..."):
            temp_paths = []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    temp_paths.append(tmp.name)

            documents = load_and_chunk_pdfs(temp_paths, chunk_size=150, overlap=30)

            for p in temp_paths:
                os.unlink(p)

            st.session_state.documents = documents

            embedding_manager = EmbeddingManager(
                use_qdrant=use_qdrant,
                qdrant_api_key=qdrant_api_key,
                qdrant_endpoint=qdrant_endpoint
            )
            embedding_manager.build_index(documents)

            retriever = HybridRetriever(embedding_manager, documents, top_k=7)
            qa_system = QASystem(groq_api_key, retriever)

            st.session_state.embedding_manager = embedding_manager
            st.session_state.retriever = retriever
            st.session_state.qa_system = qa_system
            st.session_state.chat_history = []
            st.session_state.processed_files = current_files

# =========================
# MAIN UI
# =========================
if st.session_state.documents:

    st.markdown('<div class="glass hud">', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.chat_history.append({"role":"user","content":question})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""

            for chunk in st.session_state.qa_system.answer_question(
                question,
                chat_history=st.session_state.chat_history[-10:]
            ):
                response += chunk
                placeholder.markdown(response + "▌")

            placeholder.markdown(response)

            st.session_state.chat_history.append({"role":"assistant","content":response})

else:
    st.markdown('<div class="glass hud">', unsafe_allow_html=True)
    st.info("👈 Upload PDFs to activate system")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("⚡ SYSTEM ACTIVE")
