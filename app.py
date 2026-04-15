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
# CUSTOM CSS (🔥 GLOW UI)
# =========================
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e2e8f0;
}

/* Title Glow */
.title-glow {
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    color: #38bdf8;
    text-shadow: 0 0 10px #38bdf8, 0 0 20px #0ea5e9;
}

/* Card Style */
.glow-card {
    background: rgba(15, 23, 42, 0.7);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #1e293b;
    transition: 0.3s ease-in-out;
}

/* Hover Glow Effect */
.glow-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 0 20px #38bdf8, 0 0 40px #0ea5e9;
    border: 1px solid #38bdf8;
}

/* Chat bubbles */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

/* Input box glow */
textarea {
    background-color: #020617 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #1e293b !important;
}

textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 10px #38bdf8 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #020617;
}

/* Buttons */
button[kind="primary"] {
    background: #0ea5e9 !important;
    border-radius: 10px !important;
}

button:hover {
    box-shadow: 0 0 10px #38bdf8 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="title-glow">🧠 NEURAL KNOWLEDGE ENGINE</div>', unsafe_allow_html=True)
st.markdown("### ⚡ Hybrid Intelligence (FAISS + BM25 + LLM Reasoning)")

# =========================
# SESSION STATE INIT
# =========================
for key in ["documents", "embedding_manager", "retriever", "qa_system", "chat_history", "processed_files"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["chat_history", "processed_files"] else None

# =========================
# API KEYS
# =========================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("⚠️ Add GROQ_API_KEY in Streamlit Secrets")
    st.stop()

qdrant_api_key = st.secrets.get("QDRANT_API_KEY", None)
qdrant_endpoint = st.secrets.get("QDRANT_ENDPOINT", None)
use_qdrant = bool(qdrant_api_key and qdrant_endpoint)

# =========================
# SIDEBAR (UPLOAD)
# =========================
st.sidebar.markdown("## 📁 Upload Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Drop PDFs",
    type="pdf",
    accept_multiple_files=True
)

# =========================
# PROCESS FILES
# =========================
if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if current_files != st.session_state.processed_files:
        with st.spinner("⚡ Indexing Neural Data..."):
            try:
                temp_paths = []
                for f in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(f.read())
                        temp_paths.append(tmp.name)

                documents = load_and_chunk_pdfs(temp_paths, chunk_size=150, overlap=30)

                for path in temp_paths:
                    os.unlink(path)

                if not documents:
                    st.error("❌ No extractable text found")
                    st.stop()

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

                st.sidebar.success(f"✅ {len(uploaded_files)} PDFs → {len(documents)} chunks")

            except Exception as e:
                st.error(str(e))
                st.stop()

# =========================
# MAIN CHAT UI
# =========================
if st.session_state.documents:

    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Neural Conversation")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    st.markdown('</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask anything...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""

            try:
                for chunk in st.session_state.qa_system.answer_question(
                    question,
                    chat_history=st.session_state.chat_history[-10:]
                ):
                    response += chunk
                    placeholder.markdown(response + "▌")

                placeholder.markdown(response)

                st.session_state.chat_history.append({"role": "assistant", "content": response})

                # Debug viewer
                with st.expander("🔍 Neural Retrieval Insights"):
                    results = st.session_state.retriever.hybrid_search(question)
                    for i, (doc, score) in enumerate(results[:3], 1):
                        st.markdown(f"**Chunk {i} | Score: {score:.3f}**")
                        st.text(doc[:300] + "...")

            except Exception as e:
                st.error(str(e))

    if st.sidebar.button("🧹 Reset Memory"):
        st.session_state.chat_history = []
        st.rerun()

else:
    st.markdown('<div class="glow-card">', unsafe_allow_html=True)
    st.info("👈 Upload PDFs to activate the engine")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🧠 Capabilities
        - Hybrid Retrieval (FAISS + BM25)
        - Deep Semantic Understanding
        - Multi-document reasoning
        """)

    with col2:
        st.markdown("""
        ### ⚡ Performance
        - Real-time streaming
        - Context-aware answers
        - Persistent vector DB (Qdrant optional)
        """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("⚡ NEURAL CORE ACTIVE")
