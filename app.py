import streamlit as st
import os
import tempfile
import time

import google.generativeai as genai
import faiss
import numpy as np
import pypdf

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Knowledge Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Share+Tech+Mono&display=swap');

:root {
    --cyan:    #00f2ff;
    --violet:  #7000ff;
    --pink:    #ff006e;
    --dark:    #07070a;
    --card:    #0d0d12;
    --border:  #1a1a2e;
    --text:    #c8ccd8;
    --dim:     #5a5f72;
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 16px;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,242,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,242,255,.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a10 0%, #0d0d18 100%) !important;
    border-right: 1px solid var(--violet) !important;
    box-shadow: 4px 0 30px rgba(112,0,255,.15);
}

[data-testid="stSidebar"] * { font-family: 'Rajdhani', sans-serif !important; }

/* ── Headings ── */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    font-size: 1.6rem !important;
    background: linear-gradient(90deg, var(--cyan), var(--violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 3px !important;
    text-transform: uppercase;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--cyan) !important;
    letter-spacing: 2px !important;
    font-size: 0.95rem !important;
    text-transform: uppercase;
}

/* ── Sidebar title ── */
[data-testid="stSidebar"] h1 {
    font-size: 1.1rem !important;
    letter-spacing: 4px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #0a0a10 !important;
    color: var(--cyan) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: all .3s ease;
    box-shadow: 0 0 0px transparent;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,242,255,.25) !important;
    outline: none !important;
}

/* ── Labels ── */
.stTextInput label, .stFileUploader label, p {
    color: var(--dim) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0a0a10 !important;
    border: 1px dashed var(--violet) !important;
    border-radius: 8px !important;
    transition: all .3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,242,255,.12);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--violet), #4400cc) !important;
    color: #fff !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    width: 100%;
    padding: 0.65rem 1rem !important;
    transition: all .3s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, transparent, rgba(0,242,255,.15), transparent);
    opacity: 0;
    transition: opacity .3s ease;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 25px rgba(112,0,255,.6), 0 0 60px rgba(0,242,255,.15) !important;
    transform: translateY(-1px) !important;
    color: var(--cyan) !important;
}
.stButton > button:hover::after { opacity: 1; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    padding: 1rem !important;
    transition: border-color .3s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(112,0,255,.4) !important;
}

/* User message accent */
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid var(--violet) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #0a0a10 !important;
    color: var(--cyan) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,242,255,.2) !important;
}

/* ── Success / Error / Info alerts ── */
[data-testid="stAlert"] {
    border-radius: 6px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: .5px;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb {
    background: var(--violet);
    border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* ── Status badge ── */
.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin-top: 8px;
}
.badge-online {
    background: rgba(0,242,255,.1);
    border: 1px solid var(--cyan);
    color: var(--cyan);
}
.badge-offline {
    background: rgba(112,0,255,.1);
    border: 1px solid var(--violet);
    color: var(--violet);
}

/* ── Source pills ── */
.source-pill {
    display: inline-block;
    background: rgba(112,0,255,.15);
    border: 1px solid rgba(112,0,255,.4);
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #a080ff;
    margin: 2px;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 10px;
    margin: 10px 0;
}
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--cyan);
}
.metric-label {
    font-size: 0.68rem;
    color: var(--dim);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Markdown inside chat */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
[data-testid="stChatMessage"] strong { color: var(--cyan) !important; }
[data-testid="stChatMessage"] code {
    background: #0a0a10 !important;
    color: var(--pink) !important;
    border-radius: 3px;
    padding: 1px 5px;
    font-family: 'Share Tech Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── BACKEND ───────────────────────────────────────────────────────────────────

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL  = "gemini-2.0-flash"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 50   # Gemini free-tier safe batch size
TOP_K = 5


def configure_gemini(api_key: str):
    genai.configure(api_key=api_key)


def extract_chunks(file_bytes: bytes, filename: str):
    """Extract text from PDF and split into overlapping chunks."""
    chunks, meta = [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        reader = pypdf.PdfReader(tmp)
        for page_num, page in enumerate(reader.pages):
            raw = (page.extract_text() or "").strip()
            if not raw:
                continue
            # Clean whitespace
            text = " ".join(raw.split())
            start = 0
            while start < len(text):
                piece = text[start : start + CHUNK_SIZE].strip()
                if len(piece) > 50:   # skip tiny fragments
                    chunks.append(piece)
                    meta.append({"source": filename, "page": page_num + 1})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)
    return chunks, meta


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts using Gemini text-embedding-004."""
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type="retrieval_document",
        )
        all_vecs.extend(result["embedding"])
    return np.array(all_vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return np.array(result["embedding"], dtype="float32").reshape(1, -1)


def build_index(uploaded_files):
    """Parse PDFs → chunk → embed → FAISS index."""
    all_chunks, all_meta = [], []
    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, None, None, "No readable text found in the uploaded PDFs."

    progress = st.progress(0, text="Embedding chunks…")
    try:
        vecs = []
        for i in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[i : i + EMBED_BATCH]
            result = genai.embed_content(
                model=EMBED_MODEL,
                content=batch,
                task_type="retrieval_document",
            )
            vecs.extend(result["embedding"])
            progress.progress(
                min((i + EMBED_BATCH) / len(all_chunks), 1.0),
                text=f"Embedding… {min(i + EMBED_BATCH, len(all_chunks))}/{len(all_chunks)} chunks"
            )
            time.sleep(0.05)   # gentle rate-limit buffer
    except Exception as e:
        progress.empty()
        return None, None, None, str(e)

    progress.empty()
    matrix = np.array(vecs, dtype="float32")
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, all_chunks, all_meta, None


def retrieve(query: str, index, chunks, meta):
    q_vec = embed_query(query)
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, TOP_K)
    return [{"text": chunks[i], **meta[i]} for i in ids[0] if i != -1]


def generate_answer(query: str, docs: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {d['source']} | Page {d['page']}]\n{d['text']}"
        for d in docs
    )
    prompt = (
        "You are a precise, expert research assistant. "
        "Answer the user's question using ONLY the context provided below. "
        "Be thorough but concise. Use markdown formatting where helpful. "
        "If the answer cannot be found in the context, clearly state that.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}"
    )
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    return response.text


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1>⚙ ENGINE CORE</h1>", unsafe_allow_html=True)

    # API key — prefer st.secrets
    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        placeholder="AIza...",
        help="Get your key at aistudio.google.com"
    )

    # Engine status badge
    is_ready = "index" in st.session_state
    badge_class = "badge-online" if is_ready else "badge-offline"
    badge_text  = "● ONLINE" if is_ready else "○ OFFLINE"
    st.markdown(
        f'<div class="status-badge {badge_class}">{badge_text}</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("<h3>Document Ingestion</h3>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(
            f'<p style="color:#a080ff;font-size:0.8rem;">'
            f'{len(uploaded_files)} file(s) selected</p>',
            unsafe_allow_html=True
        )

    if st.button("⚡  INITIALIZE ENGINE"):
        if not api_key:
            st.error("Gemini API key required.")
        elif not uploaded_files:
            st.error("Upload at least one PDF first.")
        else:
            try:
                configure_gemini(api_key)
                with st.spinner("Parsing documents…"):
                    idx, chunks, meta, err = build_index(uploaded_files)
                if err:
                    st.error(f"Indexing error: {err}")
                else:
                    st.session_state.update(
                        index=idx,
                        chunks=chunks,
                        meta=meta,
                        api_key=api_key,
                    )
                    total_pages = len({m["page"] for m in meta})
                    st.success(f"Indexed {len(chunks)} chunks · {total_pages} pages")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # Stats
    if is_ready:
        st.divider()
        st.markdown("<h3>Index Stats</h3>", unsafe_allow_html=True)
        chunks = st.session_state.chunks
        meta   = st.session_state.meta
        files  = list({m["source"] for m in meta})
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{len(chunks)}</div>
                    <div class="metric-label">Chunks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(files)}</div>
                    <div class="metric-label">Files</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        for f in files:
            st.markdown(
                f'<div class="source-pill">📄 {f}</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown(
        '<p style="font-size:0.68rem;color:#333;text-align:center;letter-spacing:1px;">'
        'POWERED BY GEMINI · FAISS</p>',
        unsafe_allow_html=True
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

st.markdown("<h1>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:#5a5f72;font-size:0.85rem;letter-spacing:2px;margin-bottom:1.5rem;">'
    'RAG · SEMANTIC SEARCH · DOCUMENT INTELLIGENCE</p>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Welcome state
if not st.session_state.messages and not is_ready:
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;opacity:.6;">
        <div style="font-family:'Orbitron',monospace;font-size:2.5rem;color:#1a1a2e;">⬡</div>
        <p style="font-family:'Orbitron',monospace;font-size:0.75rem;letter-spacing:3px;color:#333;margin-top:1rem;">
            AWAITING DOCUMENT INGESTION
        </p>
        <p style="font-size:0.85rem;color:#333;margin-top:0.5rem;">
            Upload PDFs in the sidebar and initialise the engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Query the knowledge base…"):
    if not api_key:
        st.error("Enter your Gemini API Key in the sidebar.")
    elif "index" not in st.session_state:
        st.error("Initialize the engine first — upload PDFs and click INITIALIZE ENGINE.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing answer…"):
                try:
                    configure_gemini(st.session_state.api_key)
                    docs = retrieve(
                        prompt,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.meta,
                    )
                    answer_text = generate_answer(prompt, docs)

                    # Build source pills
                    seen = set()
                    src_html = ""
                    for d in docs:
                        label = f"{d['source']} p.{d['page']}"
                        if label not in seen:
                            seen.add(label)
                            src_html += f'<span class="source-pill">📄 {label}</span>'

                    full_md = (
                        answer_text
                        + f"\n\n<div style='margin-top:12px;'><span style='font-size:0.72rem;"
                          f"color:#5a5f72;letter-spacing:1px;text-transform:uppercase;'>"
                          f"Sources</span><br>{src_html}</div>"
                    )
                    st.markdown(full_md, unsafe_allow_html=True)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_md}
                    )

                except Exception as e:
                    err = str(e)
                    if "API_KEY" in err.upper() or "401" in err:
                        st.error("Invalid API key — check your Gemini key.")
                    elif "429" in err or "quota" in err.lower():
                        st.error("Rate limit hit — wait a moment and try again.")
                    elif "404" in err:
                        st.error(f"Model not found: {err}")
                    else:
                        st.error(f"Error: {err}")

# Clear chat button (only when there are messages)
if st.session_state.messages:
    if st.button("Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()
