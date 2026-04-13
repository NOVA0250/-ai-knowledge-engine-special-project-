import streamlit as st
import os
import tempfile
import time

from google import genai
from google.genai import types
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&family=Share+Tech+Mono&display=swap');

:root {
    --cyan:   #00f2ff;
    --violet: #7000ff;
    --pink:   #ff006e;
    --dark:   #07070a;
    --card:   #0d0d12;
    --border: #1a1a2e;
    --text:   #c8ccd8;
    --dim:    #5a5f72;
}

html, body, .stApp {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}
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
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0a10,#0d0d18) !important;
    border-right: 1px solid var(--violet) !important;
    box-shadow: 4px 0 30px rgba(112,0,255,.15);
}
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
}
h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--cyan) !important;
    letter-spacing: 2px !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }

.stTextInput > div > div > input {
    background: #0a0a10 !important;
    color: var(--cyan) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    transition: all .3s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 12px rgba(0,242,255,.25) !important;
}
.stTextInput label, .stFileUploader label, p {
    color: var(--dim) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.78rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase;
}
[data-testid="stFileUploader"] {
    background: #0a0a10 !important;
    border: 1px dashed var(--violet) !important;
    border-radius: 8px !important;
    transition: all .3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,242,255,.1);
}
.stButton > button {
    background: linear-gradient(135deg, var(--violet), #4400cc) !important;
    color: #fff !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
    width: 100%;
    padding: 0.65rem 1rem !important;
    transition: all .3s !important;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 25px rgba(112,0,255,.6), 0 0 60px rgba(0,242,255,.15) !important;
    transform: translateY(-1px) !important;
    color: var(--cyan) !important;
}
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
    padding: 1rem !important;
    transition: border-color .3s;
}
[data-testid="stChatMessage"]:hover { border-color: rgba(112,0,255,.4) !important; }
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
[data-testid="stProgressBar"] > div { background: linear-gradient(90deg,var(--violet),var(--cyan)) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--dark); }
::-webkit-scrollbar-thumb { background: var(--violet); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

.status-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    margin-top: 8px;
}
.badge-online  { background: rgba(0,242,255,.1); border: 1px solid var(--cyan);   color: var(--cyan);   }
.badge-offline { background: rgba(112,0,255,.1); border: 1px solid var(--violet); color: var(--violet); }
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
.metric-row { display: flex; gap: 10px; margin: 10px 0; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
}
.metric-value { font-family: 'Orbitron', monospace; font-size: 1.4rem; font-weight: 700; color: var(--cyan); }
.metric-label { font-size: 0.68rem; color: var(--dim); letter-spacing: 1px; text-transform: uppercase; }
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


# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-004"   # no "models/" prefix with new SDK
CHAT_MODEL  = "gemini-2.0-flash"
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
EMBED_BATCH   = 25    # conservative for free-tier
TOP_K         = 5


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_client(api_key: str):
    return genai.Client(api_key=api_key)


def extract_chunks(file_bytes: bytes, filename: str):
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
            text = " ".join(raw.split())   # normalise whitespace
            start = 0
            while start < len(text):
                piece = text[start : start + CHUNK_SIZE].strip()
                if len(piece) > 60:
                    chunks.append(piece)
                    meta.append({"source": filename, "page": page_num + 1})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)
    return chunks, meta


def embed_batch(client, texts: list, task: str) -> np.ndarray:
    """Embed a list of strings with the new google-genai SDK."""
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return np.array([e.values for e in result.embeddings], dtype="float32")


def build_index(uploaded_files, client):
    all_chunks, all_meta = [], []
    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, None, None, "No readable text found in the uploaded PDFs."

    progress = st.progress(0, text="Embedding chunks…")
    all_vecs = []
    try:
        for i in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[i : i + EMBED_BATCH]
            vecs  = embed_batch(client, batch, "RETRIEVAL_DOCUMENT")
            all_vecs.append(vecs)
            pct = min((i + EMBED_BATCH) / len(all_chunks), 1.0)
            progress.progress(pct, text=f"Embedding {min(i+EMBED_BATCH, len(all_chunks))}/{len(all_chunks)} chunks…")
            time.sleep(0.1)
    except Exception as e:
        progress.empty()
        return None, None, None, str(e)

    progress.empty()
    matrix = np.vstack(all_vecs)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, all_chunks, all_meta, None


def retrieve(query: str, index, chunks, meta, client):
    q_vec = embed_batch(client, [query], "RETRIEVAL_QUERY")
    faiss.normalize_L2(q_vec)
    _, ids = index.search(q_vec, TOP_K)
    return [{"text": chunks[i], **meta[i]} for i in ids[0] if i != -1]


def generate_answer(query: str, docs: list, client) -> str:
    context = "\n\n---\n\n".join(
        f"[{d['source']} | Page {d['page']}]\n{d['text']}" for d in docs
    )
    prompt = (
        "You are a precise, expert research assistant. "
        "Answer using ONLY the context below. Be thorough but concise. "
        "Use markdown formatting where helpful. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    )
    response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
    return response.text


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1>⚙ Engine Core</h1>", unsafe_allow_html=True)

    default_key = ""
    try:
        default_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    api_key = st.text_input(
        "Gemini API Key",
        value=default_key,
        type="password",
        placeholder="AIza…",
        help="Get your free key at aistudio.google.com",
    )

    is_ready = "index" in st.session_state
    st.markdown(
        f'<div class="status-badge {"badge-online" if is_ready else "badge-offline"}">'
        f'{"● ONLINE" if is_ready else "○ OFFLINE"}</div>',
        unsafe_allow_html=True,
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
            f'<p style="color:#a080ff;">{len(uploaded_files)} file(s) selected</p>',
            unsafe_allow_html=True,
        )

    if st.button("⚡  INITIALIZE ENGINE"):
        if not api_key:
            st.error("Gemini API key required.")
        elif not uploaded_files:
            st.error("Upload at least one PDF first.")
        else:
            try:
                client = get_client(api_key)
                idx, chunks, meta, err = build_index(uploaded_files, client)
                if err:
                    st.error(f"Indexing error: {err}")
                else:
                    st.session_state.update(
                        index=idx, chunks=chunks, meta=meta, api_key=api_key
                    )
                    st.success(f"Indexed {len(chunks)} chunks · {len(uploaded_files)} file(s).")
                    st.rerun()
            except Exception as e:
                err = str(e)
                if "API_KEY" in err.upper() or "401" in err or "403" in err:
                    st.error("Invalid API key — check your Gemini key.")
                else:
                    st.error(f"Error: {err}")

    if is_ready:
        st.divider()
        st.markdown("<h3>Index Stats</h3>", unsafe_allow_html=True)
        meta_stored = st.session_state.meta
        files = list({m["source"] for m in meta_stored})
        st.markdown(
            f"""<div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.chunks)}</div>
                    <div class="metric-label">Chunks</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(files)}</div>
                    <div class="metric-label">Files</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        for f in files:
            st.markdown(f'<div class="source-pill">📄 {f}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<p style="font-size:.65rem;color:#222;text-align:center;letter-spacing:1px;">'
        "GEMINI · FAISS · STREAMLIT</p>",
        unsafe_allow_html=True,
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown("<h1>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:#333;font-size:.82rem;letter-spacing:2px;margin-bottom:1.5rem;">'
    "RAG · SEMANTIC SEARCH · DOCUMENT INTELLIGENCE</p>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if not st.session_state.messages and not is_ready:
    st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;opacity:.5;">
            <div style="font-family:'Orbitron',monospace;font-size:3rem;color:#1a1a2e;">⬡</div>
            <p style="font-family:'Orbitron',monospace;font-size:.72rem;letter-spacing:3px;color:#2a2a3e;margin-top:1rem;">
                AWAITING DOCUMENT INGESTION
            </p>
            <p style="font-size:.82rem;color:#2a2a3e;margin-top:.5rem;">
                Upload PDFs in the sidebar and initialise the engine.
            </p>
        </div>
    """, unsafe_allow_html=True)

if prompt := st.chat_input("Query the knowledge base…"):
    if not api_key:
        st.error("Enter your Gemini API Key in the sidebar.")
    elif "index" not in st.session_state:
        st.error("Upload PDFs and click INITIALIZE ENGINE first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing…"):
                try:
                    client = get_client(st.session_state.api_key)
                    docs = retrieve(
                        prompt,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.meta,
                        client,
                    )
                    ans = generate_answer(prompt, docs, client)
                    seen, src_html = set(), ""
                    for d in docs:
                        label = f"{d['source']} p.{d['page']}"
                        if label not in seen:
                            seen.add(label)
                            src_html += f'<span class="source-pill">📄 {label}</span>'
                    full = (
                        ans
                        + f'\n\n<div style="margin-top:12px;">'
                          f'<span style="font-size:.7rem;color:#5a5f72;letter-spacing:1px;'
                          f'text-transform:uppercase;">Sources</span><br>{src_html}</div>'
                    )
                    st.markdown(full, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    err = str(e)
                    if "401" in err or "403" in err or "API_KEY" in err.upper():
                        st.error("Invalid API key.")
                    elif "429" in err or "quota" in err.lower():
                        st.error("Rate limit — wait a moment and retry.")
                    elif "404" in err:
                        st.error(f"Model not found: {err}")
                    else:
                        st.error(f"Error: {err}")

if st.session_state.get("messages"):
    if st.button("Clear Chat", key="clear"):
        st.session_state.messages = []
        st.rerun()
