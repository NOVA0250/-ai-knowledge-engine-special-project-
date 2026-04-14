import streamlit as st
import tempfile
import os
import time
import numpy as np
import pandas as pd
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer

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
    --cyan:#00f2ff; --violet:#7000ff; --pink:#ff006e;
    --dark:#07070a; --card:#0d0d12; --border:#1a1a2e;
    --text:#c8ccd8; --dim:#5a5f72; --green:#00ff9d;
}

html, body, .stApp {
    background-color: var(--dark) !important;
    color: var(--text) !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,242,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,242,255,.025) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080810 0%, #0c0c18 100%) !important;
    border-right: 1px solid var(--violet) !important;
    box-shadow: 6px 0 40px rgba(112,0,255,.2);
}

/* Headings */
h1 {
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 900 !important;
    font-size: 1.55rem !important;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 4px !important;
    text-transform: uppercase;
    margin-bottom: 0 !important;
}
h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--cyan) !important;
    font-size: .82rem !important;
    letter-spacing: 3px !important;
    text-transform: uppercase;
}
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: .8rem 0 !important; }

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: #080810 !important;
    color: var(--cyan) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .82rem !important;
    transition: all .25s ease;
}
.stTextInput > div > div > input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 14px rgba(0,242,255,.3) !important;
    outline: none !important;
}
.stSelectbox > div > div:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 14px rgba(0,242,255,.2) !important;
}

/* Labels */
label, .stTextInput label, .stFileUploader label,
.stSelectbox label, p {
    color: var(--dim) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: .75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #080810 !important;
    border: 1px dashed rgba(112,0,255,.6) !important;
    border-radius: 8px !important;
    transition: all .3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,242,255,.12) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--violet) 0%, #3a00aa 100%) !important;
    color: #fff !important;
    border: 1px solid var(--violet) !important;
    border-radius: 6px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: .65rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase;
    width: 100%;
    padding: .6rem 1rem !important;
    transition: all .3s ease !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(112,0,255,.7), 0 0 50px rgba(0,242,255,.2) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 1rem 1.2rem !important;
    transition: border-color .3s, box-shadow .3s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(112,0,255,.5) !important;
    box-shadow: 0 0 20px rgba(112,0,255,.12) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #080810 !important;
    color: var(--cyan) !important;
    border: 1px solid var(--violet) !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .9rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,242,255,.25) !important;
}

/* Progress */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    border-radius: 4px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: var(--dark); }
::-webkit-scrollbar-thumb { background: var(--violet); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

/* Custom components */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .7rem;
    letter-spacing: 1.5px;
    margin-top: 6px;
}
.badge-online  { background: rgba(0,255,157,.08); border: 1px solid var(--green); color: var(--green); }
.badge-offline { background: rgba(112,0,255,.08); border: 1px solid var(--violet); color: #8844ff; }
.badge-csv     { background: rgba(0,242,255,.08); border: 1px solid var(--cyan);   color: var(--cyan); }

.source-pill {
    display: inline-block;
    background: rgba(112,0,255,.12);
    border: 1px solid rgba(112,0,255,.35);
    border-radius: 4px;
    padding: 2px 9px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .68rem;
    color: #a080ff;
    margin: 2px;
    transition: all .2s;
}
.source-pill:hover {
    background: rgba(0,242,255,.1);
    border-color: var(--cyan);
    color: var(--cyan);
}

.metric-row { display: flex; gap: 10px; margin: 8px 0; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
    transition: border-color .3s;
}
.metric-card:hover { border-color: rgba(0,242,255,.3); }
.metric-value { font-family: 'Orbitron', monospace; font-size: 1.35rem; font-weight: 700; color: var(--cyan); }
.metric-label { font-size: .62rem; color: var(--dim); letter-spacing: 1px; text-transform: uppercase; margin-top: 2px; }

.mode-tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .65rem;
    letter-spacing: 1px;
    margin-left: 8px;
    vertical-align: middle;
}
.mode-pdf { background: rgba(112,0,255,.15); border: 1px solid var(--violet); color: #a060ff; }
.mode-csv { background: rgba(0,242,255,.1);  border: 1px solid var(--cyan);   color: var(--cyan); }

/* Chat content text styles */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--text) !important;
    font-size: .95rem !important;
    line-height: 1.75 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stChatMessage"] strong { color: var(--cyan) !important; }
[data-testid="stChatMessage"] code {
    background: #0a0a14 !important;
    color: var(--pink) !important;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: .82rem !important;
}
[data-testid="stChatMessage"] pre {
    background: #0a0a14 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150
TOP_K         = 8
MAX_CHUNKS    = 400   # cap for stability


# ── LLM CLIENT FACTORY ────────────────────────────────────────────────────────

def get_llm(provider: str, api_key: str):
    """Return a thin wrapper dict so the rest of the code stays provider-agnostic."""
    if provider == "Gemini":
        from google import genai
        client = genai.Client(api_key=api_key)
        return {"provider": "gemini", "client": client}
    else:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return {"provider": "openai", "client": client}


def llm_complete(llm: dict, prompt: str, max_retries: int = 3) -> str:
    """Send a prompt with automatic retry + exponential backoff on rate limits."""
    last_err = None
    for attempt in range(max_retries):
        try:
            if llm["provider"] == "gemini":
                resp = llm["client"].models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )
                return resp.text
            else:
                resp = llm["client"].chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
        except Exception as e:
            last_err = str(e)
            is_rate = any(k in last_err.lower() for k in
                          ("429", "quota", "rate", "resource_exhausted", "too many"))
            if is_rate and attempt < max_retries - 1:
                wait = 20 * (attempt + 1)   # 20s, 40s
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(last_err)


def local_fallback_answer(query: str, docs: list) -> str:
    """Return a structured answer from raw chunks — no API needed."""
    if not docs:
        return "No relevant information found in the uploaded documents."
    qwords = set(query.lower().split())
    scored = []
    for d in docs:
        for sent in d["text"].split("."):
            s = sent.strip()
            if not s:
                continue
            hit = sum(1 for w in qwords if w in s.lower())
            if hit > 0:
                scored.append((hit, s))
    scored.sort(reverse=True)
    best = [s for _, s in scored[:6]]
    if best:
        body = "\n\n".join(f"• {s}." for s in best)
    else:
        body = "\n\n".join(d["text"][:400] for d in docs[:3])
    return (
        f"**Best matching content** *(local mode — LLM unavailable)*\n\n{body}\n\n"
        f"> ⚠️ API rate limit reached. Showing raw matched content. "
        f"Wait ~60s and retry for an AI-synthesised answer."
    )


def validate_key(provider: str, api_key: str) -> str | None:
    """Quick validation — returns error string or None if OK."""
    try:
        llm = get_llm(provider, api_key)
        llm_complete(llm, "Reply with the single word OK.", max_retries=1)
        return None
    except Exception as e:
        return str(e)


# ── PDF PROCESSING ────────────────────────────────────────────────────────────

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
            text = " ".join(raw.split())
            start = 0
            while start < len(text):
                piece = text[start : start + CHUNK_SIZE].strip()
                if len(piece) > 80:
                    chunks.append(piece)
                    meta.append({"source": filename, "page": page_num + 1})
                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)
    return chunks, meta


def build_pdf_index(uploaded_files):
    """Parse PDFs and build a TF-IDF index — zero embedding API calls."""
    all_chunks, all_meta = [], []
    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, "No readable text extracted from the PDFs."

    # cap for stability
    all_chunks = all_chunks[:MAX_CHUNKS]
    all_meta   = all_meta[:MAX_CHUNKS]

    vectorizer   = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

    return {
        "chunks":     all_chunks,
        "meta":       all_meta,
        "vectorizer": vectorizer,
        "matrix":     tfidf_matrix,
        "type":       "pdf",
    }, None


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def retrieve_pdf(query: str, store: dict) -> list[dict]:
    q_vec  = store["vectorizer"].transform([query])
    scores = (store["matrix"] @ q_vec.T).toarray().ravel()
    top_ids = np.argsort(scores)[::-1][:TOP_K]
    return [{"text": store["chunks"][i], **store["meta"][i], "score": float(scores[i])}
            for i in top_ids if scores[i] > 0]


def refine_context(query: str, docs: list[dict]) -> str:
    """Pull the most query-relevant sentences from the top docs."""
    qwords = set(query.lower().split())
    scored = []
    for d in docs:
        for sent in d["text"].split("."):
            s = sent.strip()
            if not s:
                continue
            hit = sum(1 for w in qwords if w in s.lower())
            if hit:
                scored.append((hit, s))
    scored.sort(reverse=True)
    top = [s for _, s in scored[:10]]
    return ". ".join(top) if top else "\n\n".join(d["text"][:500] for d in docs[:3])


# ── ANSWER GENERATION ─────────────────────────────────────────────────────────

def answer_pdf(query: str, store: dict, llm: dict) -> tuple[str, list[dict]]:
    docs = retrieve_pdf(query, store)
    if not docs:
        return "No relevant information found in the uploaded documents.", []

    context = refine_context(query, docs)
    prompt = (
        "You are a precise, expert research assistant.\n"
        "Answer the question using ONLY the context provided below.\n"
        "Be thorough but concise. Use markdown formatting where helpful.\n"
        "If the answer is not found in the context, clearly state that.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}"
    )
    return llm_complete(llm, prompt), docs


def answer_csv(query: str, df: pd.DataFrame, llm: dict) -> str:
    # Send column names + first 3 rows as sample so the LLM understands structure
    sample = df.head(3).to_string(index=False)
    cols   = ", ".join(df.columns.tolist())

    prompt = (
        "You are a data analyst assistant. The user has a pandas DataFrame called `df`.\n"
        f"Columns: {cols}\n"
        f"Sample rows:\n{sample}\n\n"
        "Write Python/pandas code to answer the user's question.\n"
        "Rules:\n"
        "- Store the final answer in a variable called `result`\n"
        "- `result` must be a string, number, or DataFrame\n"
        "- Output ONLY valid Python code, no explanation, no markdown fences\n\n"
        f"Question: {query}"
    )
    try:
        code = llm_complete(llm, prompt)
        code = code.replace("```python", "").replace("```", "").strip()
        local_vars = {"df": df.copy(), "pd": pd}
        exec(code, {}, local_vars)
        result = local_vars.get("result", "No result computed.")
        if isinstance(result, pd.DataFrame):
            return result.to_markdown(index=False)
        return str(result)
    except Exception as e:
        return f"⚠️ Could not execute generated code: {e}\n\nGenerated code:\n```python\n{code}\n```"


# ── ERROR HELPER ──────────────────────────────────────────────────────────────

def friendly_error(err: str) -> str:
    e = err.lower()
    if "401" in e or "403" in e or "invalid_api_key" in e or "api key" in e:
        return "❌ Invalid API key. Double-check it and try again."
    if "429" in e or "quota" in e or "resource_exhausted" in e or "rate" in e:
        return "⏱ Rate limit hit. Wait 30–60 seconds and try again."
    if "404" in e and "model" in e:
        return f"❌ Model not found. Check your API key has access to the model.\n\n`{err}`"
    if "connection" in e or "timeout" in e:
        return "🌐 Network error. Check your internet connection."
    return f"❌ Error: {err}"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1>⚙ Engine Core</h1>", unsafe_allow_html=True)

    # ── API PROVIDER ──
    st.markdown("<h3>AI Provider</h3>", unsafe_allow_html=True)
    provider = st.selectbox(
        "Provider", ["Gemini", "OpenAI"],
        label_visibility="collapsed",
        key="provider_select",
    )

    # Pull from secrets if available
    secret_key = ""
    try:
        secret_map = {"Gemini": "GEMINI_API_KEY", "OpenAI": "OPENAI_API_KEY"}
        secret_key = st.secrets.get(secret_map[provider], "")
    except Exception:
        pass

    api_key = st.text_input(
        "API Key",
        value=secret_key,
        type="password",
        placeholder="AIza… / sk-…",
        help="Gemini: aistudio.google.com · OpenAI: platform.openai.com",
    )

    hint_map = {
        "Gemini": "Free · gemini-2.0-flash",
        "OpenAI": "Paid · gpt-4o-mini",
    }
    st.markdown(
        f'<p style="color:#444;font-size:.65rem;margin-top:-8px;">{hint_map[provider]}</p>',
        unsafe_allow_html=True,
    )

    # Engine status badge
    is_pdf_ready = "pdf_store" in st.session_state
    is_csv_ready = "csv_df"    in st.session_state
    is_ready     = is_pdf_ready or is_csv_ready

    if is_pdf_ready and is_csv_ready:
        badge = '<div class="status-badge badge-online">● PDF + CSV ONLINE</div>'
    elif is_pdf_ready:
        badge = '<div class="status-badge badge-online">● PDF ONLINE</div>'
    elif is_csv_ready:
        badge = '<div class="status-badge badge-csv">● CSV ONLINE</div>'
    else:
        badge = '<div class="status-badge badge-offline">○ OFFLINE</div>'
    st.markdown(badge, unsafe_allow_html=True)

    st.divider()

    # ── FILE UPLOAD ──
    st.markdown("<h3>Document Ingestion</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        pdfs = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
        csvs = [f for f in uploaded_files if f.name.lower().endswith(".csv")]
        if pdfs:
            st.markdown(f'<p style="color:#a080ff;">📄 {len(pdfs)} PDF(s)</p>', unsafe_allow_html=True)
        if csvs:
            st.markdown(f'<p style="color:var(--cyan);">📊 {len(csvs)} CSV(s)</p>', unsafe_allow_html=True)

    if st.button("⚡  INITIALIZE ENGINE"):
        if not api_key:
            st.error("API key required.")
        elif not uploaded_files:
            st.error("Upload at least one PDF or CSV.")
        else:
            pdfs = [f for f in uploaded_files if f.name.lower().endswith(".pdf")]
            csvs = [f for f in uploaded_files if f.name.lower().endswith(".csv")]

            # PDF indexing
            if pdfs:
                with st.spinner("Indexing PDFs (TF-IDF, no API calls)…"):
                    store, err = build_pdf_index(pdfs)
                if err:
                    st.error(f"PDF Error: {err}")
                else:
                    st.session_state.pdf_store = store
                    st.success(f"PDF: {len(store['chunks'])} chunks indexed.")

            # CSV loading
            if csvs:
                try:
                    df = pd.concat(
                        [pd.read_csv(f) for f in csvs], ignore_index=True
                    )
                    st.session_state.csv_df = df
                    st.success(f"CSV: {len(df):,} rows · {len(df.columns)} columns.")
                except Exception as e:
                    st.error(f"CSV Error: {e}")

            # Validate API key with a tiny test call
            with st.spinner("Validating API key…"):
                err = validate_key(provider, api_key)
            if err:
                st.error(friendly_error(err))
            else:
                st.session_state.api_key  = api_key
                st.session_state.provider = provider
                st.success("API key verified ✓")
                time.sleep(0.5)
                st.rerun()

    # ── RESET ──
    if is_ready:
        if st.button("🗑 Reset Engine"):
            for k in ["pdf_store", "csv_df", "api_key", "provider", "messages"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── STATS ──
    if is_pdf_ready:
        st.divider()
        st.markdown("<h3>PDF Index</h3>", unsafe_allow_html=True)
        store = st.session_state.pdf_store
        files = list({m["source"] for m in store["meta"]})
        st.markdown(
            f"""<div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{len(store['chunks'])}</div>
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

    if is_csv_ready:
        st.divider()
        st.markdown("<h3>CSV Dataset</h3>", unsafe_allow_html=True)
        df = st.session_state.csv_df
        st.markdown(
            f"""<div class="metric-row">
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">Cols</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        with st.expander("Column names"):
            st.code(", ".join(df.columns.tolist()), language=None)

    st.divider()
    prov_label = st.session_state.get("provider", provider)
    st.markdown(
        f'<p style="font-size:.58rem;color:#1c1c2e;text-align:center;letter-spacing:1px;">'
        f'{prov_label.upper()} · TF-IDF · FAISS · STREAMLIT</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Title row
col_t, col_m = st.columns([3, 1])
with col_t:
    st.markdown("<h1>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)
    mode_label = ""
    if is_pdf_ready and is_csv_ready:
        mode_label = '<span class="mode-tag mode-pdf">PDF</span><span class="mode-tag mode-csv">CSV</span>'
    elif is_pdf_ready:
        mode_label = '<span class="mode-tag mode-pdf">PDF MODE</span>'
    elif is_csv_ready:
        mode_label = '<span class="mode-tag mode-csv">CSV MODE</span>'
    st.markdown(
        f'<p style="color:#2a2a3e;font-size:.78rem;letter-spacing:2px;margin-bottom:1rem;">'
        f'RAG · SEMANTIC SEARCH · DATA INTELLIGENCE {mode_label}</p>',
        unsafe_allow_html=True,
    )

# CSV preview toggle
if is_csv_ready:
    with st.expander("📊 Preview dataset"):
        st.dataframe(
            st.session_state.csv_df.head(10),
            use_container_width=True,
        )

# ── CHAT HISTORY ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── EMPTY STATE ──
if not st.session_state.messages and not is_ready:
    st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;opacity:.35;">
            <div style="font-family:'Orbitron',monospace;font-size:3.5rem;
                        background:linear-gradient(135deg,#7000ff,#00f2ff);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;">⬡</div>
            <p style="font-family:'Orbitron',monospace;font-size:.7rem;
                       letter-spacing:4px;color:#1a1a2e;margin-top:1rem;">
                AWAITING DOCUMENT INGESTION
            </p>
            <p style="font-size:.82rem;color:#1a1a2e;margin-top:.4rem;">
                Upload PDFs or CSVs in the sidebar and initialise the engine.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ── CHAT INPUT ──
placeholder = (
    "Ask about your documents or data…"
    if is_ready else
    "Initialize the engine first →"
)

if prompt := st.chat_input(placeholder, disabled=not is_ready):
    if "api_key" not in st.session_state:
        st.error("API key not verified — re-initialize the engine.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing…"):
                try:
                    llm       = get_llm(st.session_state.provider, st.session_state.api_key)
                    docs      = []
                    mode_used = "none"

                    # Route: CSV takes priority if both loaded and query looks data-like
                    data_keywords = {"average","mean","max","min","count","sum","total",
                                     "how many","list","show","top","bottom","filter",
                                     "where","group","column","row","table","dataset"}
                    looks_like_data = any(kw in prompt.lower() for kw in data_keywords)

                    if is_csv_ready and (looks_like_data or not is_pdf_ready):
                        ans = answer_csv(prompt, st.session_state.csv_df, llm)
                        mode_used = "csv"
                    elif is_pdf_ready:
                        ans, docs = answer_pdf(prompt, st.session_state.pdf_store, llm)
                        mode_used = "pdf"
                    else:
                        ans, mode_used = "No data loaded.", "none"

                    # Build source pills for PDF answers
                    src_html = ""
                    if docs:
                        seen = set()
                        for d in docs:
                            label = f"{d['source']} · p.{d['page']}"
                            if label not in seen:
                                seen.add(label)
                                src_html += f'<span class="source-pill">📄 {label}</span>'
                        src_html = (
                            f'<div style="margin-top:14px;">'
                            f'<span style="font-size:.65rem;color:var(--dim);letter-spacing:1.5px;'
                            f'text-transform:uppercase;">Sources</span><br>{src_html}</div>'
                        )

                    full = ans + ("\n\n" + src_html if src_html else "")
                    st.markdown(full, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full})

                except Exception as e:
                    err = str(e)
                    is_rate = any(k in err.lower() for k in
                                  ("429", "quota", "rate", "resource_exhausted", "too many"))
                    if is_rate and mode_used == "pdf" and docs:
                        # Graceful local fallback — no API needed
                        fallback = local_fallback_answer(prompt, docs)
                        st.markdown(fallback)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": fallback}
                        )
                    elif is_rate and mode_used == "pdf":
                        # Retrieve was fine, generation failed — do local anyway
                        docs2 = retrieve_pdf(prompt, st.session_state.pdf_store)
                        fallback = local_fallback_answer(prompt, docs2)
                        st.markdown(fallback)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": fallback}
                        )
                    else:
                        st.error(friendly_error(err))

# ── CLEAR CHAT ──
if st.session_state.get("messages"):
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()
