import streamlit as st
import os
import tempfile

import faiss
import numpy as np
import pypdf

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NEURAL KNOWLEDGE ENGINE", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0a0a0c; color: #e0e0e0; }
h1, h2, h3 {
    color: #00f2ff !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    text-shadow: 0 0 10px #00f2ff55;
}
[data-testid="stSidebar"] {
    background-color: #111114;
    border-right: 1px solid #7000ff;
}
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background-color: #1a1a1e;
    color: #00f2ff;
    border: 1px solid #7000ff;
}
.stButton > button {
    background: linear-gradient(45deg, #7000ff, #00f2ff);
    color: white; border: none; border-radius: 5px;
    font-weight: bold; width: 100%; transition: 0.3s;
}
.stButton > button:hover {
    box-shadow: 0 0 20px #7000ffaa;
    transform: scale(1.02);
}
.stChatMessage {
    background-color: #16161a;
    border: 1px solid #333;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── PROVIDER CONFIG ───────────────────────────────────────────────────────────

PROVIDERS = {
    "OpenAI":        {"models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"], "has_embed": True},
    "Anthropic":     {"models": ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"], "has_embed": False},
    "Google Gemini": {"models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"], "has_embed": True},
    "Groq":          {"models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"], "has_embed": False},
}

NEEDS_EMBED_KEY = {"Anthropic", "Groq"}


# ── EMBEDDINGS ────────────────────────────────────────────────────────────────

def embed_openai(texts, api_key):
    import openai
    client = openai.OpenAI(api_key=api_key)
    clean = [t.replace("\n", " ").strip() or " " for t in texts]
    resp = client.embeddings.create(model="text-embedding-3-small", input=clean)
    return np.array([d.embedding for d in resp.data], dtype="float32")


def embed_google(texts, api_key):
    """Embed texts one-by-one using Google's text-embedding-004 model."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    vecs = []
    for t in texts:
        clean = t.replace("\n", " ").strip() or " "
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=clean,
            task_type="retrieval_document",
        )
        vecs.append(result["embedding"])
    return np.array(vecs, dtype="float32")


def get_embeddings(texts, provider, api_key, embed_key=None):
    if provider in NEEDS_EMBED_KEY:
        return embed_openai(texts, embed_key)
    elif provider == "Google Gemini":
        return embed_google(texts, api_key)
    else:
        return embed_openai(texts, api_key)


# ── CHAT ──────────────────────────────────────────────────────────────────────

def chat_openai(query, ctx, api_key, model):
    import openai
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer using ONLY the context below. "
             "If the answer isn't there, say so.\n\nCONTEXT:\n" + ctx},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content


def chat_anthropic(query, ctx, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=1024, temperature=0,
        system="Answer using ONLY the context below. If the answer isn't there, say so.\n\nCONTEXT:\n" + ctx,
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text


def chat_google(query, ctx, api_key, model):
    """Use Gemini for chat — context injected directly into the prompt."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model_name=model)
    # Inject context directly into the user prompt (avoids system_instruction issues)
    full_prompt = (
        "You are a precise research assistant. "
        "Answer using ONLY the context below. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION:\n{query}"
    )
    resp = m.generate_content(full_prompt)
    return resp.text


def chat_groq(query, ctx, api_key, model):
    from groq import Groq
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[
            {"role": "system", "content":
             "Answer using ONLY the context below. "
             "If the answer isn't there, say so.\n\nCONTEXT:\n" + ctx},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content


def get_answer(query, docs, provider, api_key, model):
    ctx = "\n\n---\n\n".join(
        f"[{d['source']} | p.{d['page']}]\n{d['text']}" for d in docs
    )
    dispatch = {
        "OpenAI":        chat_openai,
        "Anthropic":     chat_anthropic,
        "Google Gemini": chat_google,
        "Groq":          chat_groq,
    }
    return dispatch[provider](query, ctx, api_key, model)


# ── PDF + INDEXING ────────────────────────────────────────────────────────────

def extract_chunks(file_bytes, filename):
    chunks, meta = [], []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name
    try:
        reader = pypdf.PdfReader(tmp)
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            size, overlap = 800, 150
            start = 0
            while start < len(text):
                piece = text[start:start + size].strip()
                if piece and len(piece) <= 6000:
                    chunks.append(piece)
                    meta.append({"source": filename, "page": i + 1})
                start += size - overlap
    finally:
        os.remove(tmp)
    return chunks, meta


def build_index(uploaded_files, provider, api_key, embed_key=None):
    all_chunks, all_meta = [], []
    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        st.error("No text could be extracted from the PDFs.")
        return None, None, None

    st.info(f"Embedding {len(all_chunks)} chunks via {provider}…")
    vecs = []
    # Google embeds one-at-a-time inside embed_google, so batch=100 is fine for others
    batch_size = 100
    progress = st.progress(0)
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        vecs.append(get_embeddings(batch, provider, api_key, embed_key))
        progress.progress(min((i + batch_size) / len(all_chunks), 1.0))
    progress.empty()

    matrix = np.vstack(vecs)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, all_chunks, all_meta


def retrieve(query, index, chunks, meta, provider, api_key, embed_key=None, k=5):
    q = get_embeddings([query], provider, api_key, embed_key)
    faiss.normalize_L2(q)
    _, ids = index.search(q, k)
    return [{"text": chunks[i], **meta[i]} for i in ids[0] if i != -1]


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ENGINE CORE")

    provider = st.selectbox("AI Provider", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider]["models"])

    def get_secret(key):
        try:
            return st.secrets.get(key, "")
        except Exception:
            return ""

    secret_key_map = {
        "OpenAI":        "OPENAI_API_KEY",
        "Anthropic":     "ANTHROPIC_API_KEY",
        "Google Gemini": "GOOGLE_API_KEY",
        "Groq":          "GROQ_API_KEY",
    }

    api_key = st.text_input(
        f"{provider} API Key",
        value=get_secret(secret_key_map[provider]),
        type="password",
    )

    embed_key = None
    if provider in NEEDS_EMBED_KEY:
        st.caption("⚠️ This provider has no embedding API. An OpenAI key is needed for document indexing.")
        embed_key = st.text_input(
            "OpenAI API Key (for embeddings)",
            value=get_secret("OPENAI_API_KEY"),
            type="password",
        )

    st.divider()
    uploaded_files = st.file_uploader(
        "Ingest Research Papers (PDF)", type="pdf", accept_multiple_files=True
    )

    if st.button("INITIALIZE ENGINE"):
        missing = []
        if not api_key:
            missing.append(f"{provider} API Key")
        if provider in NEEDS_EMBED_KEY and not embed_key:
            missing.append("OpenAI API Key (embeddings)")
        if not uploaded_files:
            missing.append("at least one PDF")

        if missing:
            st.error("Missing: " + ", ".join(missing))
        else:
            try:
                idx, chunks, meta = build_index(uploaded_files, provider, api_key, embed_key)
                if idx is not None:
                    st.session_state.update(
                        index=idx, chunks=chunks, meta=meta,
                        api_key=api_key, embed_key=embed_key,
                        provider=provider, model=model,
                    )
                    st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
            except Exception as e:
                st.error(f"Indexing error: {e}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("⚡ AI KNOWLEDGE ENGINE")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Query the documents…"):
    if not api_key:
        st.error("Enter your API Key in the sidebar.")
    elif "index" not in st.session_state:
        st.error("Upload PDFs and click INITIALIZE ENGINE first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing…"):
                try:
                    docs = retrieve(
                        prompt,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.meta,
                        st.session_state.provider,
                        st.session_state.api_key,
                        st.session_state.embed_key,
                    )
                    ans = get_answer(
                        prompt, docs,
                        st.session_state.provider,
                        st.session_state.api_key,
                        st.session_state.model,
                    )
                    srcs = list({f"{d['source']} (p.{d['page']})" for d in docs})
                    full = ans + "\n\n**SOURCES:**\n" + "\n".join(f"- {s}" for s in srcs)
                    st.markdown(full)
                    st.session_state.messages.append({"role": "assistant", "content": full})
                except Exception as e:
                    st.error(f"Error: {e}")
