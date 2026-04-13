import streamlit as st
import os
import tempfile
import time

from google import genai
from google.genai import types
import faiss
import numpy as np
import pypdf

# ── CONFIG ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "gemini-embedding-001"   # ✅ FIXED
FALLBACK_MODEL = "gemini-embedding-2-preview"
CHAT_MODEL  = "gemini-2.0-flash"

CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 100
EMBED_BATCH = 150
TOP_K         = 5


# ── CLIENT ────────────────────────────────────────────────────────────────────
def get_client(api_key: str):
    return genai.Client(api_key=api_key)


# ── PDF CHUNKING ──────────────────────────────────────────────────────────────
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
                piece = text[start:start + CHUNK_SIZE].strip()
                if len(piece) > 60:
                    chunks.append(piece)
                    meta.append({
                        "source": filename,
                        "page": page_num + 1
                    })
                start += CHUNK_SIZE - CHUNK_OVERLAP
    finally:
        os.remove(tmp)

    return chunks, meta


# ── EMBEDDING (ROBUST) ────────────────────────────────────────────────────────
def embed_batch(client, texts: list, task: str) -> np.ndarray:
    try:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        return np.array([e.values for e in result.embeddings], dtype="float32")

    except Exception as e:
        print("Primary model failed → switching to fallback:", str(e))

        result = client.models.embed_content(
            model=FALLBACK_MODEL,
            contents=texts,
        )
        return np.array([e.values for e in result.embeddings], dtype="float32")


# ── BUILD INDEX ───────────────────────────────────────────────────────────────
def build_index(uploaded_files, client):
    all_chunks, all_meta = [], []

    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, None, None, "No readable text found."

    progress = st.progress(0, text="Embedding...")

    all_vecs = []
    try:
        for i in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[i:i + EMBED_BATCH]
            vecs = embed_batch(client, batch, "RETRIEVAL_DOCUMENT")
            all_vecs.append(vecs)

            pct = min((i + EMBED_BATCH) / len(all_chunks), 1.0)
            progress.progress(pct)

            time.sleep(0.05)

    except Exception as e:
        progress.empty()
        return None, None, None, str(e)

    progress.empty()

    matrix = np.vstack(all_vecs)
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index, all_chunks, all_meta, None


# ── RETRIEVE ──────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, meta, client):
    q_vec = embed_batch(client, [query], "RETRIEVAL_QUERY")
    faiss.normalize_L2(q_vec)

    _, ids = index.search(q_vec, TOP_K)

    return [
        {"text": chunks[i], **meta[i]}
        for i in ids[0] if i != -1
    ]


# ── GENERATE ANSWER ───────────────────────────────────────────────────────────
def generate_answer(query, docs, client):
    context = "\n\n---\n\n".join(
        f"[{d['source']} | Page {d['page']}]\n{d['text']}"
        for d in docs
    )

    prompt = f"""
You are an expert assistant.

Answer ONLY from the context below.
Be concise and clear.

CONTEXT:
{context}

QUESTION:
{query}
"""

    import time

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=prompt
            )

            # ✅ SAFE RETURN
            if hasattr(response, "text") and response.text:
                return response.text
            else:
                return "⚠️ No response generated."

        except Exception as e:
            err = str(e)

            # 🔥 HANDLE QUOTA ERROR
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
                continue

            # 🔥 HANDLE INVALID KEY / PERMISSION
            elif "401" in err or "403" in err:
                return "❌ Invalid API key or permission issue."

            # 🔥 HANDLE MODEL NOT FOUND
            elif "404" in err:
                return "❌ Model not available. Check model name."

            # 🔥 UNKNOWN ERROR
            else:
                return f"❌ Error: {err}"

    return "⚠️ Quota exceeded. Try again later."


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

st.title("⚡ Neural Knowledge Engine")

api_key = st.sidebar.text_input("Gemini API Key", type="password")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if st.sidebar.button("Initialize"):
    if not api_key or not uploaded_files:
        st.error("Provide API key + PDFs")
    else:
        client = get_client(api_key)
        idx, chunks, meta, err = build_index(uploaded_files, client)

        if err:
            st.error(err)
        else:
            st.session_state.index = idx
            st.session_state.chunks = chunks
            st.session_state.meta = meta
            st.session_state.api_key = api_key
            st.success("Index built successfully 🚀")


# ── CHAT ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something..."):
    if "index" not in st.session_state:
        st.error("Initialize first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            client = get_client(st.session_state.api_key)

            docs = retrieve(
                prompt,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.meta,
                client
            )

            answer = generate_answer(prompt, docs, client)

            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
