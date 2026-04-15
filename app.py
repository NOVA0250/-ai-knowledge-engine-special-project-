# ─────────────────────────────────────────────────────────────────────────────
# ⚡ Neural Knowledge Engine (HYBRID: FAISS + BM25)
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import tempfile, os, time
import numpy as np
import pandas as pd
import pypdf

# ── HYBRID SEARCH IMPORTS ──
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Neural Knowledge Engine ⚡", layout="wide")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 8
MAX_CHUNKS = 400

# ── EMBEDDING MODEL ──
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

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
                piece = text[start:start+CHUNK_SIZE].strip()
                if len(piece) > 80:
                    chunks.append(piece)
                    meta.append({"source": filename, "page": page_num + 1})
                start += CHUNK_SIZE - CHUNK_OVERLAP

    finally:
        os.remove(tmp)

    return chunks, meta


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID INDEX (FAISS + BM25)
# ─────────────────────────────────────────────────────────────────────────────

def build_pdf_index(uploaded_files):
    all_chunks, all_meta = [], []

    for uf in uploaded_files:
        c, m = extract_chunks(uf.getvalue(), uf.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    if not all_chunks:
        return None, "No readable text found."

    all_chunks = all_chunks[:MAX_CHUNKS]
    all_meta   = all_meta[:MAX_CHUNKS]

    # BM25
    tokenized = [c.lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized)

    # FAISS
    embeddings = EMBED_MODEL.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return {
        "chunks": all_chunks,
        "meta": all_meta,
        "bm25": bm25,
        "faiss": index,
        "embeddings": embeddings
    }, None


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_pdf(query, store):
    chunks = store["chunks"]

    # BM25
    tokenized_query = query.lower().split()
    bm25_scores = store["bm25"].get_scores(tokenized_query)

    # FAISS
    q_embed = EMBED_MODEL.encode([query])
    q_embed = np.array(q_embed).astype("float32")

    distances, indices = store["faiss"].search(q_embed, TOP_K)

    faiss_scores = np.zeros(len(chunks))
    for rank, idx in enumerate(indices[0]):
        faiss_scores[idx] = 1 / (1 + distances[0][rank])

    # Hybrid score
    alpha = 0.65
    beta = 0.35

    final_scores = alpha * faiss_scores + beta * bm25_scores

    top_ids = np.argsort(final_scores)[::-1][:TOP_K]

    results = []
    for i in top_ids:
        if final_scores[i] > 0:
            results.append({
                "text": chunks[i],
                **store["meta"][i],
                "score": float(final_scores[i])
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────

def refine_context(query, docs):
    qwords = set(query.lower().split())
    scored = []

    for d in docs:
        for sent in d["text"].split("."):
            s = sent.strip()
            if len(s) < 30:
                continue
            score = sum(1 for w in qwords if w in s.lower())
            if score:
                scored.append((score, s))

    scored.sort(reverse=True)
    top = [s for _, s in scored[:12]]

    if top:
        return ". ".join(top)

    return "\n\n".join(d["text"][:400] for d in docs[:3])


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE LLM (GROQ / OPENAI COMPATIBLE)
# ─────────────────────────────────────────────────────────────────────────────

def get_llm(api_key):
    from groq import Groq
    return Groq(api_key=api_key)


def llm_answer(llm, prompt):
    resp = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def answer_query(query, store, llm):
    docs = retrieve_pdf(query, store)

    if not docs:
        return "No relevant information found."

    context = refine_context(query, docs)

    prompt = f"""
You are an expert assistant.
Answer ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    try:
        answer = llm_answer(llm, prompt)
    except:
        answer = context[:800]

    return answer, docs


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("⚡ Neural Knowledge Engine (Hybrid RAG)")

api_key = st.text_input("Enter Groq API Key", type="password")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Initialize"):
    if not files or not api_key:
        st.error("Upload files and API key")
    else:
        with st.spinner("Indexing..."):
            store, err = build_pdf_index(files)
            if err:
                st.error(err)
            else:
                st.session_state.store = store
                st.session_state.llm = get_llm(api_key)
                st.success("Ready!")

# Chat
if "store" in st.session_state:
    query = st.text_input("Ask something")

    if query:
        with st.spinner("Thinking..."):
            ans, docs = answer_query(query, st.session_state.store, st.session_state.llm)

        st.write(ans)

        st.markdown("### Sources")
        for d in docs:
            st.write(f"{d['source']} (Page {d['page']})")
