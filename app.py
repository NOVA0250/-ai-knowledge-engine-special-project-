import streamlit as st
import tempfile, os
import numpy as np
import pypdf

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="⚡ Neural Knowledge Engine", layout="wide")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6

# ─────────────────────────────────────────────────────────
# PDF PROCESSING
# ─────────────────────────────────────────────────────────
def extract_chunks(file_bytes, filename):
    chunks, meta = [], []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        tmp = f.name

    try:
        reader = pypdf.PdfReader(tmp)
        for page_num, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            text = " ".join(text.split())

            start = 0
            while start < len(text):
                chunk = text[start:start+CHUNK_SIZE]
                if len(chunk) > 80:
                    chunks.append(chunk)
                    meta.append({"source": filename, "page": page_num + 1})
                start += CHUNK_SIZE - CHUNK_OVERLAP

    finally:
        os.remove(tmp)

    return chunks, meta

# ─────────────────────────────────────────────────────────
# HYBRID INDEX (BM25 + TF-IDF)
# ─────────────────────────────────────────────────────────
def build_index(files):
    chunks, meta = [], []

    for f in files:
        c, m = extract_chunks(f.getvalue(), f.name)
        chunks.extend(c)
        meta.extend(m)

    if not chunks:
        return None

    # BM25
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)

    # TF-IDF (semantic fallback)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    return {
        "chunks": chunks,
        "meta": meta,
        "bm25": bm25,
        "vectorizer": vectorizer,
        "tfidf": tfidf_matrix
    }

# ─────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────
def retrieve(query, store):
    chunks = store["chunks"]

    # BM25
    tokenized_query = query.lower().split()
    bm25_scores = store["bm25"].get_scores(tokenized_query)

    # TF-IDF
    q_vec = store["vectorizer"].transform([query])
    tfidf_scores = (store["tfidf"] @ q_vec.T).toarray().ravel()

    # Hybrid score
    alpha = 0.6
    beta = 0.4

    final_scores = alpha * tfidf_scores + beta * bm25_scores

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

# ─────────────────────────────────────────────────────────
# CONTEXT REFINEMENT
# ─────────────────────────────────────────────────────────
def refine_context(query, docs):
    qwords = set(query.lower().split())
    scored = []

    for d in docs:
        for sent in d["text"].split("."):
            s = sent.strip()
            if len(s) < 25:
                continue
            score = sum(1 for w in qwords if w in s.lower())
            if score:
                scored.append((score, s))

    scored.sort(reverse=True)

    if scored:
        return ". ".join([s for _, s in scored[:10]])

    return "\n\n".join(d["text"][:300] for d in docs[:3])

# ─────────────────────────────────────────────────────────
# LOCAL ANSWER (NO LLM → ALWAYS WORKS)
# ─────────────────────────────────────────────────────────
def local_answer(query, docs):
    context = refine_context(query, docs)
    return f"**Answer (from document):**\n\n{context}"

# ─────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────

st.title("⚡ Neural Knowledge Engine (Stable Edition)")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Initialize Engine"):
    if not files:
        st.error("Upload PDFs first.")
    else:
        with st.spinner("Indexing..."):
            store = build_index(files)
            if not store:
                st.error("No readable text found.")
            else:
                st.session_state.store = store
                st.success("Engine Ready 🚀")

# ─────────────────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────────────────

if "store" in st.session_state:
    query = st.text_input("Ask anything from your PDFs")

    if query:
        with st.spinner("Searching..."):
            docs = retrieve(query, st.session_state.store)
            answer = local_answer(query, docs)

        st.markdown(answer)

        st.markdown("### 📄 Sources")
        for d in docs:
            st.write(f"{d['source']} (Page {d['page']})")
