import streamlit as st
import os
import tempfile
import numpy as np
import faiss
import pypdf
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional OpenAI
try:
    from openai import OpenAI
    USE_OPENAI = True
except:
    USE_OPENAI = False


# ── CONFIG ─────────────────────────────────────────────────────
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 20
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100

st.set_page_config(page_title="Neural Knowledge Engine", layout="wide")

st.markdown("""
<style>
body {background:#050507;color:#d1d5db;}
.stButton > button:hover {box-shadow:0 0 12px cyan;}
[data-testid="stChatMessage"]:hover {box-shadow:0 0 12px cyan;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>⚡ Neural Knowledge Engine</h1>", unsafe_allow_html=True)


# ── LOAD MODEL ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embed_model = load_model()

if USE_OPENAI:
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except:
        USE_OPENAI = False


# ── PDF PROCESSING ─────────────────────────────────────────────
def extract_chunks(file_bytes, filename):
    chunks, meta = [], []

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(file_bytes)
        tmp = f.name

    reader = pypdf.PdfReader(tmp)

    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        text = " ".join(text.split())

        start = 0
        while start < len(text):
            piece = text[start:start+CHUNK_SIZE]

            if len(piece) > 60:
                chunks.append(piece)
                meta.append({"source": filename, "page": i+1})

            start += CHUNK_SIZE - CHUNK_OVERLAP

    os.remove(tmp)
    return chunks, meta


def build_index(files):
    all_chunks, all_meta = [], []

    for f in files:
        c, m = extract_chunks(f.getvalue(), f.name)
        all_chunks.extend(c)
        all_meta.extend(m)

    embeddings = embed_model.encode(all_chunks).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    vectorizer = TfidfVectorizer().fit(all_chunks)
    tfidf_matrix = vectorizer.transform(all_chunks)

    return index, all_chunks, all_meta, vectorizer, tfidf_matrix


# ── RETRIEVE + RERANK ──────────────────────────────────────────
def retrieve(query, index, chunks, meta, vectorizer, tfidf_matrix):
    q_vec = embed_model.encode([query]).astype("float32")
    faiss.normalize_L2(q_vec)

    _, sem_ids = index.search(q_vec, TOP_K)

    q_tfidf = vectorizer.transform([query])
    scores = (tfidf_matrix @ q_tfidf.T).toarray().ravel()
    kw_ids = np.argsort(scores)[::-1][:TOP_K]

    combined = list(set(sem_ids[0]) | set(kw_ids))
    docs = [{"text": chunks[i], **meta[i]} for i in combined if i != -1]

    return docs


def rerank(query, docs):
    q_vec = embed_model.encode([query])[0]

    scored = []
    for d in docs:
        d_vec = embed_model.encode([d["text"]])[0]
        score = np.dot(q_vec, d_vec)
        scored.append((score, d))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [d for _, d in scored[:5]]


# ── SMART LOCAL ANSWER ─────────────────────────────────────────
def local_answer(query, docs):
    query_words = set(query.lower().split())
    scored_sentences = []

    for d in docs:
        for s in d["text"].split("."):
            score = sum(1 for w in query_words if w in s.lower())
            if score > 0:
                scored_sentences.append((score, s.strip()))

    scored_sentences.sort(reverse=True)
    selected = [s for _, s in scored_sentences[:5]]

    return "📌 Answer:\n\n" + " ".join(selected) if selected else "⚠️ No answer found."


# ── OPENAI ANSWER ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def openai_answer(prompt):
    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return res.choices[0].message.content


def generate_answer(query, docs):
    docs = rerank(query, docs)
    context = "\n\n".join(d["text"] for d in docs[:3])

    prompt = f"""
Answer directly in 2-4 lines.

Context:
{context}

Question:
{query}
"""

    if USE_OPENAI:
        try:
            return openai_answer(prompt)
        except:
            return local_answer(query, docs)

    return local_answer(query, docs)


# ── DATA MODE (CSV INTELLIGENCE) ───────────────────────────────
def data_answer(query, df):
    q = query.lower()

    try:
        if "average salary" in q:
            return f"Average salary: {df['salary'].mean():.2f}"

        elif "max salary" in q:
            return f"Max salary: {df['salary'].max()}"

        elif "missing" in q:
            return str(df.isnull().sum())

        elif "duplicate" in q:
            return str(df[df.duplicated()])

        elif "top performer" in q:
            return str(df.sort_values("performance_score", ascending=False).head(1))

        elif "department" in q:
            return str(df.groupby("department")["salary"].mean())

        else:
            return str(df.head())

    except Exception as e:
        return f"⚠️ Error: {e}"


# ── SIDEBAR ────────────────────────────────────────────────────
uploaded = st.sidebar.file_uploader(
    "Upload Files",
    type=["pdf", "csv"],
    accept_multiple_files=True
)

if st.sidebar.button("Initialize"):
    pdfs, csvs = [], []

    for f in uploaded:
        if f.name.endswith(".pdf"):
            pdfs.append(f)
        elif f.name.endswith(".csv"):
            csvs.append(f)

    if pdfs:
        idx, chunks, meta, vec, tfidf = build_index(pdfs)
        st.session_state.index = idx
        st.session_state.chunks = chunks
        st.session_state.meta = meta
        st.session_state.vectorizer = vec
        st.session_state.tfidf = tfidf

    if csvs:
        df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
        st.session_state.df = df

    st.sidebar.success("✅ Ready")


# ── CHAT ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("⚡ Thinking..."):

            if "df" in st.session_state:
                ans = data_answer(prompt, st.session_state.df)

            elif "index" in st.session_state:
                docs = retrieve(
                    prompt,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.meta,
                    st.session_state.vectorizer,
                    st.session_state.tfidf
                )
                ans = generate_answer(prompt, docs)

            else:
                ans = "⚠️ Upload and initialize files first."

            st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
