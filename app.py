import streamlit as st
import os
import tempfile
from pdf_utils import load_and_chunk_pdfs
from embeddings import EmbeddingManager
from retrieval import HybridRetriever
from qa import QASystem

st.set_page_config(
    page_title="NEURAL KNOWLEDGE ENGINE",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 NEURAL KNOWLEDGE ENGINE")
st.markdown("Upload PDFs and ask questions")

# Session state
if "documents" not in st.session_state:
    st.session_state.documents = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# API Key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Add GROQ_API_KEY in secrets")
    st.stop()

# Upload PDFs
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        temp_paths = []

        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                temp_paths.append(tmp.name)

        documents = load_and_chunk_pdfs(temp_paths)

        for p in temp_paths:
            os.unlink(p)

        embedding_manager = EmbeddingManager()
        embedding_manager.build_index(documents)

        retriever = HybridRetriever(embedding_manager, documents)
        qa_system = QASystem(groq_api_key, retriever)

        st.session_state.documents = documents
        st.session_state.retriever = retriever
        st.session_state.qa_system = qa_system
        st.session_state.chat_history = []

# Chat UI
if st.session_state.documents:

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask something...")

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = ""

            for chunk in st.session_state.qa_system.answer_question(question):
                response += chunk
                placeholder.markdown(response + "▌")

            placeholder.markdown(response)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

else:
    st.info("Upload PDFs to begin")
