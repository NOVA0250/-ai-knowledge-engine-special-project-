__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# ... the rest of your imports (streamlit, os, etc.) follow below ...
import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
import tempfile

# --- UI CONFIGURATION & NEON THEME ---
st.set_page_config(page_title="NEURAL KNOWLEDGE ENGINE", layout="wide")

def apply_neon_theme():
    st.markdown("""
        <style>
        /* Main Background */
        .stApp {
            background-color: #0a0a0c;
            color: #e0e0e0;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00f2ff !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px #00f2ff55;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #111114;
            border-right: 1px solid #7000ff;
        }

        /* Input Boxes */
        .stTextInput > div > div > input {
            background-color: #1a1a1e;
            color: #00f2ff;
            border: 1px solid #7000ff;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(45deg, #7000ff, #00f2ff);
            color: white;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            width: 100%;
            transition: 0.3s;
        }
        
        .stButton > button:hover {
            box-shadow: 0 0 20px #7000ffaa;
            transform: scale(1.02);
        }

        /* Chat Messages */
        .stChatMessage {
            background-color: #16161a;
            border: 1px solid #333;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        
        /* Neon Accents */
        .css-1offfwp { 
            background-color: #7000ff !important; 
        }
        </style>
    """, unsafe_allow_html=True)

apply_neon_theme()

# --- BACKEND LOGIC ---

def process_documents(uploaded_files, openai_api_key):
    """Processes uploaded PDFs and stores them in a vector database."""
    all_docs = []
    for uploaded_file in uploaded_files:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)
        os.remove(tmp_path)

    # Create Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(documents=all_docs, embedding=embeddings)
    return vectorstore

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ ENGINE CORE")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.divider()
    uploaded_files = st.file_uploader("Ingest Research Papers (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("INITIALIZE ENGINE") and uploaded_files and api_key:
        with st.spinner("Decoding Data Structures..."):
            st.session_state.vectorstore = process_documents(uploaded_files, api_key)
            st.success("Knowledge Base Synced.")

# --- MAIN INTERFACE ---
st.title("⚡ AI KNOWLEDGE ENGINE")
st.caption("Advanced Semantic Retrieval & Synthesis System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Query
if prompt := st.chat_input("Query the knowledge base..."):
    if not api_key:
        st.error("Please enter an OpenAI API Key in the sidebar.")
    elif "vectorstore" not in st.session_state:
        st.error("Please upload and initialize documents first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing response..."):
                llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                
                response = chain({"question": prompt}, return_only_outputs=True)
                
                answer = response['answer']
                sources = response.get('sources', 'N/A')
                
                full_response = f"{answer}\n\n**SOURCES:**\n{sources}"
                st.markdown(full_response)
                
        st.session_state.messages.append({"role": "assistant", "content": full_response})
