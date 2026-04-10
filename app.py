# --- 1. SQLITE FIX (CRITICAL: MUST BE THE FIRST CODE IN THE FILE) ---
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # This fallback is for local environments where pysqlite3 isn't needed
    pass

import streamlit as st
import os
import tempfile

# --- 2. MODERN LANGCHAIN IMPORTS ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 3. UI CONFIGURATION & NEON THEME ---
st.set_page_config(page_title="NEURAL KNOWLEDGE ENGINE", layout="wide")

def apply_neon_theme():
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
        .stTextInput > div > div > input { 
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

apply_neon_theme()

# --- 4. BACKEND LOGIC ---
def process_documents(uploaded_files, openai_api_key):
    """Processes uploaded PDFs and creates a vector store."""
    all_docs = []
    try:
        for uploaded_file in uploaded_files:
            # Create a secure temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                # Metadata extraction for sources
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                all_docs.extend(splits)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Using langchain_chroma for stability
        vectorstore = Chroma.from_documents(
            documents=all_docs, 
            embedding=embeddings,
            collection_name="neural_engine_data"
        )
        return vectorstore
    except Exception as e:
        st.error(f"Initialization Error: {str(e)}")
        return None

# --- 5. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("⚙️ ENGINE CORE")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.divider()
    uploaded_files = st.file_uploader("Ingest Research Papers (PDF)", type="pdf", accept_multiple_files=True)
    
    if st.button("INITIALIZE ENGINE"):
        if not api_key:
            st.error("Missing API Key.")
        elif not uploaded_files:
            st.error("No Documents Provided.")
        else:
            with st.spinner("Decoding Data Structures..."):
                vs = process_documents(uploaded_files, api_key)
                if vs:
                    st.session_state.vectorstore = vs
                    st.success("Knowledge Base Synced.")

# --- 6. MAIN INTERFACE ---
st.title("⚡ AI KNOWLEDGE ENGINE")
st.caption("Advanced Semantic Retrieval & Synthesis System")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Query the knowledge base..."):
    if not api_key:
        st.error("Enter OpenAI API Key in sidebar.")
    elif "vectorstore" not in st.session_state:
        st.error("Please initialize documents first.")
    else:
        # Add user query to UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Synthesizing..."):
                try:
                    # Initialize LLM
                    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
                    
                    # Modern Chain Setup (LCEL Style)
                    system_prompt = (
                        "You are a technical research assistant. Answer the question based ONLY on the context provided below. "
                        "If the answer is not in the context, say you don't know. "
                        "\n\n"
                        "Context: {context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # 1. Create the chain that combines documents into the prompt
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    # 2. Create the retrieval chain
                    rag_chain = create_retrieval_chain(
                        st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}), 
                        question_answer_chain
                    )
                    
                    # Execute
                    response = rag_chain.invoke({"input": prompt})
                    
                    # Format Response
                    answer = response["answer"]
                    # Extract unique source names from document metadata
                    source_list = list(set([doc.metadata.get("source", "Unknown") for doc in response["context"]]))
                    source_text = "\n".join([f"- {s}" for s in source_list])
                    
                    full_response = f"{answer}\n\n**SOURCES:**\n{source_text}"
                    st.markdown(full_response)
                    
                    # Store history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Synthesis Failed: {str(e)}")
