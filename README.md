📚 Hybrid PDF Q&A System using RAG
An AI-powered PDF Question-Answering System built using Streamlit, combining semantic search (FAISS) and keyword search (BM25) with optional Qdrant vector database and Groq LLM for fast, accurate responses.
---
🚀 Features
📄 Upload and process multiple PDF documents
🔍 Hybrid Retrieval:
Semantic Search using FAISS
Keyword Search using BM25
Combined using Reciprocal Rank Fusion (RRF)
🧠 AI-powered answers using Groq (LLaMA models)
⚡ Fast response generation with streaming support
💾 Optional Qdrant integration for persistent vector storage
🧩 Token-based intelligent text chunking
💬 Chat history support
---
🏗️ Architecture
User Query
↓
Hybrid Retriever (FAISS + BM25)
↓
Context Builder
↓
Groq LLM (LLaMA)
↓
Final Answer (Streamlit UI)
---
📂 Project Structure
hybrid-rag-main/
│
├── app.py # Streamlit frontend
├── embeddings.py # Embedding + FAISS/Qdrant management
├── retrieval.py # Hybrid search (BM25 + semantic)
├── qa.py # Groq-based Q&A system
├── pdf_utils.py # PDF loading + text chunking
├── requirements.txt # Dependencies
└── README.md # Documentation
---
⚙️ Tech Stack
Frontend: Streamlit
LLM: Groq (LLaMA models)
Embeddings: SentenceTransformers
Vector DB: FAISS (default), Qdrant (optional)
Search: BM25 + Semantic Search
PDF Processing: PyPDF2
Tokenization: tiktoken
---
🔑 Setup Instructions
1. Clone the Repository
git clone <github.com/saimukesh23/hybrid-rag/>
cd hybrid-rag-main
---
2. Install Dependencies
pip install -r requirements.txt
---
3. Add API Keys
Create a `.streamlit/secrets.toml` file:
GROQ_API_KEY = "your_groq_api_key"
Optional (for persistent vector DB)
QDRANT_API_KEY = "your_qdrant_api_key"
QDRANT_ENDPOINT = "your_qdrant_url"
---
4. Run the Application
streamlit run app.py
---
🌐 Deployment (Streamlit Cloud)
Push project to GitHub
Go to Streamlit Cloud
Click New App
Connect your repository
Add secrets in App Settings → Secrets:
GROQ_API_KEY = "your_key"
Click Deploy
---
🧠 How It Works
1. PDF Processing
Extracts text using PyPDF2
Cleans and chunks text using token-based splitting
2. Embedding Generation
Uses SentenceTransformers
Stores embeddings in FAISS (or Qdrant)
3. Hybrid Retrieval
Semantic Search → FAISS
Keyword Search → BM25
Combined using RRF ranking
4. Answer Generation
Retrieves top relevant chunks
Sends context + query to Groq
Streams response back to UI
---
🔄 Optional: Qdrant Integration
Enable persistent storage by adding:
QDRANT_API_KEY
QDRANT_ENDPOINT
If not provided, system defaults to:
➡️ In-memory FAISS (no persistence)
---
⚠️ Limitations
Free API limits (Groq usage caps)
Large PDFs may increase processing time
No authentication system (single-user app)
---
🔮 Future Improvements
Multi-user support
Document metadata filtering
UI enhancements (chat-style interface, history panel)
Support for DOCX, TXT files
Caching embeddings for faster reloads
---
🤝 Contribution
Feel free to fork and improve the project!
---
📜 License
This project is open-source and available under the MIT License.
---
👨‍💻 Author
Developed as an AI-powered document intelligence system using hybrid retrieval and LLM integration.
