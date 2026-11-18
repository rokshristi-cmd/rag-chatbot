# NEXUS 📄🤖
A RAG-based document Q&A app. Upload a PDF, ask questions, and get streamed answers with page citations.

🔗 **Live demo:** https://shristi-nexus.streamlit.app

## Features
- PDF upload and parsing
- Streaming responses
- Conversation memory
- Page-number citations
- Dark UI

## Tech stack
Python · Streamlit · PyMuPDF · spaCy · SentenceTransformers · Groq API

## How it works
1. PDF text is extracted and split into sentence chunks
2. Chunks are embedded with SentenceTransformers
3. Your question is embedded and matched to the closest chunks
4. The chunks plus your question go to the LLM, which answers using only that context

## Run locally
git clone https://github.com/rokshristi-cmd/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt
# add GROQ_API_KEY to a .env file
streamlit run Nexus.py
