# ============================================================
#  NEXUS — Advanced RAG Chatbot
#  Upload ANY PDF → chat with it instantly
#  Groq-powered streaming responses
#
#  Run: streamlit run nexus.py
#  Install: pip install streamlit groq pymupdf sentence-transformers torch spacy pandas numpy requests
#  Spacy model: python -m spacy download en_core_web_sm
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import torch
import streamlit as st
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import fitz
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-20b"
EMBED_MODEL  = "all-mpnet-base-v2"
CHUNK_SIZE   = 10
MIN_TOKENS   = 30
TOP_K        = 5


# ─────────────────────────────────────────────
# RAG PIPELINE
# ─────────────────────────────────────────────

def read_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().replace("\n", " ").strip()
        if text:
            pages.append({"page_number": page_num + 1, "text": text})
    return pages

def add_sentences(pages):
    nlp = English()
    nlp.add_pipe("sentencizer")
    for item in pages:
        item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
    return pages

def build_chunks(pages):
    chunks = []
    for item in pages:
        sentences = item["sentences"]
        groups = [sentences[i:i+CHUNK_SIZE] for i in range(0, len(sentences), CHUNK_SIZE)]
        for group in groups:
            text = " ".join(group).strip()
            text = re.sub(r'\.([A-Z])', r'. \1', text)
            if len(text) / 4 > MIN_TOKENS:
                chunks.append({
                    "page_number": item["page_number"],
                    "sentence_chunk": text,
                })
    return chunks

@st.cache_resource(show_spinner=False)
def get_embed_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)

def embed_chunks(chunks, embed_model):
    texts = [c["sentence_chunk"] for c in chunks]
    embeddings = embed_model.encode(texts, batch_size=16, convert_to_tensor=True, show_progress_bar=False)
    return embeddings

def retrieve(query, embed_model, embeddings, k=TOP_K):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_vec = embed_model.encode(query, convert_to_tensor=True).to(device)
    scores = util.dot_score(query_vec, embeddings)[0]
    top_scores, top_indices = torch.topk(scores, k=min(k, len(scores)))
    return top_scores, top_indices

def build_prompt(query, context_chunks, chat_history):
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    context = "\n\n".join([
        f"[Page {c['page_number']}] {c['sentence_chunk']}"
        for c in context_chunks
    ])

    return f"""You are NEXUS, an intelligent document assistant. Answer questions based ONLY on the document context provided.
Be clear, concise, and helpful. If the answer isn't in the context, say so honestly.
Never mention "chunks", "embeddings", or technical retrieval details — just answer naturally.
At the end of your answer, add a subtle source reference like: (Source: Page X)

CONVERSATION HISTORY:
{history_text}

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def stream_answer(query, embed_model, embeddings, chunks, chat_history):
    scores, indices = retrieve(query, embed_model, embeddings)
    context_chunks = [chunks[i] for i in indices]
    pages_used = list(set([c["page_number"] for c in context_chunks]))

    prompt = build_prompt(query, context_chunks, chat_history)
    client = Groq(api_key=GROQ_API_KEY)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="NEXUS — Document AI",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

* { box-sizing: border-box; }

.stApp {
    background: #070b14;
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Hero header ── */
.nexus-header {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
.nexus-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #60a5fa 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s linear infinite;
    letter-spacing: -1px;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.nexus-tagline {
    color: #4b5563;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    font-weight: 500;
}

/* ── Upload zone ── */
.upload-zone {
    background: #0f1525;
    border: 1.5px dashed #1e3a5f;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #3b82f6; }

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0d2137;
    border: 1px solid #1e3a5f;
    color: #60a5fa;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 4px 12px;
    border-radius: 99px;
    margin: 0.5rem 0 1rem 0;
    letter-spacing: 0.05em;
}
.status-dot {
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Chat messages ── */
.stChatMessage {
    background: transparent !important;
    border: none !important;
}
[data-testid="stChatMessageContent"] {
    font-size: 0.95rem;
    line-height: 1.7;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: #eff6ff !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.75rem 1rem !important;
    border: none !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
    background: #111827 !important;
    color: #e5e7eb !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 0.75rem 1rem !important;
    border: 1px solid #1f2937 !important;
}

/* ── Chat input ── */
.stChatInputContainer {
    background: #0f1525 !important;
    border: 1.5px solid #1e3a5f !important;
    border-radius: 16px !important;
    padding: 0.25rem !important;
    transition: border-color 0.2s !important;
}
.stChatInputContainer:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}
.stChatInputContainer textarea {
    background: transparent !important;
    color: #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
}
.stChatInputContainer textarea::placeholder { color: #374151 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1525 !important;
    border: 1.5px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label { color: #9ca3af !important; }

/* ── Buttons ── */
.stButton button {
    background: #111827 !important;
    color: #9ca3af !important;
    border: 1px solid #1f2937 !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background: #1f2937 !important;
    color: #e5e7eb !important;
    border-color: #374151 !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #3b82f6, #a78bfa) !important;
    border-radius: 99px !important;
}

/* ── Divider ── */
hr { border-color: #111827 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── PDF info card ── */
.pdf-card {
    background: #0f1525;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 0.5rem 0;
}
.pdf-icon { font-size: 1.5rem; }
.pdf-name { color: #e5e7eb; font-size: 0.9rem; font-weight: 500; }
.pdf-meta { color: #4b5563; font-size: 0.75rem; margin-top: 2px; }

/* ── Suggestions ── */
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.suggestion-chip {
    background: #0f1525;
    border: 1px solid #1e3a5f;
    color: #60a5fa;
    font-size: 0.8rem;
    padding: 0.35rem 0.75rem;
    border-radius: 99px;
    cursor: pointer;
    transition: all 0.2s;
}
.suggestion-chip:hover {
    background: #1d4ed8;
    border-color: #3b82f6;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.markdown("""
<div class="nexus-header">
    <div class="nexus-logo">⚡ NEXUS</div>
    <div class="nexus-tagline">Document Intelligence · Powered by Groq</div>
</div>
""", unsafe_allow_html=True)


# ── Load embedding model once ──
# ── Load embedding model once ──
embed_model = get_embed_model()       


# ── Session state init ──
if "messages"   not in st.session_state: st.session_state.messages   = []
if "chunks"     not in st.session_state: st.session_state.chunks     = None
if "embeddings" not in st.session_state: st.session_state.embeddings = None
if "pdf_name"   not in st.session_state: st.session_state.pdf_name   = None
if "pdf_pages"  not in st.session_state: st.session_state.pdf_pages  = 0
if "pdf_chunks" not in st.session_state: st.session_state.pdf_chunks = 0


# ── PDF Upload ──
if st.session_state.chunks is None:
    st.markdown("---")
    uploaded = st.file_uploader(
        "Drop any PDF here to begin",
        type=["pdf"],
        label_visibility="visible"
    )

    if uploaded:
        with st.status("Processing your document...", expanded=True) as status:
            st.write("📖 Reading PDF...")
            pages = read_pdf_bytes(uploaded.read())
            st.write(f"✓ Found {len(pages)} pages")

            st.write("✂️ Splitting into sentences...")
            pages = add_sentences(pages)

            st.write("🧩 Building chunks...")
            chunks = build_chunks(pages)
            st.write(f"✓ Created {len(chunks)} chunks")

            st.write("🧠 Generating embeddings...")
            progress = st.progress(0)
            embeddings = embed_chunks(chunks, embed_model)
            progress.progress(100)
            st.write("✓ Knowledge base ready!")

            status.update(label="✅ Document processed!", state="complete")

        st.session_state.chunks     = chunks
        st.session_state.embeddings = embeddings
        st.session_state.pdf_name   = uploaded.name
        st.session_state.pdf_pages  = len(pages)
        st.session_state.pdf_chunks = len(chunks)
        st.session_state.messages   = []

        # Welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I've read **{uploaded.name}** — {len(pages)} pages, {len(chunks)} knowledge chunks indexed. Ask me anything about it! 🚀"
        })
        st.rerun()

else:
    # ── Active chat interface ──

    # PDF info bar
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"""
        <div class="pdf-card">
            <div class="pdf-icon">📄</div>
            <div>
                <div class="pdf-name">{st.session_state.pdf_name}</div>
                <div class="pdf-meta">{st.session_state.pdf_pages} pages · {st.session_state.pdf_chunks} chunks indexed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("📂 New PDF"):
            st.session_state.chunks     = None
            st.session_state.embeddings = None
            st.session_state.pdf_name   = None
            st.session_state.messages   = []
            st.rerun()

    # Status badge
    st.markdown('<div class="status-badge"><span class="status-dot"></span>NEXUS READY</div>', unsafe_allow_html=True)

    # Render messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions (only at start)
    if len(st.session_state.messages) == 1:
        st.markdown("**Try asking:**")
        suggestions = [
            "Summarise this document",
            "What are the key findings?",
            "What is the main topic?",
            "List the most important points",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}"):
                    st.session_state._pending_input = s
                    st.rerun()

    # Handle suggestion click
    pending = st.session_state.pop("_pending_input", None)

    # Chat input
    user_input = st.chat_input("Ask anything about your document...") or pending

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Stream assistant response
        with st.chat_message("assistant"):
            full_response = ""
            placeholder = st.empty()
            try:
                for token in stream_answer(
                    query        = user_input,
                    embed_model  = embed_model,
                    embeddings   = st.session_state.embeddings,
                    chunks       = st.session_state.chunks,
                    chat_history = st.session_state.messages[:-1]
                ):
                    full_response += token
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"⚠️ Error: {str(e)}"
                placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Clear chat
    if len(st.session_state.messages) > 1:
        st.markdown("---")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = [{
                "role": "assistant",
                "content": f"Conversation cleared! Still here with **{st.session_state.pdf_name}** loaded. What would you like to know? 👋"
            }]
            st.rerun()