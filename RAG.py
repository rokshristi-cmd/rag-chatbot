import os
import re
import numpy as np
import pandas as pd
import torch
import requests
import streamlit as st
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer, util
from groq import Groq
import fitz

# ─ CONFIG ──
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "your_new_key_here")
Q_MODEL   = "openai/gpt-oss-20b"
EMBED_MODEL  = "all-mpnet-base-v2"
CHUNK_SIZE   = 10
MIN_TOKENS   = 30
TOP_K        = 5
PDF_URL      = "https://arxiv.org/pdf/1404.7828"
PDF_PATH     = "ml_paper.pdf"
CSV_PATH     = "chunks_and_embeddings.csv"

def download_pdf(url, path):
    if os.path.exists(path):
        return
    response = requests.get(url)
    with open(path, "wb") as f:
        f.write(response.content)

def read_pdf(path):
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text().replace("\n", " ").strip()
        pages.append({
            "page_number": page_num,
            "text": text,
            "token_count": len(text) / 4,
        })
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
            chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": text,
                "token_count": len(text) / 4,
            })
    chunks = [c for c in chunks if c["token_count"] > MIN_TOKENS]
    return chunks

def save_embeddings(chunks, embeddings):
    df = pd.DataFrame(chunks)
    df["embedding"] = [e.cpu().numpy().tolist() for e in embeddings]
    df.to_csv(CSV_PATH, index=False)

def load_embeddings():
    df = pd.read_csv(CSV_PATH)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(eval(x), dtype=np.float32)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = torch.tensor(
        np.stack(df["embedding"].tolist()), dtype=torch.float32
    ).to(device)
    chunks = df.drop(columns=["embedding"]).to_dict(orient="records")
    return chunks, embeddings

def retrieve(query, embed_model, embeddings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_vec = embed_model.encode(query, convert_to_tensor=True).to(device)
    scores = util.dot_score(query_vec, embeddings)[0]
    _, top_indices = torch.topk(scores, k=TOP_K)
    return top_indices

def build_prompt(query, context_chunks, chat_history):
    history_text = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
    context = "\n\n".join([c["sentence_chunk"] for c in context_chunks])
    return f"""You are a helpful assistant answering questions about a document.
Answer ONLY using the context below.
If the answer is not in the context say: I don't have that in the document.
Do not mention chunks pages or retrieval. Just answer naturally.

PREVIOUS CONVERSATION:
{history_text}

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def get_answer(query, embed_model, embeddings, chunks, chat_history):
    indices = retrieve(query, embed_model, embeddings)
    context_chunks = [chunks[i] for i in indices]
    prompt = build_prompt(query, context_chunks, chat_history)
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content

@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_rag_system():
    st.write("Step 1: Downloading PDF...")
    download_pdf(PDF_URL, PDF_PATH)
    st.write("Step 2: Loading embedding model...")
    device = "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    st.write("Step 3: Checking CSV...")
    if not os.path.exists(CSV_PATH):
        st.write("Step 4: Building chunks...")
        pages = read_pdf(PDF_PATH)
        pages = add_sentences(pages)
        chunks = build_chunks(pages)
        st.write("Step 5: Embedding...")
        embeddings = embed_model.encode(
            [c["sentence_chunk"] for c in chunks],
            batch_size=16,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        save_embeddings(chunks, embeddings)
    st.write("Step 6: Loading embeddings...")
    chunks, embeddings = load_embeddings()
    st.write("Done!")
    return embed_model, embeddings, chunks

# ── UI ──
st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")
st.title("🤖 shristi~RAG" )
st.caption("Powered by RAG + Groq.")

with st.spinner("Loading knowledge base..."):
    embed_model, embeddings, chunks = load_rag_system()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I've read the Documents. Ask me anything about it 👋"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about the document...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = get_answer(
                query=user_input,
                embed_model=embed_model,
                embeddings=embeddings,
                chunks=chunks,
                chat_history=st.session_state.messages[:-1]
            )
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

if len(st.session_state.messages) > 1:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared! Ask me anything 👋"}
        ]
        st.rerun()