import streamlit as st
import PyPDF2
import numpy as np
import nltk
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter

# -------------------- NLTK --------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# -------------------- SESSION STATE --------------------
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# -------------------- MODELS --------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"  # smaller = safer on Streamlit Cloud
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=256,
        do_sample=False,
    )

# -------------------- PDF PROCESSING --------------------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    return text.strip(), len(reader.pages)

def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def calculate_statistics(text, page_count):
    words = text.split()
    stop_words = set(stopwords.words("english"))

    filtered = [
        w.lower()
        for w in words
        if w.isalpha() and len(w) > 2 and w.lower() not in stop_words
    ]

    return {
        "page_count": page_count,
        "word_count": len(words),
        "avg_words_per_page": round(len(words) / page_count, 2) if page_count else 0,
        "top_words": Counter(filtered).most_common(10),
    }

# -------------------- VECTOR SEARCH (NO FAISS) --------------------
def create_embeddings(chunks, model):
    return model.encode(chunks, show_progress_bar=False)

def retrieve_chunks(query, model, embeddings, chunks, k=4):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[-k:][::-1]
    return [chunks[i] for i in top_indices]

# -------------------- LLM --------------------
def generate_response(query, context_chunks, llm):
    context = "\n\n".join(context_chunks[:3])
    prompt = f"""
You are a helpful hostel assistant.

Context:
{context}

Question:
{query}
"""

    response = llm(prompt)[0]["generated_text"]
    return response.strip()

def generate_summary(text, llm):
    prompt = f"Summarize this hostel document in 2–3 sentences:\n\n{text[:1500]}"
    return llm(prompt)[0]["generated_text"].strip()

# -------------------- UI --------------------
st.set_page_config("Hostel RAG Chat", "🏨", layout="wide")
st.title("🏨 Hostel Document Chat Assistant")

with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload hostel PDF", type="pdf")

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing..."):
            text, pages = extract_text_from_pdf(uploaded_file)
            st.session_state.full_text = text
            st.session_state.chunks = chunk_text(text)
            st.session_state.stats = calculate_statistics(text, pages)

            embed_model = load_embedding_model()
            st.session_state.embeddings = create_embeddings(
                st.session_state.chunks, embed_model
            )

            st.session_state.processed = True
            st.success("Document processed successfully!")

    if st.session_state.processed:
        st.divider()
        st.header("📊 Document Stats")
        s = st.session_state.stats
        st.metric("Pages", s["page_count"])
        st.metric("Words", s["word_count"])
        st.metric("Avg Words/Page", s["avg_words_per_page"])

        st.subheader("Top Keywords")
        for w, c in s["top_words"]:
            st.write(f"• {w}: {c}")

        if st.button("📝 Generate Summary"):
            llm = load_llm()
            st.info(generate_summary(st.session_state.full_text, llm))

# -------------------- CHAT --------------------
st.header("💬 Chat")

if not st.session_state.processed:
    st.info("Upload and process a PDF to start chatting.")
else:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    query = st.chat_input("Ask about rules, facilities, check-in...")
    if query:
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)

        embed_model = load_embedding_model()
        llm = load_llm()

        context = retrieve_chunks(
            query,
            embed_model,
            st.session_state.embeddings,
            st.session_state.chunks,
        )

        answer = generate_response(query, context, llm)

        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.write(answer)

        st.rerun()
