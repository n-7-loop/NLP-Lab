import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from collections import Counter
import requests
import json

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# HuggingFace API Configuration
HF_API_TOKEN = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Session state initialization
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'stats' not in st.session_state:
    st.session_state.stats = {}
if 'full_text' not in st.session_state:
    st.session_state.full_text = ""

@st.cache_resource
def load_embedding_model():
    """Load sentence embedding model"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def query_huggingface_api(payload):
    """Query HuggingFace API"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        page_count = len(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip(), page_count
    except Exception as e:
        st.error(f"PDF reading error: {str(e)}")
        return None, 0

def chunk_text(text, chunk_size=700):
    """Split text into chunks with overlap"""
    words = text.split()
    chunks = []
    overlap = 100
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def calculate_statistics(text, page_count):
    """Generate document statistics"""
    words = text.split()
    word_count = len(words)
    avg_words = word_count / page_count if page_count > 0 else 0
    
    # Top frequent words (excluding stopwords)
    stop_words = set(stopwords.words('english'))
    filtered = [w.lower() for w in words if w.isalpha() and len(w) > 2 and w.lower() not in stop_words]
    top_words = Counter(filtered).most_common(10)
    
    return {
        'page_count': page_count,
        'word_count': word_count,
        'avg_words_per_page': round(avg_words, 2),
        'top_words': top_words
    }

def create_vector_store(chunks, model):
    """Create FAISS index"""
    embeddings = model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_chunks(query, model, index, chunks, k=4):
    """Retrieve relevant chunks"""
    query_embedding = model.encode([query], show_progress_bar=False)
    _, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_response(query, context_chunks):
    """Generate response using HuggingFace API"""
    context = "\n\n".join(context_chunks[:3])
    
    prompt = f"""As a helpful hostel assistant, based on the following context from the hostel PDF:

Context: {context[:1000]}

Question: {query}

Answer:"""
    
    result = query_huggingface_api({"inputs": prompt, "parameters": {"max_length": 300, "temperature": 0.7}})
    
    if "error" in result:
        return f"**Relevant information from document:**\n\n{context[:600]}...\n\n*API Error: {result['error']}*"
    
    try:
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', context[:500])
        else:
            return context[:500]
    except:
        return context[:500]

def generate_summary(text):
    """Generate document summary using HuggingFace API"""
    prompt = f"Summarize the following hostel document in 2-3 sentences, highlighting key facilities and policies:\n\n{text[:1500]}"
    
    result = query_huggingface_api({"inputs": prompt, "parameters": {"max_length": 150, "temperature": 0.5}})
    
    if "error" in result:
        return f"Summary generation failed: {result['error']}"
    
    try:
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', "Summary generation failed")
        else:
            return "Summary generation failed"
    except:
        return "Summary generation failed"

# Streamlit UI
st.set_page_config(page_title="Hostel RAG Chat", page_icon="🏨", layout="wide")
st.title("🏨 Hostel Document Chat Assistant")
st.markdown("*Upload hostel PDF and ask questions about policies, facilities, and rules*")

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload Hostel PDF", type=['pdf'])
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing PDF..."):
            text, pages = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.session_state.full_text = text
                st.session_state.chunks = chunk_text(text)
                st.session_state.stats = calculate_statistics(text, pages)
                
                embed_model = load_embedding_model()
                st.session_state.vector_store = create_vector_store(st.session_state.chunks, embed_model)
                st.session_state.processed = True
                st.success("✅ Document processed successfully!")
    
    if st.session_state.processed:
        st.divider()
        st.header("📊 Hostel Document Stats")
        stats = st.session_state.stats
        
        col1, col2 = st.columns(2)
        col1.metric("Total Pages", stats['page_count'])
        col2.metric("Total Words", f"{stats['word_count']:,}")
        st.metric("Avg Words/Page", stats['avg_words_per_page'])
        
        st.subheader("Top 10 Keywords")
        for word, count in stats['top_words']:
            st.text(f"• {word.capitalize()}: {count}")
        
        st.divider()
        if st.button("📝 Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(st.session_state.full_text)
                st.info(f"**Summary:** {summary}")

# Main chat interface
st.header("💬 Chat with Hostel Document")

if not st.session_state.processed:
    st.info("👈 Please upload and process a PDF document to start chatting!")
else:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    # Chat input
    query = st.chat_input("Ask about check-in times, facilities, rules, etc...")
    
    if query:
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("Generating answer..."):
            embed_model = load_embedding_model()
            relevant = retrieve_chunks(query, embed_model, st.session_state.vector_store, st.session_state.chunks)
            response = generate_response(query, relevant)
        
        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.write(response)
        st.rerun()

st.divider()
st.caption("💡 Example questions: 'What are the check-in procedures?' or 'What facilities are available?'")
