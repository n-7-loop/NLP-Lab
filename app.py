import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Download NLTK stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

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

@st.cache_resource
def load_llm():
    """Load local LLM model - using FLAN-T5 base for CPU efficiency"""
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        device = 0 if torch.cuda.is_available() else -1
        llm = pipeline("text2text-generation", model=model, tokenizer=tokenizer, 
                      max_length=512, device=device, truncation=True)
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

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

def generate_response(query, context_chunks, llm):
    """Generate response using local LLM"""
    context = "\n\n".join(context_chunks[:3])
    
    if llm:
        prompt = f"""As a helpful hostel assistant, based on the following context from the hostel PDF:

{context[:1000]}

Answer the question: {query}"""
        
        try:
            response = llm(prompt, max_length=300, do_sample=False)[0]['generated_text']
            return response.strip()
        except Exception as e:
            return f"Error: {str(e)}\n\nRelevant context:\n{context[:400]}..."
    else:
        return f"**Relevant information from document:**\n\n{context[:600]}...\n\n*LLM not loaded. Showing retrieved context only.*"

def generate_summary(text, llm):
    """Generate document summary"""
    if llm:
        prompt = f"Summarize the following hostel document in 2-3 sentences, highlighting key facilities and policies:\n\n{text[:1500]}"
        try:
            summary = llm(prompt, max_length=150, do_sample=False)[0]['generated_text']
            return summary.strip()
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    return "LLM not available for summary generation."

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
                st.success("✅ Processed successfully!")
    
    if st.session_state.processed:
        st.divider()
        st.header("📊 Hostel Document Stats")
        stats = st.session_state.stats
        
        col1, col2 = st.columns(2)
        col1.metric("Pages", stats['page_count'])
        col2.metric("Words", f"{stats['word_count']:,}")
        st.metric("Avg Words/Page", stats['avg_words_per_page'])
        
        st.subheader("Top 10 Keywords")
        for word, count in stats['top_words']:
            st.text(f"• {word.capitalize()}: {count}")
        
        st.divider()
        if st.button("📝 Generate Summary"):
            with st.spinner("Generating summary..."):
                llm = load_llm()
                summary = generate_summary(st.session_state.full_text, llm)
                st.info(summary)

# Main chat interface
st.header("💬 Chat Interface")

if not st.session_state.processed:
    st.info("👈 Upload and process a PDF to begin chatting!")
else:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    # Chat input
    query = st.chat_input("Ask about check-in times, facilities, rules...")
    
    if query:
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("Generating answer..."):
            embed_model = load_embedding_model()
            llm = load_llm()
            relevant = retrieve_chunks(query, embed_model, st.session_state.vector_store, st.session_state.chunks)
            response = generate_response(query, relevant, llm)
        
        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.write(response)
        st.rerun()

st.divider()
st.caption("💡 Example: 'What are the check-in procedures?' or 'What facilities are available?'")
