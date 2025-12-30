import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from collections import Counter
import requests
import json
import time

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

def query_huggingface_api(payload, retries=3):
    """Query HuggingFace API with retry logic"""
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    for attempt in range(retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(10)
                continue
            else:
                return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
                continue
            return {"error": str(e)}
    
    return {"error": "Max retries reached"}

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        page_count = len(pdf_reader.pages)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
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
    try:
        embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index
    except Exception as e:
        st.error(f"Vector store creation error: {str(e)}")
        return None

def retrieve_chunks(query, model, index, chunks, k=4):
    """Retrieve relevant chunks"""
    try:
        query_embedding = model.encode([query], show_progress_bar=False)
        _, indices = index.search(np.array(query_embedding).astype('float32'), k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]
    except Exception as e:
        st.error(f"Retrieval error: {str(e)}")
        return []

def generate_response(query, context_chunks):
    """Generate response using HuggingFace API"""
    if not context_chunks:
        return "No relevant information found in the document."
    
    context = "\n\n".join(context_chunks[:3])
    
    prompt = f"""As a helpful hostel assistant, answer this question based on the hostel document context below.

Context: {context[:1200]}

Question: {query}

Provide a clear and concise answer:"""
    
    result = query_huggingface_api({
        "inputs": prompt,
        "parameters": {
            "max_length": 250,
            "temperature": 0.7,
            "top_p": 0.9
        }
    })
    
    if "error" in result:
        return f"**Relevant Information from Document:**\n\n{context[:600]}...\n\n*Note: {result['error']}*"
    
    try:
        if isinstance(result, list) and len(result) > 0:
            answer = result[0].get('generated_text', '').strip()
            if answer:
                return answer
        return context[:500] + "..."
    except:
        return context[:500] + "..."

def generate_summary(text):
    """Generate document summary using HuggingFace API"""
    prompt = f"Summarize this hostel document in 3 sentences, highlighting key facilities, policies, and rules:\n\n{text[:2000]}"
    
    result = query_huggingface_api({
        "inputs": prompt,
        "parameters": {
            "max_length": 150,
            "temperature": 0.5
        }
    })
    
    if "error" in result:
        return f"Summary unavailable: {result['error']}"
    
    try:
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', "Summary generation failed").strip()
        return "Summary generation failed"
    except:
        return "Summary generation failed"

# Streamlit UI Configuration
st.set_page_config(
    page_title="Hostel RAG Chat Assistant",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏨 Hostel Document Chat Assistant")
st.markdown("*Your AI-powered hostel information assistant - Upload a PDF and ask questions!*")

# Sidebar - Document Upload and Statistics
with st.sidebar:
    st.header("📄 Document Management")
    
    uploaded_file = st.file_uploader(
        "Upload Hostel PDF",
        type=['pdf'],
        help="Upload a PDF containing hostel information"
    )
    
    if uploaded_file:
        if st.button("🔄 Process Document", type="primary", use_container_width=True):
            with st.spinner("📖 Reading PDF..."):
                text, pages = extract_text_from_pdf(uploaded_file)
            
            if text and len(text) > 100:
                with st.spinner("✂️ Chunking text..."):
                    st.session_state.full_text = text
                    st.session_state.chunks = chunk_text(text)
                
                with st.spinner("📊 Calculating statistics..."):
                    st.session_state.stats = calculate_statistics(text, pages)
                
                with st.spinner("🧠 Creating embeddings..."):
                    embed_model = load_embedding_model()
                    st.session_state.vector_store = create_vector_store(
                        st.session_state.chunks, 
                        embed_model
                    )
                
                if st.session_state.vector_store:
                    st.session_state.processed = True
                    st.success("✅ Document processed successfully!")
                else:
                    st.error("Failed to create vector store")
            else:
                st.error("PDF appears to be empty or unreadable")
    
    # Display Statistics
    if st.session_state.processed:
        st.divider()
        st.header("📊 Document Statistics")
        stats = st.session_state.stats
        
        col1, col2 = st.columns(2)
        col1.metric("📄 Pages", stats['page_count'])
        col2.metric("📝 Words", f"{stats['word_count']:,}")
        
        st.metric("📈 Avg Words/Page", stats['avg_words_per_page'])
        
        with st.expander("🔑 Top 10 Keywords", expanded=False):
            for word, count in stats['top_words']:
                st.text(f"• {word.capitalize()}: {count}")
        
        st.divider()
        
        if st.button("📝 Generate Summary", use_container_width=True):
            with st.spinner("Generating AI summary..."):
                summary = generate_summary(st.session_state.full_text)
                st.info(f"**Summary:**\n\n{summary}")

# Main Chat Interface
st.header("💬 Chat with Your Document")

if not st.session_state.processed:
    st.info("👈 Please upload and process a PDF document from the sidebar to start chatting!")
    
    st.markdown("""
    ### How to use:
    1. Upload a hostel-related PDF (rules, policies, facilities, etc.)
    2. Click "Process Document" to analyze the PDF
    3. Ask questions in the chat below
    4. Get AI-powered answers based on your document
    
    ### Example Questions:
    - What are the check-in and check-out times?
    - What facilities are available?
    - What are the house rules?
    - How do I make a booking?
    - What is the cancellation policy?
    """)
else:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    # Chat input
    query = st.chat_input("Ask about check-in times, facilities, rules, booking policies...")
    
    if query:
        # Add user message
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching document..."):
                embed_model = load_embedding_model()
                relevant = retrieve_chunks(
                    query,
                    embed_model,
                    st.session_state.vector_store,
                    st.session_state.chunks
                )
            
            with st.spinner("💭 Generating answer..."):
                response = generate_response(query, relevant)
            
            st.write(response)
        
        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", response))
        st.rerun()

# Footer
st.divider()
st.caption("🤖 Powered by HuggingFace AI | 🔒 Your data is processed securely")
st.caption("💡 Pro Tip: Ask specific questions for more accurate answers!")
