import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import nltk
from collections import Counter
import os
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Download NLTK stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

load_dotenv()

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
    """Load sentence-transformers model for embeddings"""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource
def load_llm():
    """Load LLM - requires HUGGINGFACEHUB_API_TOKEN in .env file"""
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_token:
        try:
            # Using google/flan-t5-large for better quality responses
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-large",
                model_kwargs={"temperature": 0.3, "max_length": 512},
                huggingfacehub_api_token=api_token
            )
            return llm
        except Exception as e:
            st.warning(f"LLM initialization failed: {str(e)}")
            return None
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text content from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        page_count = len(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        return text.strip(), page_count
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, 0

def chunk_text(text, chunk_size=700):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    overlap = 100  # Overlap for context continuity
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def calculate_statistics(text, page_count):
    """Calculate document statistics"""
    words = text.split()
    word_count = len(words)
    avg_words = word_count / page_count if page_count > 0 else 0
    
    # Get top frequent words excluding stopwords
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
    """Create FAISS vector index from text chunks"""
    embeddings = model.encode(chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_chunks(query, model, index, chunks, k=4):
    """Retrieve top-k relevant chunks"""
    query_embedding = model.encode([query], show_progress_bar=False)
    _, indices = index.search(np.array(query_embedding).astype('float32'), k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def generate_response(query, context_chunks, llm):
    """Generate contextual response using LLM"""
    context = "\n\n".join(context_chunks[:3])
    
    if llm:
        prompt_template = """As a helpful hostel assistant, based on the following context from the hostel PDF:

{context}

Answer the question: {question}

Answer:"""
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = LLMChain(llm=llm, prompt=prompt)
        
        try:
            response = chain.run(context=context, question=query)
            return response.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}\n\nRelevant context:\n{context[:400]}..."
    else:
        return f"""**LLM not configured.** Here's relevant information from the document:

{context[:500]}...

*To enable AI responses:*
1. Get a free HuggingFace API token at https://huggingface.co/settings/tokens
2. Create a .env file with: HUGGINGFACEHUB_API_TOKEN=your_token_here
3. Restart the application"""

def generate_summary(text, llm):
    """Generate document summary"""
    if llm:
        prompt = f"Summarize the following hostel document in 2-3 sentences:\n\n{text[:2000]}"
        try:
            return llm(prompt).strip()
        except:
            return "Summary generation failed."
    return "LLM required for summary generation."

# Streamlit UI
st.set_page_config(page_title="Hostel RAG Assistant", page_icon="🏨", layout="wide")
st.title("🏨 Hostel Document Chat Assistant")
st.markdown("*Upload hostel documentation and get instant answers about policies, facilities, and rules*")

# Sidebar: Upload and Statistics
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("Upload Hostel PDF", type=['pdf'])
    
    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing PDF..."):
            text, pages = extract_text_from_pdf(uploaded_file)
            
            if text:
                st.session_state.full_text = text
                st.session_state.chunks = chunk_text(text)
                st.session_state.stats = calculate_statistics(text, pages)
                
                model = load_embedding_model()
                st.session_state.vector_store = create_vector_store(st.session_state.chunks, model)
                st.session_state.processed = True
                st.success("✅ Document processed!")
    
    # Display statistics
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

# Main Chat Interface
st.header("💬 Ask About Your Hostel")

if not st.session_state.processed:
    st.info("👈 Upload and process a PDF document to start chatting!")
else:
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    # Chat input
    query = st.chat_input("e.g., What are the check-in times?")
    
    if query:
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.write(query)
        
        with st.spinner("Searching document..."):
            model = load_embedding_model()
            llm = load_llm()
            relevant = retrieve_chunks(query, model, st.session_state.vector_store, st.session_state.chunks)
            response = generate_response(query, relevant, llm)
        
        st.session_state.chat_history.append(("assistant", response))
        with st.chat_message("assistant"):
            st.write(response)
        st.rerun()

# Footer
st.divider()
st.caption("💡 Ask specific questions like: 'What facilities are available?' or 'What are the house rules?'")
