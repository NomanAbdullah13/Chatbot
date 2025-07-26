# app.py
import os
import tempfile
import pickle
import time
from dotenv import load_dotenv
import streamlit as st
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
import faiss
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "data"
CACHE_DIR = "cache"
PDF_NAME = "bangla_document.pdf"  
PDF_PATH = os.path.join(DATA_DIR, PDF_NAME)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
TESSERACT_CONFIG = '--oem 3 --psm 6 -l ben'

# Create directories if missing
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    """Extract Bangla text from PDF using OCR"""
    images = convert_from_path(pdf_path, dpi=300)
    full_text = ""
    
    with st.spinner("Extracting text from PDF..."):
        progress_bar = st.progress(0)
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, config=TESSERACT_CONFIG)
            full_text += text + "\n\n"
            progress_bar.progress((i + 1) / len(images))
        st.success("Text extraction complete!")
    return full_text

def process_pdf(pdf_path, model):
    """Process PDF and create embeddings"""
    cache_path = os.path.join(CACHE_DIR, f"{os.path.basename(pdf_path)}.pkl")
    
    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    # Process PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=['\n\n', '।', '\n', ' ', '']
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    with st.spinner("Creating embeddings..."):
        embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save to cache
    with open(cache_path, "wb") as f:
        pickle.dump((chunks, index), f)
    
    return chunks, index

def query_openai(prompt):
    """Send query to OpenAI GPT-4o"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions in Bangla."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"OpenAI Error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Bangla PDF QA", page_icon="📄", layout="wide")
st.title("📄 Bangla PDF Question Answering System")

# Sidebar configuration
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
else:
    st.sidebar.warning("Enter OpenAI API key to enable GPT-4o")

# Main processing
if not os.path.exists(PDF_PATH):
    st.warning(f"PDF not found at: {PDF_PATH}")
    st.info(f"Please place your Bangla PDF in the '{DATA_DIR}' folder as '{PDF_NAME}'")
    st.stop()

embedding_model = load_embedding_model()
chunks, index = process_pdf(PDF_PATH, embedding_model)

# Query interface
st.subheader("Ask questions about the document")
question = st.text_input("Enter your question in Bangla:", "")

if question and openai_api_key:
    # Embed question
    question_embedding = embedding_model.encode([question])
    
    # Search index
    D, I = index.search(question_embedding, k=3)
    context = "\n\n".join([chunks[i] for i in I[0]])
    
    # Build prompt
    prompt = f"""
    নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দাও:
    
    প্রসঙ্গ:
    {context}
    
    প্রশ্ন: {question}
    উত্তর:
    """
    
    # Get response
    with st.spinner("Generating answer..."):
        answer = query_openai(prompt)
    
    if answer:
        st.subheader("Answer:")
        st.success(answer)
        
        with st.expander("See context used"):
            st.text(context)

# Add footer
st.markdown("---")
st.caption("Note: First run will take time for OCR processing. Subsequent runs will use cached embeddings.")