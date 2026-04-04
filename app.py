import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader # Faster for 1300 pages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor (Turbo Mode)")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("2. Speed Settings")
    # For a 1300 page book, 200-300 pages is best for a free prototype
    page_limit = st.slider("Number of pages to read", 50, 500, 250)
    
    uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")

# --- DATA CLEANING ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- LOGIC ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        
        # We try 2.0 Flash first, then fallback to 1.5 Flash
        model_name = 'gemini-2.0-flash' 
        model = genai.GenerativeModel(model_name)
        
        @st.cache_resource
        def process_pdf(file, limit):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            # FAST LOADING
            loader = PyMuPDFLoader(tmp_path)
            all_pages = loader.load()
            pages = all_pages[:limit] # Speed trick
            
            for page in pages:
                page.page_content = clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner(f"Reading first {page_limit} pages..."):
            vector_db = process_pdf(uploaded_file, page_limit)
        
        st.success("Tutor is ready!")

        # --- CHAT ---
        user_query = st.chat_input("Ask a DBMS question...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer as a DBMS Professor:"
            
            with st.chat_message("assistant"):
                try:
                    # Execute the AI generation
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as api_err:
                    # FALLBACK: If 2.0 fails, try 1.5
                    st.warning("Trying fallback model...")
                    alt_model = genai.GenerativeModel('gemini-1.5-flash')
                    response = alt_model.generate_content(prompt)
                    st.write(response.text)
                    
    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("Enter API Key and upload PDF to start.")
