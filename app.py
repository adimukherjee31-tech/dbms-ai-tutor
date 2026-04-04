import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")

# --- CLEANING FUNCTION (Prevents the InvalidArgument Error) ---
def clean_text(text):
    # This removes hidden "junk" characters that crash the AI
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- LOGIC ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        # We use 'gemini-1.5-flash' - this is the correct official name
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        @st.cache_resource
        def process_pdf(file):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Clean the text from each page
            for page in pages:
                page.page_content = clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner("Reading textbook..."):
            vector_db = process_pdf(uploaded_file)
        
        st.success("Tutor is ready!")

        # --- CHAT ---
        user_query = st.chat_input("Ask a DBMS question...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # Search book
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Final Prompt
            prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer as a DBMS Professor:"
            
            with st.chat_message("assistant"):
                # We add a try-except here to catch any final API hiccups
                try:
                    response = model.generate_content(prompt)
                    st.write(response.text)
                except Exception as api_err:
                    st.error(f"AI Error: {api_err}")
                    
    except Exception as e:
        st.error(f"General Error: {e}")
else:
    st.info("Enter API Key and upload PDF to start.")
