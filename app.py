import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="DBMS AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor (Adaptive Mode)")
st.caption("Fix: Model naming updated for API stability.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get it from Google AI Studio")
    uploaded_file = st.file_uploader("Upload DBMS Textbook (PDF)", type="pdf")
    
    st.header("2. Study Settings")
    page_limit = st.slider("Read first 'X' pages", 10, 1000, 250)
    
    st.header("3. Teaching Style")
    tone = st.selectbox("Choose Teaching Style", 
                        ["JNTUH Professor (Standard)", 
                         "Munnabhai Style (Hinglish)", 
                         "Class 8 Level (Simple)"])

# --- HELPERS ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # FIX: Using the most stable model identifier
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        @st.cache_resource(show_spinner=False)
        def process_pdf(file, limit):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_pages = loader.load()
            pages = all_pages[:limit] 
            
            for page in pages:
                page.page_content = clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            
            # Cleanup temp file
            os.remove(tmp_path)
            return db

        with st.spinner("Analyzing textbook for exam topics..."):
            vector_db = process_pdf(uploaded_file, page_limit)
        
        st.success(f"Ready! Knowledge base built from {page_limit} pages.")

        # --- CHAT ---
        user_query = st.chat_input("Ask a DBMS question (e.g., 'What is Normalization?')")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # RAG Search
            docs = vector_db.similarity_search(user_query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Personality Prompting
            if tone == "Munnabhai Style (Hinglish)":
                personality = "Answer like Munnabhai. Use tapori Hinglish, call the student 'Mammu'. Keep facts correct but explanation funny."
            elif tone == "Class 8 Level (Simple)":
                personality = "Explain like I am a 13-year-old using simple real-world examples."
            else:
                personality = "Answer as a professional JNTUH Professor. Provide structured answers with bullet points suitable for a 10-mark question."

            full_prompt = f"""
            You are a helpful AI Tutor.
            Context: {context}
            
            Style: {personality}
            
            Question: {user_query}
            
            Note: If the answer is not in the context, use your general DBMS knowledge but prioritize the book content.
            """
            
            with st.chat_message("assistant"):
                try:
                    # FIX: Handle potential API call version issues
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as api_err:
                    st.error(f"AI Error: {api_err}")
                    st.info("Check if your API Key is valid and has Gemini 1.5 access.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Welcome! Please enter your Gemini API Key and upload your PDF to begin your preparation.")
