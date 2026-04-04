import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader # Ultra-fast loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor (Adaptive Mode)")

# --- SIDEBAR (Setup & Controls) ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")
    
    st.header("2. Speed & Scale")
    # This slider allows you to handle 1300+ page books on a free server
    page_limit = st.slider("Number of pages to read", 10, 1000, 250)
    
    st.header("3. Personality")
    tone = st.selectbox("Choose Teaching Style", 
                        ["JNTUH Professor (Standard)", 
                         "Munnabhai Style (Hinglish)", 
                         "Class 8 Level (Simple)"])

# --- DATA CLEANING ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        # Use the most stable free model: gemini-1.5-flash
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # We cache this so moving the slider or switching tones doesn't re-read the book
        @st.cache_resource(show_spinner=False)
        def process_pdf(file, limit):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            # TURBO LOADING
            loader = PyMuPDFLoader(tmp_path)
            all_pages = loader.load()
            pages = all_pages[:limit] # Applying the Speed Setting
            
            # Pre-processing
            for page in pages:
                page.page_content = clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            # Unlimited local embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner(f"Turbo-reading {page_limit} pages... Please wait."):
            vector_db = process_pdf(uploaded_file, page_limit)
        
        st.success(f"Tutor is ready with {page_limit} pages of context!")

        # --- CHAT INTERFACE ---
        user_query = st.chat_input("Ask a DBMS question...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # Step 1: Semantic Search (RAG)
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Step 2: Personality Logic
            if tone == "Munnabhai Style (Hinglish)":
                personality = "Answer like Munnabhai from the movie. Use tapori Hinglish, call the student 'Mammu', and use local analogies. Keep technical facts correct."
            elif tone == "Class 8 Level (Simple)":
                personality = "Explain like I am a 13-year-old child. Use very simple toy analogies and avoid jargon."
            else:
                personality = "Answer as a professional JNTUH DBMS Professor. Structure it for a 10-mark exam question with bullet points."

            # Step 3: Woven Prompt
            full_prompt = f"System Instruction: {personality}\n\nContext from Book: {context}\n\nQuestion: {user_query}"
            
            with st.chat_message("assistant"):
                try:
                    response = model.generate_content(full_prompt)
                    st.markdown(response.text)
                except Exception as api_err:
                    st.error(f"AI Error: {api_err}")
                    st.info("Try a different API key or check your Google AI Studio quota.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Welcome! Please enter your Gemini API Key and upload your DBMS textbook in the sidebar to start learning.")
