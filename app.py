import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Socrates AI", layout="wide")

# ==========================================================
# 1. EDIT YOUR TITLES HERE
# ==========================================================
st.title("🎓 Socrates: The Pedagogical AI Tutor") 
st.markdown("### Grounded Learning Bridge App for EEE and ECE students getting skill ready for. cse ,ai,ml")
# ==========================================================

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ 1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("⚡ 2. Speed Setting")
    page_limit = st.slider("Number of pages to index", 10, 1000, 300)
    
    st.header("📂 3. Knowledge Base")
    uploaded_file = st.file_uploader("Upload your Textbook (PDF)", type="pdf")
    
    st.markdown("---")
    st.info("A PhD Entrance Prototype focusing on Grounded RAG.")

# --- DATA CLEANING ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- MAIN LOGIC ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        
        # FIX: We try the most stable model name string
        # If 'gemini-1.5-flash' fails, we will catch it below
        model_name = 'gemini-1.5-flash'
        model = genai.GenerativeModel(model_name=model_name)
        
        @st.cache_resource(show_spinner=False)
        def process_textbook(file, limit):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
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
            return db

        with st.spinner("Socrates is analyzing the text..."):
            vector_db = process_textbook(uploaded_file, page_limit)
        
        st.success("Analysis Complete! Ask Socrates a question.")

        # --- CHAT ---
        user_query = st.chat_input("Ask a question from the book...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            ROLE: You are 'Socrates', a pedagogical AI tutor.
            STRICT RULE: Use the 'TEXTBOOK CONTEXT' below to answer. 
            If not found, say: "This point isn't in your textbook, but..."
            End with a Socratic Question.

            TEXTBOOK CONTEXT:
            {context}

            QUESTION:
            {user_query}
            """
            
            with st.chat_message("assistant"):
                try:
                    # Execute generation
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    
                    with st.expander("🔍 View Textbook References"):
                        st.info(context)
                except Exception as api_err:
                    st.error(f"AI Model Error: {api_err}")
                    st.info("TIP: Check your API Key or try 'gemini-1.5-pro' in the code.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Welcome! Please enter your API Key and upload a PDF to begin.")
