import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide")
st.title("🎓 Socrates: The Pedagogical AI Tutor")
st.markdown("### Specialized for JNTUH Engineering Students (CSE/AI/ML)")

# --- SIDEBAR: Configuration & Speed ---
with st.sidebar:
    st.header("⚙️ 1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    
    st.header("⚡ 2. Speed Setting")
    # Limiting pages makes the prototype run fast on free servers
    page_limit = st.slider("Number of pages to index", 10, 1000, 300)
    
    st.header("📂 3. Knowledge Base")
    uploaded_file = st.file_uploader("Upload your Textbook (PDF)", type="pdf")
    
    st.markdown("---")
    st.info("Built for PhD Entrance Prototype. Focus: Grounded RAG & Socratic Pedagogy.")

# --- DATA CLEANING ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- MAIN APPLICATION LOGIC ---
if api_key and uploaded_file:
    try:
        # Configure the AI Brain
        genai.configure(api_key=api_key)
        # Using the stable 1.5-flash model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # We cache the reading process so it doesn't repeat on every click
        @st.cache_resource(show_spinner=False)
        def process_textbook(file, limit):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            # TURBO LOADING: Using PyMuPDF for speed
            loader = PyMuPDFLoader(tmp_path)
            all_pages = loader.load()
            pages = all_pages[:limit]
            
            # Clean and Chunk
            for page in pages:
                page.page_content = clean_text(page.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            # LOCAL EMBEDDINGS: 100% Free and Unlimited
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner(f"Socrates is reading the first {page_limit} pages of your book..."):
            vector_db = process_textbook(uploaded_file, page_limit)
        
        st.success(f"I have studied {page_limit} pages. Ask me anything!")

        # --- CHAT INTERFACE ---
        user_query = st.chat_input("Ask Socrates a question from the book...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # STEP A: Retrieval (Finding the right pages)
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # STEP B: The Strict Socratic Prompt (Stopping hallucinations)
            prompt = f"""
            ROLE: You are 'Socrates', a pedagogical AI tutor for Indian B.Tech students.
            
            STRICT GROUNDING RULE: 
            1. Search the 'TEXTBOOK CONTEXT' provided below for the answer.
            2. If the answer exists in the book, explain it using a simple Indian analogy (like IRCTC, a local market, or a cricket match).
            3. If the answer IS NOT in the book, start your response by saying: "This specific point isn't in your textbook, but for your general understanding..."
            4. Never invent fake technical facts.
            5. Conclude your answer with a single 'Socratic Question' that tests if the student understood your explanation.
            
            TEXTBOOK CONTEXT:
            {context}
            
            STUDENT QUESTION:
            {user_query}
            """
            
            with st.chat_message("assistant"):
                try:
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                    
                    # Technical transparency (Good for PhD Demos)
                    with st.expander("🔍 View Textbook References (Grounded Source)"):
                        st.info(context)
                except Exception as api_err:
                    st.error(f"AI Error: {api_err}")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("👋 Welcome! Please enter your API Key and upload a PDF in the sidebar to start the Socratic learning session.")
