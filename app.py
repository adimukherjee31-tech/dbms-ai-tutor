import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")
    
    st.header("2. Personality")
    tone = st.selectbox("Teaching Style", ["Professor", "Munnabhai", "Class 8"])

# --- LOGIC ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        
        # This is the most STABLE model name globally for free tier
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        @st.cache_resource
        def process_pdf(file):
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            pages = loader.load()
            
            # Limit to 300 pages to ensure it fits in free memory
            pages = pages[:300] 
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(pages)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner("Tutor is reading..."):
            vector_db = process_pdf(uploaded_file)
        
        st.success("Tutor is Ready!")

        # --- CHAT ---
        user_query = st.chat_input("Ask a question...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # Search context
            docs = vector_db.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Persona Logic
            if tone == "Munnabhai":
                p = "Answer like Munnabhai from the movie. Use tapori Hinglish and local analogies."
            elif tone == "Class 8":
                p = "Explain like I am a 13 year old. Use very simple toy analogies."
            else:
                p = "Answer as a JNTUH DBMS Professor for a 10-mark exam question."

            prompt = f"System Instruction: {p}\n\nContext: {context}\n\nQuestion: {user_query}"
            
            with st.chat_message("assistant"):
                try:
                    response = model.generate_content(prompt)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")
                    st.info("Check if your API key is correct and your internet is stable.")

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.info("Please enter your API Key and upload a PDF.")
