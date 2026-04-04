import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# 1. UI Setup
st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("ðŸŽ“ Personalized AI Tutor for Indian Engineering Students")
st.markdown("### Subjects: CSE, AI, ML (Focus: DBMS)")

# 2. Sidebar for Configuration
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.header("2. Knowledge Base")
    uploaded_file = st.file_uploader("Upload your Textbook (PDF)", type="pdf")

# 3. The Logic (RAG Engine)
if api_key and uploaded_file:
    genai.configure(api_key=api_key)
    
    # Process PDF only once (Caching)
    @st.cache_resource
    def process_textbook(file):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        
        # Load and Split
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(pages)
        
        # Create Vector Database (Using local HuggingFace to stay free/unlimited)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        return db

    with st.spinner("Tutor is reading the book..."):
        vector_db = process_textbook(uploaded_file)
        st.sidebar.success("Book Indexed Successfully!")

    # 4. Chat Interface
    st.write("---")
    query = st.text_input("Ask a question from the book:")

    if query:
        # Search for answers in the book
        docs = vector_db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate Answer using Gemini
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are a helpful Indian Engineering Professor (JNTUH style).
        Explain the concept based ONLY on the context provided below. 
        If the answer isn't in the context, say you can't find it in the book but provide a general explanation.
        Use simple language, bullet points, and an example.
        
        CONTEXT FROM BOOK:
        {context}
        
        STUDENT QUESTION:
        {query}
        """
        
        with st.spinner("Thinking..."):
            response = model.generate_content(prompt)
            st.markdown("### Tutor's Explanation:")
            st.write(response.text)
            
            with st.expander("Show Book References"):
                st.info(context)
else:
    st.info("Please enter your API key and upload a PDF in the sidebar to start.")
