import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates Pedagogical Bridge AI App for EEE and ECE", layout="wide")
st.title("🎓 Socrates Pedagogical Bridge AI App for EEE and ECE")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF Textbook", type="pdf")
    
    st.header("2. Study Settings")
    tone = st.selectbox("Teaching Style", ["Professor", "Munnabhai (Hinglish)", "Simple"])
    page_limit = st.slider("Pages to index", 10, 500, 200)

# --- AUTO-DETECT MODELS ---
def get_working_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # Clean the names (remove 'models/')
        clean_models = [m.replace('models/', '') for m in models]
        
        # Priority list
        for preferred in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
            if preferred in clean_models:
                return preferred
        return clean_models[0] if clean_models else None
    except Exception:
        return "gemini-1.5-flash" # Fallback

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        # Detect model
        active_model = get_working_model(api_key)
        st.sidebar.success(f"Connected to: {active_model}")

        llm = ChatGoogleGenerativeAI(
            model=active_model,
            google_api_key=api_key,
            temperature=0.3
        )

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file, limit):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()[:limit]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner("Preparing your DBMS knowledge base..."):
            vector_db = get_vector_db(uploaded_file, page_limit)

        # --- CHAT ---
        query = st.chat_input("Ask a question related to CSE/AI/ML ...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)

            context_docs = vector_db.similarity_search(query, k=4)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Professional JNTUH Professor. Use bullet points and exam-style headings.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call the user 'Mammu', use funny analogies.",
                "Simple": "Explain like I'm 10 years old."
            }

            prompt = ChatPromptTemplate.from_template("""
            Context: {context}
            Style: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                try:
                    response = chain.invoke({
                        "personality": styles[tone],
                        "context": context_text,
                        "question": query
                    })
                    st.markdown(response)
                except Exception as e:
                    st.error(f"AI Connection Error: {e}")
                    st.info("Please verify your API key at Google AI Studio.")

    except Exception as general_err:
        st.error(f"System Error: {general_err}")
else:
    st.warning("Enter API Key and upload PDF to start.")
