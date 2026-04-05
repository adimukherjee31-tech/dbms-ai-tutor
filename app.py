import streamlit as st
import os
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="DBMS AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor")
st.info("Study Mode: RAG (Retrieval Augmented Generation) enabled.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF Textbook", type="pdf")
    
    st.header("2. Settings")
    # Using 1.5-flash as default, adding 1.0-pro as backup
    model_choice = st.selectbox("Select AI Model", ["gemini-1.5-flash", "gemini-1.0-pro"])
    tone = st.selectbox("Teaching Style", ["Professor", "Munnabhai (Hinglish)", "Simple"])
    page_limit = st.slider("Pages to index", 10, 500, 200)

# --- CORE FUNCTIONS ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

# --- PROCESSING LOGIC ---
if api_key and uploaded_file:
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=model_choice,
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
            
            # Clean and Split
            for d in docs:
                d.page_content = clean_text(d.page_content)
                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner("Processing textbook..."):
            vector_db = get_vector_db(uploaded_file, page_limit)

        # --- CHAT UI ---
        query = st.chat_input("Ask a DBMS question...")
        
        if query:
            with st.chat_message("user"):
                st.write(query)

            # 1. Search Vector DB
            context_docs = vector_db.similarity_search(query, k=3)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            # 2. Define Personality
            styles = {
                "Professor": "Professional JNTUH Professor. Use bullet points, clear headings, and exam-oriented language.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call the user 'Mammu', use tapori analogies but keep technical facts 100% correct.",
                "Simple": "Explain like I'm 10 years old. Use simple everyday analogies."
            }

            # 3. Modern LangChain Chain (LCEL)
            prompt = ChatPromptTemplate.from_template("""
            Role: {personality}
            
            Context from Textbook:
            {context}
            
            Question: {question}
            
            Instructions: If the answer is in the context, use it. If not, use your general DBMS knowledge to help the student prepare for their computer science exam.
            """)

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
                    st.error(f"AI Error: {e}")
                    st.info("Note: If you get a 404, try switching the 'Model' in the sidebar to gemini-1.0-pro.")

    except Exception as general_err:
        st.error(f"System Error: {general_err}")
else:
    st.warning("Please enter your API Key and upload a PDF in the sidebar.")
