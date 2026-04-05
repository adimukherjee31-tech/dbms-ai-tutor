import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="DBMS AI Tutor", layout="wide")
st.title("🎓 JNTUH DBMS AI Tutor")
st.caption("Using LangChain + Gemini Stable for Exam Prep")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload DBMS Textbook (PDF)", type="pdf")
    
    st.header("2. Study Settings")
    page_limit = st.slider("Number of pages to analyze", 10, 1000, 250)
    
    st.header("3. Teaching Style")
    tone = st.selectbox("Choose Teaching Style", 
                        ["JNTUH Professor (Standard)", 
                         "Munnabhai Style (Hinglish)", 
                         "Class 8 Level (Simple)"])

# --- DATA PROCESSING ---
def clean_text(text):
    return text.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')

if api_key and uploaded_file:
    try:
        # Initialize the LLM via LangChain (More stable than raw genai calls)
        # We try 1.5-flash, but LangChain handles the API versioning logic
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0.7
        )

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
            os.remove(tmp_path)
            return db

        with st.spinner("Turbo-reading your textbook..."):
            vector_db = process_pdf(uploaded_file, page_limit)
        
        st.success("Tutor is online! Ready for your DBMS questions.")

        # --- CHAT INTERFACE ---
        user_query = st.chat_input("Ask about SQL, Normalization, Transactions...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            # 1. Search for context
            docs = vector_db.similarity_search(user_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # 2. Set Personality
            if tone == "Munnabhai Style (Hinglish)":
                personality = "Answer like Munnabhai. Use tapori Hinglish, call the student 'Mammu'. Explain correctly but use funny street analogies."
            elif tone == "Class 8 Level (Simple)":
                personality = "Explain like I am a 13-year-old child using toys or school examples."
            else:
                personality = "Answer as a professional JNTUH Professor. Provide a structured answer with points suitable for an exam."

            # 3. Create Prompt
            template = """
            System Instructions: {personality}
            
            Context from Textbook:
            {context}
            
            Question: {question}
            
            Helpful Answer:"""
            
            prompt = PromptTemplate(
                template=template, 
                input_variables=["personality", "context", "question"]
            )
            
            # 4. Generate Response
            with st.chat_message("assistant"):
                try:
                    chain = prompt | llm
                    response = chain.invoke({
                        "personality": personality,
                        "context": context_text,
                        "question": user_query
                    })
                    st.markdown(response.content)
                except Exception as api_err:
                    st.error(f"Model Error: {api_err}")
                    st.info("Attempting fallback to Gemini 1.0 Pro...")
                    # Fallback to older model if Flash is restricted
                    fallback_llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
                    chain = prompt | fallback_llm
                    response = chain.invoke({
                        "personality": personality,
                        "context": context_text,
                        "question": user_query
                    })
                    st.markdown(response.content)

    except Exception as e:
        st.error(f"General System Error: {e}")
else:
    st.info("Welcome! Please provide your API Key and PDF to start studying.")
