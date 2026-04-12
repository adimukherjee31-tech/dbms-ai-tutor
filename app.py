import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide")

# FIX: Using 'gemini-1.5-pro' is the most stable and avoids the 404 error
MODEL_NAME = "gemini-1.5-pro" 

def get_pdf_text(pdf_docs):
    """High-performance extraction for technical textbooks."""
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    """Chunks text so the AI can handle 500+ page books."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Saves a local search index using HuggingFace."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Socratic pedagogical prompt logic."""
    prompt_template = """
    You are Socrates, a pedagogical tutor helping EEE/ECE students bridge into CSE, AI, and ML.
    
    RULES:
    1. If the answer is in the Context, provide it and end with "[SOURCE: TEXTBOOK]".
    2. If the answer is NOT in the Context, explain from your knowledge and start with "[SOURCE: GENERAL AI KNOWLEDGE]".
    3. Use analogies to circuits, hardware, or logic gates to help the student understand.

    Context:\n {context}?\n
    Question: \n{question}\n

    Socratic Answer:
    """
    # Using the stable 'pro' model to ensure connection
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    """Processes questions and handles the Gemini connection."""
    try:
        genai.configure(api_key=api_key)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if not os.path.exists("faiss_index"):
            st.error("Please upload and 'Process' your PDFs first.")
            return

        # Load the index with safety flag enabled
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.markdown(f"### Socrates says:\n{response['output_text']}")
    except Exception as e:
        st.error(f"AI Connection Error: {e}")
        st.info("TIP: If you see a 404, please ensure your API key has 'Gemini 1.5 Pro' enabled in Google AI Studio.")

def main():
    st.title("🎓 Socrates: EEE/ECE to CSE Bridge")
    st.write("Ready for your **DBMS, AI, and Research Aptitude** exams.")

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input("Enter Google API Key:", type="password")
        pdf_docs = st.file_uploader("Upload Exam PDFs", accept_multiple_files=True)
        
        if st.button("Process & Analyze"):
            if api_key and pdf_docs:
                with st.spinner("Socrates is reading..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Ready for your questions!")
            else:
                st.warning("Please provide both an API key and PDF files.")

    user_question = st.chat_input("Ask Socrates a question...")

    if user_question:
        if api_key:
            user_input(user_question, api_key)
        else:
            st.error("Missing API Key in sidebar.")

if __name__ == "__main__":
    main()
