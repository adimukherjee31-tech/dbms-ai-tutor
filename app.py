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

# This is the FIX: 'gemini-1.5-pro' avoids the 404 error currently seen with 'flash'
STABLE_MODEL = "gemini-1.5-pro"

def get_pdf_text(pdf_docs):
    """Fast extraction for technical CS textbooks."""
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are Socrates, a pedagogical tutor helping EEE/ECE students bridge into CSE, AI, and ML.
    
    RULES:
    1. If information is in Context, answer and end with: "[SOURCE: TEXTBOOK]"
    2. If NOT in Context, use your internal knowledge but start with: "[SOURCE: GENERAL AI KNOWLEDGE]"
    3. Use hardware/circuit analogies to explain software concepts.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=STABLE_MODEL, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    try:
        genai.configure(api_key=api_key)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load index with safety flag
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.write("Socrates:", response["output_text"])
    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("🎓 Socrates: Pedagogical Exam Tutor")
    
    with st.sidebar:
        st.header("Setup Center")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        pdf_docs = st.file_uploader("Upload Exam Materials (PDF)", accept_multiple_files=True)
        
        if st.button("Process & Study"):
            if api_key and pdf_docs:
                with st.spinner("Analyzing books..."):
                    raw_text = get_pdf_text(pdf_docs)
                    get_vector_store(get_text_chunks(raw_text))
                    st.success("Analysis Complete!")
            else:
                st.error("API Key and PDF required.")

    user_question = st.chat_input("Ask about DBMS, Research, or AI...")
    if user_question:
        if api_key:
            user_input(user_question, api_key)
        else:
            st.error("Enter API Key in sidebar.")

if __name__ == "__main__":
    main()
