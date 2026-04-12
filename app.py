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

# --- CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide")

# The fix for the 404: Using 'gemini-1.5-pro' or 'gemini-1.5-flash-latest'
# 'pro' is better for the complex pedagogical explanations you need for DBMS/AI
MODEL_NAME = "gemini-1.5-pro" 

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # Use PyMuPDF (fitz) for better extraction of technical text
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Restoring HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are Socrates, a pedagogical tutor helping EEE/ECE students bridge into CSE, AI, and ML.
    Your goal is to explain concepts like DBMS, Research Aptitude, and AI by building on their 
    existing knowledge of hardware and logic.
    
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    genai.configure(api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the index with safety check
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.markdown(f"**Socrates:** {response['output_text']}")

def main():
    st.title("🎓 Socrates: EEE/ECE to CSE Bridge")
    st.write("Ready to help with Research Aptitude, DBMS, and AI exams.")

    with st.sidebar:
        api_key = st.text_input("Enter Google API Key:", type="password")
        pdf_docs = st.file_uploader("Upload your textbooks (PDF)", accept_multiple_files=True)
        if st.button("Process Books"):
            if not api_key:
                st.error("Please provide API Key first")
            else:
                with st.spinner("Socrates is analyzing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Ready for questions!")

    user_question = st.chat_input("Ask a question from the books...")

    if user_question:
        if api_key:
            user_input(user_question, api_key)
        else:
            st.error("Please enter your API Key in the sidebar.")

if __name__ == "__main__":
    main()
