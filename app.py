import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# --- CONFIG ---
st.set_page_config(page_title="Socrates: Exam Tutor", layout="wide")

# We define a list of models. If 'flash' fails, the code will automatically try 'pro'.
MODELS_TO_TRY = ["gemini-1.5-pro", "gemini-1.5-flash-latest", "gemini-1.0-pro"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # HF Embeddings (Keep local to save API quota)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(model_name):
    prompt_template = """
    You are Socrates, a pedagogical tutor helping EEE/ECE students bridge into CSE/AI/ML.
    
    RULES:
    1. If the answer is in the Context, provide it and end with: "[SOURCE: TEXTBOOK]"
    2. If the answer is NOT in the Context, provide an explanation but start with: "[SOURCE: GENERAL AI KNOWLEDGE]"
    3. Use analogies to circuits, hardware, or logic gates to help the student understand.

    Context:\n {context}?\n
    Question: \n{question}\n

    Socratic Answer:
    """
    model = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    genai.configure(api_key=api_key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # TRY MODELS UNTIL ONE WORKS (Prevents 404)
    response_text = ""
    success = False
    
    for model_name in MODELS_TO_TRY:
        try:
            chain = get_conversational_chain(model_name)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            response_text = response["output_text"]
            success = True
            break # Exit loop if model works
        except Exception as e:
            continue # Try next model if this one 404s
            
    if success:
        st.markdown(f"### Socrates Response:\n{response_text}")
    else:
        st.error("All Gemini models are currently unavailable. Please check your API key.")

def main():
    st.title("🎓 Socrates: EEE/ECE to CSE Bridge")
    st.markdown("Focused on **DBMS, AI, and Research Aptitude** exams.")

    with st.sidebar:
        st.title("Setup")
        api_key = st.text_input("Enter Gemini API Key:", type="password")
        pdf_docs = st.file_uploader("Upload Exam PDFs", accept_multiple_files=True)
        
        if st.button("Process & Index"):
            if api_key and pdf_docs:
                with st.spinner("Socrates is analyzing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Analysis Complete!")
            else:
                st.error("Please provide Key and PDF files.")

    user_question = st.chat_input("Ask a question...")

    if user_question:
        if api_key:
            user_input(user_question, api_key)
        else:
            st.error("API Key missing in sidebar.")

if __name__ == "__main__":
    main()
