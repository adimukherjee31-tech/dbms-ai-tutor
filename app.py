import streamlit as st
import os
import tempfile
import time
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import fitz  # This comes from the pymupdf package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# CSS: FORCE COLORFUL EMOJIS & HIDE BLACK DOTS
st.markdown("""
    <style>
    li::marker { content: none !important; }
    ul { list-style-type: none !important; padding-left: 0 !important; }
    li { list-style-type: none !important; padding-left: 0 !important; margin-bottom: 12px !important; }
    .stMarkdown p, .stMarkdown li { 
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
        font-size: 1.1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- DIRECT GOOGLE EMBEDDING CLASS ---
class DirectGoogleEmbeddings:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts):
        embeddings = []
        # Batching 50 texts at a time for speed and quota safety
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            try:
                response = genai.embed_content(model=self.model, content=batch, task_type="retrieval_document")
                embeddings.extend(response['embedding'])
                time.sleep(0.3) 
            except:
                time.sleep(2) 
                response = genai.embed_content(model=self.model, content=batch, task_type="retrieval_document")
                embeddings.extend(response['embedding'])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(model=self.model, content=text, task_type="retrieval_query")
        return response['embedding']

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    st.header("2. Study Settings")
    tone = st.selectbox("Teaching Style", [
        "Professor", 
        "Munnabhai (Hinglish)", 
        "Physicswallah UGC-NET Coach", 
        "Simple"
    ])
    
    st.info("⚡ **Turbo Mode**: Loading specific pages only.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_pg, end_pg = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        genai.configure(api_key=api_key)
        chat_model = genai.GenerativeModel('gemini-1.5-flash')

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_bytes, _start, _end, _key):
            # Open PDF with fitz (very fast loading)
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            text_docs = []
            final_end = min(_end, len(doc))
            for pg_num in range(_start-1, final_end):
                page_text = doc[pg_num].get_text()
                if page_text.strip():
                    text_docs.append(Document(page_content=page_text, metadata={"page": pg_num+1}))
            
            if not text_docs:
                return None

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(text_docs)
            
            embeddings = DirectGoogleEmbeddings(_key)
            db = FAISS.from_documents(chunks, embeddings)
            return db

        with st.spinner(f"🚀 Indexing pages {start_pg} to {end_pg}..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_pg, end_pg, api_key)
            
            if vector_db:
                st.sidebar.success("✅ Book Ready!")
            else:
                st.error("No text found in those pages. Try a different range.")

        # --- CHAT ---
        if vector_db:
            query = st.chat_input("Ask a question from the book...")
            if query:
                with st.chat_message("user"): st.write(query)

                context_docs = vector_db.similarity_search(query, k=5)
                context_text = "\n\n".join([d.page_content for d in context_docs])

                styles = {
                    "Professor": "Academic Tutor. Professional headers, aesthetic markers.",
                    "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, 'Mammu', and funny analogies.",
                    "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'. Focus on key exam points in Hinglish.",
                    "Simple": "Explain like I'm 10 with bright, colorful examples."
                }

                system_prompt = f"""
                You are Socrates. Answer the Question using the Context provided.
                
                RULES:
                1. If in Context: End with "[SOURCE: TEXTBOOK]"
                2. If not: Start with "[SOURCE: GENERAL AI KNOWLEDGE]"
                
                AESTHETIC RULES:
                - NEVER use black dots or dashes (-, *, •).
                - START EVERY POINT with a unique BRIGHT Pinterest emoji (🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸).
                - Sub-points MUST use "╰┈➤ 💖" and a DIFFERENT emoji.
                
                Style: {styles[tone]}
                Context: {context_text}
                """

                with st.chat_message("assistant"):
                    response = chat_model.generate_content(f"{system_prompt}\n\nQuestion: {query}")
                    st.markdown(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Enter API Key and upload PDF to begin.")
