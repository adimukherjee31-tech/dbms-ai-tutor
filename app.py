import streamlit as st
import os
import tempfile
import time
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# CSS: FORCE COLORFUL EMOJIS & HIDE BLACK BULLETS
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

# --- NO-DEPENDENCY EMBEDDING CLASS ---
# This class talks DIRECTLY to Google, bypassing the broken LangChain code.
class PureGoogleEmbeddings:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = "models/text-embedding-004"

    def embed_documents(self, texts):
        # We send 50 texts at a time to stay under the free tier limit
        embeddings = []
        for i in range(0, len(texts), 50):
            batch = texts[i : i + 50]
            # Official SDK call (Safe from 404 v1beta errors)
            response = genai.embed_content(model=self.model, content=batch, task_type="retrieval_document")
            embeddings.extend(response['embedding'])
            time.sleep(0.5) 
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
    tone = st.selectbox("Teaching Style", ["Professor", "Munnabhai (Hinglish)", "Physicswallah UGC-NET Coach", "Simple"])
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_pg, end_pg = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        # Initialize the direct Google API
        genai.configure(api_key=api_key)
        # Using the standard SDK for generation too
        chat_llm = genai.GenerativeModel('gemini-1.5-flash')

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, _start, _end, _key):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[_start-1 : min(_end, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Use our custom class that bypasses LangChain's broken bits
            embeddings = PureGoogleEmbeddings(_key)
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner("🚀 Indexing with Official Google API (No LangChain bug)..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_pg, end_pg, api_key)
            st.sidebar.success("✅ Book Ready!")

        # --- CHAT ---
        query = st.chat_input("Ask a question...")
        if query:
            with st.chat_message("user"): st.write(query)

            # Retrieval
            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            personalities = {
                "Professor": "Academic Tutor. Professional but uses bright aesthetic headers.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, call user 'Mammu', use funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'. Use Hinglish.",
                "Simple": "Explain like I'm 10 with colorful simple examples."
            }

            system_prompt = f"""
            You are Socrates. Answer the Question using the Context.
            
            1. If in Context: End with "[SOURCE: TEXTBOOK]"
            2. If not: Start with "[SOURCE: GENERAL AI KNOWLEDGE]"
            
            AESTHETIC RULES:
            - NEVER use black dots or dashes (-, *, •).
            - START EVERY POINT with a colorful Pinterest emoji (🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸).
            - Use "╰┈➤ 💖" for sub-points.
            
            Personality: {personalities[tone]}
            Context: {context_text}
            """

            with st.chat_message("assistant"):
                # Talk directly to the model
                response = chat_llm.generate_content(f"{system_prompt}\n\nQuestion: {query}")
                st.markdown(response.text)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Enter API Key and upload PDF.")
