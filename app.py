import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# CSS: Force colorful emojis and remove black bullets
st.markdown("""
    <style>
    ul, li { list-style-type: none !important; padding-left: 0 !important; }
    .stMarkdown p, .stMarkdown li { 
        font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

# --- CUSTOM ROBUST EMBEDDINGS (THE FIX) ---
class StableGoogleEmbeddings(Embeddings):
    def __init__(self, api_key):
        # Force the use of 'v1' stable API instead of 'v1beta'
        genai.configure(api_key=api_key, transport='rest')
        self.model = "models/text-embedding-004"
        
        # Verify if 004 exists for this key, otherwise fallback to 001
        try:
            genai.get_model(self.model)
        except:
            self.model = "models/embedding-001"

    def embed_documents(self, texts):
        return [genai.embed_content(model=self.model, content=t, task_type="retrieval_document")["embedding"] for t in texts]

    def embed_query(self, text):
        return genai.embed_content(model=self.model, content=text, task_type="retrieval_query")["embedding"]

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload Textbook/Notes (PDF)", type="pdf")
    
    st.header("2. Study Settings")
    tone = st.selectbox("Teaching Style", [
        "Professor", 
        "Munnabhai (Hinglish)", 
        "Physicswallah UGC-NET Coach", 
        "Simple"
    ])
    
    st.info("🎨 **Pinterest Stickers**: Active & Colorful.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 100))
    start_page, end_page = page_range

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        # Initialize direct Google SDK
        genai.configure(api_key=api_key)
        
        # Initialize Chat Model (Gemini 1.5 Flash is best for this)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key, 
            temperature=0.7
        )

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, start_pg, end_pg, _api_key):
            # USE OUR NEW STABLE EMBEDDING CLASS
            embeddings = StableGoogleEmbeddings(_api_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[start_pg-1 : min(end_pg, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Create FAISS DB
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner(f"🚀 Speed-indexing pages {start_page} to {end_page}..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_page, end_page, api_key)
            st.sidebar.success(f"✅ Ready! Using {StableGoogleEmbeddings(api_key).model}")

        # --- CHAT ---
        query = st.chat_input("Ask a question from this section...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            # Retrieve context
            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Professional headers, clear points.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish and call user 'Mammu'.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'.",
                "Simple": "Explain like I'm 10 with very colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. 
            
            GROUNDING:
            - If found in Context: Explain it. End with "[SOURCE: TEXTBOOK]"
            - If not: Use General Knowledge. Start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            AESTHETIC & COLOR RULES (STRICT):
            - NEVER use black/gray dots, dashes, or stars (•, -, *, 🔘).
            - YOU MUST start EVERY new point with a DIFFERENT bright, colorful Pinterest emoji.
            - USE THESE ONLY: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦, 🍭, 🎠.
            - SUB-POINTS: Use "╰┈➤ 💖" followed by a different colorful emoji.
            - Ensure the answer looks like high-vibrancy, aesthetic study notes.
            
            Context: {context}
            Style: {personality}
            Question: {question}
            
            Answer:""")

            chain = prompt | llm | StrOutputParser()

            with st.chat_message("assistant"):
                response = chain.invoke({"personality": styles[tone], "context": context_text, "question": query})
                st.markdown(response)

    except Exception as e:
        st.error(f"System Error: {e}")
else:
    st.warning("Enter API Key and upload PDF to start.")
