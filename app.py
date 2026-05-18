import streamlit as st
import os
import tempfile
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIG ---
st.set_page_config(page_title="Socrates AI Tutor", layout="wide", page_icon="🎓")

# CSS HACK: Force colorful emojis and hide standard black dots
st.markdown("""
    <style>
    /* Hide standard bullets */
    ul, li { list-style-type: none !important; padding-left: 0 !important; margin-left: 0 !important; }
    /* Force emojis to render in color and slightly larger */
    .stMarkdown p { 
        font-size: 1.15rem !important; 
        line-height: 1.7 !important;
        font-family: 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'Segoe UI Symbol', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 Socrates: Pedagogical AI Tutor")

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
    
    st.info("🎨 Pinterest Mode: Colorful Stickers Enabled.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 200))
    start_page, end_page = page_range

# --- SMART MODEL DISCOVERY ---
def get_working_models(api_key):
    """Dynamically finds the best available chat and embedding models for this API Key"""
    try:
        genai.configure(api_key=api_key)
        all_models = [m.name for m in genai.list_models()]
        
        # 1. Find Chat Model
        chat_model = "gemini-1.5-flash" # Default
        for m in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']:
            if m in all_models:
                chat_model = m.replace('models/', '')
                break
        
        # 2. Find Embedding Model (The fix for your 404 error)
        embed_model = "models/embedding-001" # Default
        embedding_options = [
            'models/text-embedding-004', 
            'models/embedding-001', 
            'models/text-embedding-004'
        ]
        for e in embedding_options:
            if e in all_models:
                embed_model = e
                break
                
        return chat_model, embed_model
    except:
        return "gemini-1.5-flash", "models/embedding-001"

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        chat_name, embed_name = get_working_models(api_key)
        st.sidebar.success(f"Models: {chat_name} + {embed_name.split('/')[-1]}")

        llm = ChatGoogleGenerativeAI(model=chat_name, google_api_key=api_key, temperature=0.7)

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, start_pg, end_pg, _api_key, _embed_model_name):
            embeddings = GoogleGenerativeAIEmbeddings(
                model=_embed_model_name, 
                google_api_key=_api_key
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_content)
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            all_docs = loader.load()
            docs = all_docs[start_pg-1 : min(end_pg, len(all_docs))]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            db = FAISS.from_documents(chunks, embeddings)
            os.remove(tmp_path)
            return db

        with st.spinner("🚀 Reading your materials..."):
            # Passing the discovered embed_name here
            vector_db = get_vector_db(uploaded_file.getvalue(), start_page, end_page, api_key, embed_name)

        # --- CHAT ---
        query = st.chat_input("Ask anything from this section...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Clear headings and aesthetic markers.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, 'Mammu', funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'.",
                "Simple": "Explain like I'm 10 with colorful simple examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. 
            
            GROUNDING:
            - If found in Context: Explain and end with "[SOURCE: TEXTBOOK]"
            - If not: Use general knowledge and start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            PINTEREST AESTHETIC RULES (STRICT):
            - NEVER use black dots, dashes, or stars (•, -, *) for lists. 
            - START EVERY POINT with a colorful high-vibrancy emoji (🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡, 🍓, 🍦).
            - Use a DIFFERENT colorful emoji for every point to make it look like a vibrant digital scrapbook.
            - Use ╰┈➤ 💖 for sub-points.
            - The entire output MUST be bright and visually pleasing.
            
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
