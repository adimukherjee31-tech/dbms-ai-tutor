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

# CSS HACK: Force emojis to be large and colorful, and hide standard black bullets
st.markdown("""
    <style>
    ul { list-style-type: none !important; margin-left: 0px !important; padding-left: 0px !important; }
    li { list-style-type: none !important; margin-bottom: 10px; }
    .stMarkdown p { font-size: 1.1rem; line-height: 1.6; }
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
    
    st.info("🌈 **Color Mode**: Vivid Aesthetic Stickers Enabled.")
    page_range = st.slider("Select Page Range", 1, 2500, (1, 200))
    start_page, end_page = page_range

# --- AUTO-DETECT MODELS ---
def get_working_model(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        clean_models = [m.replace('models/', '') for m in models]
        for preferred in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
            if preferred in clean_models: return preferred
        return clean_models[0] if clean_models else "gemini-1.5-flash"
    except Exception: return "gemini-1.5-flash"

# --- PROCESSING ---
if api_key and uploaded_file:
    try:
        active_model = get_working_model(api_key)
        st.sidebar.success(f"Connected to: {active_model}")

        llm = ChatGoogleGenerativeAI(model=active_model, google_api_key=api_key, temperature=0.7)

        @st.cache_resource(show_spinner=False)
        def get_vector_db(file_content, start_pg, end_pg, _api_key):
            # Try the most modern embedding model first
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-04", google_api_key=_api_key)
            except:
                # Fallback to the older stable version if 04 fails
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=_api_key)
            
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

        with st.spinner("🚀 Reading and coloring your notes..."):
            vector_db = get_vector_db(uploaded_file.getvalue(), start_page, end_page, api_key)

        # --- CHAT ---
        query = st.chat_input("Ask anything from this section...")
        
        if query:
            with st.chat_message("user"): st.write(query)

            context_docs = vector_db.similarity_search(query, k=5)
            context_text = "\n\n".join([d.page_content for d in context_docs])

            styles = {
                "Professor": "Academic Tutor. Use clear headings and aesthetic markers.",
                "Munnabhai (Hinglish)": "Munnabhai style. Use Hinglish, 'Mammu', and funny analogies.",
                "Physicswallah UGC-NET Coach": "High-energy coach. 'Hello Baccho!', 'Selection rukna nahi chahiye!'.",
                "Simple": "Explain like I'm 10 with colorful examples."
            }

            prompt = ChatPromptTemplate.from_template("""
            You are Socrates, a pedagogical tutor. 
            
            GROUNDING:
            - If found in Context: Explain and end with "[SOURCE: TEXTBOOK]"
            - If not: Use general knowledge and start with "[SOURCE: GENERAL AI KNOWLEDGE]"

            AESTHETIC RULES (VERY IMPORTANT):
            - NEVER use black dots, dashes, or asterisks (•, -, *) for lists.
            - Start EVERY new point with a unique, BRIGHT, COLORFUL Pinterest emoji.
            - Use these specific vibrant emojis: 🌈, 🍭, 🎀, ✨, 🎨, 🌟, 🍬, 🦋, 🦄, 🎈, 🧁, 🌸, 🎡.
            - Use "╰┈➤ 💖" for sub-points.
            - The entire output should look like a VIBRANT, COLORFUL digital scrapbook.
            
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
