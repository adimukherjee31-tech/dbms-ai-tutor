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
# 'pro' is better for the complex pedagogical explanations you nee
