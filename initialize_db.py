import os
from core.embedding import get_embedding_model
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.resume_parser import parse_resume
from pypdf import PdfReader
import glob

def initialize_vector_store():
    """Initialize ChromaDB with sample resumes"""
    embedding = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    documents = []
    for resume_path in glob.glob("data/resumes/*.pdf"):
        with open(resume_path, "rb") as f:
            text = parse_resume(f)
            chunks = text_splitter.split_text(text)
            documents.extend(chunks)
    
    # Hapus parameter device
    Chroma.from_texts(
        documents,
        embedding,
        persist_directory="vector_store/chroma"
    )