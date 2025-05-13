import os
from core.embedding import get_embedding_model
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.resume_parser import parse_resume
from pypdf import PdfReader
import glob

def initialize_vector_store():
    """Inisialisasi ChromaDB dengan contoh resume"""
    embedding = get_embedding_model()
    # Pembagi teks untuk memecah dokumen menjadi bagian-bagian kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    documents = []
    # Proses semua file PDF di direktori data/resumes
    for resume_path in glob.glob("data/resumes/*.pdf"):
        with open(resume_path, "rb") as f:
            text = parse_resume(f)
            chunks = text_splitter.split_text(text)
            documents.extend(chunks)
    
    # Buat vector store dari teks-teks yang sudah diproses
    Chroma.from_texts(
        documents,
        embedding,
        persist_directory="vector_store/chroma"
    )
