import os
from core.embedding import get_embedding_model
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.resume_parser import parse_resume
import glob

def initialize_vector_store():
    """Inisialisasi ChromaDB dengan contoh resume menggunakan ID unik"""
    embedding = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    all_chunks = []
    all_ids = []
    for resume_path in glob.glob("data/resumes/*.pdf"):
        filename = os.path.basename(resume_path)
        with open(resume_path, "rb") as f:
            text, _ = parse_resume(f, filename)
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
    
    Chroma.from_texts(
        all_chunks,
        embedding,
        persist_directory="vector_store/chroma",
        ids=all_ids
    )
