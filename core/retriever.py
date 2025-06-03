from langchain_community.vectorstores import Chroma
from core.embedding import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import streamlit as st

def get_retriever():
    """Initialize and return retriever with session state caching"""
    if 'retriever' not in st.session_state:
        embedding = get_embedding_model()
        vector_store = Chroma(
            persist_directory="vector_store/chroma",
            embedding_function=embedding
        )
        st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return st.session_state.retriever

def add_resume_to_vector_store(resume_text: str, filename: str):
    """Add new resume to vector store with unique ID"""
    embedding = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(resume_text)
    ids = [f"{filename}_chunk{i}" for i in range(len(chunks))]
    
    vector_store = Chroma(
        persist_directory="vector_store/chroma",
        embedding_function=embedding
    )
    vector_store.add_texts(texts=chunks, ids=ids)
