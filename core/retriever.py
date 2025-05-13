from langchain_community.vectorstores import Chroma
from core.embedding import get_embedding_model
import os

def get_retriever():
    """Inisialisasi dan kembalikan retriever"""
    embedding = get_embedding_model()
    vector_store = Chroma(
        persist_directory="vector_store/chroma",
        embedding_function=embedding
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})
