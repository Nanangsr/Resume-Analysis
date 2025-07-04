from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
import torch
import streamlit as st

def get_embedding_model():
    """Load optimized embedding model with session state caching"""
    if 'embedding_model' not in st.session_state:
        # DIUBAH: Gunakan st.secrets.get() untuk mengambil nama model dengan fallback
        model_name = st.secrets.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tambahkan validasi untuk Streamlit Cloud (yang hanya punya CPU)
        if 'st' in globals() and hasattr(st, 'secrets'):
            device = 'cpu'
            
        st.session_state.embedding_model = SentenceTransformerEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device, 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    return st.session_state.embedding_model
