from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os
import torch

def get_embedding_model():
    """Optimized embedding model loader with caching"""
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Auto-configure device with fallback
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return SentenceTransformerEmbeddings(
        model_name=model_name,
        model_kwargs={
            'device': device,
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,
        }
    )