__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
# from dotenv import load_dotenv # DIHAPUS
from app.ui import render_ui, display_scoring_results
from app.controller import process_use_case
import pandas as pd
import plotly.express as px
import json
from utils.name_extractor import name_extractor
import torch
import logging

# Paksa semua komponen PyTorch ke CPU
torch.set_default_tensor_type(torch.FloatTensor)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Muat variabel lingkungan dari file .env
# load_dotenv() # DIHAPUS

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_errors.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    # Inisialisasi semua session state yang diperlukan
    if 'uploaded_resumes' not in st.session_state:
        st.session_state.uploaded_resumes = {
            "Candidate Search by Job Description": None,
            "Candidate Profiling / Resume QA": None,
            "Compare Multiple Candidates": [],
            "Compare with Scoring": []
        }
    if 'last_jd_text' not in st.session_state:
        st.session_state.last_jd_text = None
    if 'upload_errors' not in st.session_state:
        st.session_state.upload_errors = []
    if 'last_scoring_results' not in st.session_state:
        st.session_state.last_scoring_results = None
    if 'last_narrative_analysis' not in st.session_state:
        st.session_state.last_narrative_analysis = None
    if 'show_scoring_results' not in st.session_state:
        st.session_state.show_scoring_results = False
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = "General"
    if 'processed_flag' not in st.session_state:
        st.session_state.processed_flag = False

    st.set_page_config(page_title="RAG Resume Analyzer", layout="wide")
    st.title("📄 AI Resume Analyzer with Groq")
    
    logger.info("Starting application")
    use_case, inputs, question = render_ui()
    
    if use_case == "Compare with Scoring":
        logger.info("Processing Compare with Scoring use case")
        if inputs:
            with st.spinner("Memproses..."):
                logger.debug(f"Inputs received: {len(inputs['resume_data'])} resumes")
                results = process_use_case(use_case, inputs, question)
                if isinstance(results, dict) and "ranking" in results:
                    st.session_state["last_scoring_results"] = results
                    st.session_state["show_scoring_results"] = True
                    logger.info(f"Processing completed, results stored with {len(results['ranking'])} candidates")
                    st.subheader("Hasil Perbandingan Skor")
                    display_scoring_results(results)  # Tampilkan hasil langsung
                else:
                    st.error("Gagal memproses skor. Pastikan input valid.")
                    logger.error(f"Processing failed: {results}")
        else:
            if st.session_state.get("show_scoring_results") and st.session_state.get("last_scoring_results"):
                st.subheader("Hasil Perbandingan Skor Terakhir")
                display_scoring_results(st.session_state["last_scoring_results"])
            else:
                logger.info("No inputs provided for Compare with Scoring")
    
    else:
        if inputs:
            with st.spinner("Processing..."):
                logger.info(f"Processing use case: {use_case}")
                results = process_use_case(use_case, inputs, question)
                st.subheader("Hasil")
                st.write(results)
                logger.info(f"Completed processing for {use_case}")

if __name__ == "__main__":
    main()
