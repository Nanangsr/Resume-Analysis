from typing import Dict, List, Union, Optional
from core.rag_chain import (
    candidate_search,
    candidate_profiling,
    compare_candidates,
    resume_qa
)
import streamlit as st
from core.scoring import score_and_rank_candidates  

def process_use_case(use_case: str, inputs: Union[Dict, List], question: Optional[str] = None) -> Union[str, Dict]:
    """Arahkan use case ke fungsi yang sesuai"""
    if use_case == "Candidate Search by Job Description":
        return candidate_search(inputs["jd_text"])
    elif use_case == "Candidate Profiling / Resume QA":
        if question:
            return resume_qa(inputs["resume_text"], question)
        return candidate_profiling(inputs["resume_text"])
    elif use_case == "Compare Multiple Candidates":
        jd_text = st.session_state.get("last_jd_text", None)
        return compare_candidates(inputs["resume_texts"], jd_text)
    elif use_case == "Compare with Scoring":
        jd_text = st.session_state.get("last_jd_text", None)
        results = score_and_rank_candidates(
            inputs["resume_texts"],
            jd_text,
            inputs.get("criteria")
        )
        # Pastikan results memiliki struktur yang benar
        if isinstance(results, dict) and "ranking" in results:
            return results
        return {"error": "Hasil scoring tidak valid"}  # Fallback
    else:
        return "Use case tidak valid"
