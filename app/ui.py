import streamlit as st
from typing import Tuple, Union, Dict, List, Optional
from utils.resume_parser import parse_resume, parse_uploaded_folder
from utils.jd_parser import parse_jd
from utils.name_extractor import NameExtractor  # Updated import
import pandas as pd
import plotly.express as px

# Initialize NameExtractor
name_extractor = NameExtractor()

# Initialize session state
if 'uploaded_resumes' not in st.session_state:
    st.session_state.uploaded_resumes = {
        "Candidate Search by Job Description": None,
        "Candidate Profiling / Resume QA": None,
        "Compare Multiple Candidates": [],
        "Compare with Scoring": []
    }

if 'upload_errors' not in st.session_state:
    st.session_state.upload_errors = []

if 'last_jd_text' not in st.session_state:
    st.session_state.last_jd_text = None

def render_ui() -> Tuple[str, Union[Dict, List, None], Optional[str]]:
    """Render Streamlit UI and collect inputs"""
    # Initialize default values
    use_case = "Candidate Search by Job Description"
    inputs = None
    question = None
    
    # Ensure all required keys exist in session state
    if 'uploaded_resumes' not in st.session_state:
        st.session_state.uploaded_resumes = {
            "Candidate Search by Job Description": None,
            "Candidate Profiling / Resume QA": None,
            "Compare Multiple Candidates": [],
            "Compare with Scoring": []
        }
    
    if 'upload_errors' not in st.session_state:
        st.session_state.upload_errors = []
    
    if 'last_jd_text' not in st.session_state:
        st.session_state.last_jd_text = None
    
    st.sidebar.header("Use Cases")
    use_case = st.sidebar.radio(
        "Select Use Case",
        ["Candidate Search by Job Description", 
         "Candidate Profiling / Resume QA",
         "Compare Multiple Candidates",
         "Compare with Scoring"]
    )
    
    if use_case == "Candidate Search by Job Description":
        st.header("🔍 Candidate Search by Job Description")
        jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
        if jd_file:
            try:
                jd_text, error = parse_jd(jd_file)
                if error:
                    st.error(f"Error processing JD: {error}")
                else:
                    inputs = {"jd_text": jd_text}
                    st.session_state.last_jd_text = jd_text
            except Exception as e:
                st.error(f"Failed to parse JD: {str(e)}")
                
    elif use_case == "Candidate Profiling / Resume QA":
        st.header("📝 Candidate Profiling / Resume QA")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            resume_file = st.file_uploader(
                "Upload Resume (PDF/DOCX)", 
                type=["pdf", "docx"],
                key="profiling_uploader"
            )
        with col2:
            if st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] and st.button("Clear Resume"):
                st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] = None
                
        if resume_file:
            try:
                resume_text, error = parse_resume(resume_file, resume_file.name)
                if error:
                    st.error(f"Error processing resume: {error}")
                else:
                    st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"] = resume_text
            except Exception as e:
                st.error(f"Failed to parse resume: {str(e)}")
            
        if st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"]:
            inputs = {
                "resume_text": st.session_state.uploaded_resumes["Candidate Profiling / Resume QA"],
                "filename": resume_file.name if resume_file else ""
            }
            
            st.subheader("Ask Questions About the Resume")
            question = st.text_input("Enter your question about the resume")
            
    elif use_case == "Compare Multiple Candidates":
        st.header("📊 Compare Multiple Candidates")
        
        if st.session_state.last_jd_text:
            st.info("Comparing candidates against the last uploaded job description")
            with st.expander("View Job Description"):
                st.write(st.session_state.last_jd_text)
        
        upload_option = st.radio(
            "Upload option:", 
            ["Multiple Files", "Folder (ZIP)"],
            key="compare_upload_option"
        )
        
        if upload_option == "Multiple Files":
            resume_files = st.file_uploader(
                "Upload Resumes (PDF/DOCX)", 
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="compare_multiple_files"
            )
            if resume_files:
                valid_resumes = []
                error_messages = []
                for file in resume_files:
                    text, error = parse_resume(file, file.name)
                    if text:
                        valid_resumes.append((text, file.name))  # Store both text and filename
                    if error:
                        error_messages.append(error)
                
                st.session_state.uploaded_resumes["Compare Multiple Candidates"] = valid_resumes
                st.session_state.upload_errors = error_messages
        else:
            uploaded_folder = st.file_uploader(
                "Upload Folder (as ZIP)", 
                type=["zip"],
                accept_multiple_files=False,
                key="compare_folder_upload"
            )
            if uploaded_folder:
                valid_resumes, error_messages = parse_uploaded_folder(uploaded_folder)
                # For ZIP files, we don't have individual filenames
                st.session_state.uploaded_resumes["Compare Multiple Candidates"] = [(text, "") for text in valid_resumes]
                st.session_state.upload_errors = error_messages
        
        current_resumes = st.session_state.uploaded_resumes["Compare Multiple Candidates"]
        if current_resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.success(f"Successfully uploaded {len(current_resumes)} resumes")
            with col2:
                if st.button("Clear All Resumes", key="clear_compare_resume"):
                    st.session_state.uploaded_resumes["Compare Multiple Candidates"] = []
                    st.session_state.upload_errors = []
                    st.rerun()
            
        if st.session_state.upload_errors:
            st.warning("Some files could not be processed:")
            for error in st.session_state.upload_errors:
                st.error(error)
        
        if current_resumes and len(current_resumes) >= 2:
            inputs = {
                "resume_texts": [resume[0] for resume in current_resumes],
                "filenames": [resume[1] for resume in current_resumes]
            }

    elif use_case == "Compare with Scoring":
        st.header("📈 Compare with Scoring")

        # Add criteria configuration
        with st.expander("⚙️ Scoring Criteria Configuration", expanded=True):
            st.write("Configure the weights for each scoring criterion (1-10):")
            criteria = {}
            cols = st.columns(3)
            default_criteria = {
                "Technical Skills": 8,
                "Education": 6,
                "Work Experience": 9,
                "Leadership": 7,
                "Communication": 6
            }
            for i, criterion in enumerate(default_criteria.keys()):
                with cols[i % 3]:
                    criteria[criterion] = st.slider(
                        criterion, 1, 10, default_criteria[criterion],
                        key=f"score_criteria_{criterion}"
                    )
        
        upload_option = st.radio(
            "Upload option:", 
            ["Multiple Files", "Folder (ZIP)"],
            key="score_upload_option"
        )
        
        if upload_option == "Multiple Files":
            resume_files = st.file_uploader(
                "Upload Resumes (PDF/DOCX)", 
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="score_multiple_files"
            )
            if resume_files:
                valid_resumes = []
                error_messages = []
                for file in resume_files:
                    text, error = parse_resume(file, file.name)
                    if text:
                        valid_resumes.append((text, file.name))  # Store both text and filename
                    if error:
                        error_messages.append(error)
                
                st.session_state.uploaded_resumes["Compare with Scoring"] = valid_resumes
                st.session_state.upload_errors = error_messages
        else:
            uploaded_folder = st.file_uploader(
                "Upload Folder (as ZIP)", 
                type=["zip"],
                accept_multiple_files=False,
                key="score_folder_upload"
            )
            if uploaded_folder:
                valid_resumes, error_messages = parse_uploaded_folder(uploaded_folder)
                # For ZIP files, we don't have individual filenames
                st.session_state.uploaded_resumes["Compare with Scoring"] = [(text, "") for text in valid_resumes]
                st.session_state.upload_errors = error_messages
        
        current_resumes = st.session_state.uploaded_resumes.get("Compare with Scoring", [])
        
        if current_resumes:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.success(f"Successfully uploaded {len(current_resumes)} resumes")
            with col2:
                if st.button("Clear All Resumes", key="clear_score_resumes"):
                    st.session_state.uploaded_resumes["Compare with Scoring"] = []
                    st.session_state.upload_errors = []
                    st.rerun()
        
        if st.session_state.upload_errors:
            st.warning("Some files could not be processed:")
            for error in st.session_state.upload_errors:
                st.error(error)
        
        if current_resumes and len(current_resumes) >= 2:
            # Generate candidate names
            candidate_names = []
            for i, (resume_text, filename) in enumerate(current_resumes):
                name = name_extractor.extract_name_from_resume(resume_text, filename)
                candidate_names.append(name or f"Candidate {i+1}")
            st.session_state.candidate_names = candidate_names
            
            inputs = {
                "resume_texts": [resume[0] for resume in current_resumes],
                "filenames": [resume[1] for resume in current_resumes],
                "criteria": criteria
            }
    
    return use_case, inputs, question