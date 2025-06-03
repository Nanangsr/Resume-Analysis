from typing import Dict, List, Union, Optional, Tuple
import asyncio
from core.rag_chain import ResumeRagChain
from utils.resume_standardizer import ResumeStandardizer
import streamlit as st
import logging
import traceback

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
logger.setLevel(logging.DEBUG)  # Set ke DEBUG untuk lebih banyak informasi

def get_resume_data(inputs: Union[Dict, List, str, Tuple]) -> Tuple[str, str]:
    """Extract standardized resume data from inputs with robust handling"""
    logger.debug(f"Extracting resume data from inputs: {type(inputs)}")
    try:
        if isinstance(inputs, dict):
            resume_text = inputs.get("resume_text", "") or str(inputs)
            filename = inputs.get("filename", "")
            logger.debug(f"Dict input - Resume text length: {len(resume_text)}, Filename: {filename}")
            return (resume_text.strip(), filename.strip())
        elif isinstance(inputs, tuple):
            if len(inputs) >= 2:
                logger.debug(f"Tuple input - Text: {inputs[0][:50]}..., Filename: {inputs[1]}")
                return (str(inputs[0]).strip(), str(inputs[1]).strip())
            elif len(inputs) == 1:
                logger.warning(f"Incomplete tuple input: {inputs}")
                return (str(inputs[0]).strip(), "")
            else:
                logger.warning(f"Empty tuple input: {inputs}")
                return ("", "")
        elif isinstance(inputs, str):
            logger.debug(f"String input - Text length: {len(inputs)}")
            return (inputs.strip(), "")
        else:
            logger.warning(f"Invalid input format: {inputs}")
            return (str(inputs).strip(), "")
    except Exception as e:
        logger.error(f"Error in get_resume_data: {str(e)}\n{traceback.format_exc()}")
        return ("", "")

def process_use_case(use_case: str, inputs: Union[Dict, List], question: Optional[str] = None) -> Union[str, Dict]:
    """Route use cases to appropriate handlers with improved consistency"""
    logger.info(f"Starting process_use_case: {use_case}")
    try:
        domain = inputs.get("domain", st.session_state.get("selected_domain", "general")) if isinstance(inputs, dict) else st.session_state.get("selected_domain", "general")
        logger.info(f"Processing use case '{use_case}' with domain: {domain}")

        rag_chain = ResumeRagChain(domain=domain)
        logger.debug(f"RagChain initialized for domain: {domain}")

        if use_case == "Candidate Search by Job Description":
            if not isinstance(inputs, dict) or not inputs.get("jd_text"):
                error_msg = "Input deskripsi pekerjaan tidak valid"
                logger.error(error_msg)
                return {"error": error_msg}
            logger.info("Running candidate_search")
            return asyncio.run(rag_chain.candidate_search(inputs["jd_text"]))
        
        elif use_case == "Candidate Profiling / Resume QA":
            resume_text, filename = get_resume_data(inputs)
            if not resume_text:
                error_msg = "Teks resume kosong atau tidak valid"
                logger.error(error_msg)
                return {"error": error_msg}
            logger.info(f"Processing resume QA/Profiling for file: {filename}")
            if question:
                return asyncio.run(rag_chain.resume_qa(resume_text, question, filename))
            return asyncio.run(rag_chain.candidate_profiling(resume_text, filename))
        
        elif use_case in ("Compare Multiple Candidates", "Compare with Scoring"):
            jd_text = st.session_state.get("last_jd_text", "")
            logger.debug(f"JD text length: {len(jd_text)}")
            if isinstance(inputs, dict):
                resume_data = inputs.get("resume_data", [])
            else:
                resume_data = inputs
                
            if not resume_data:
                error_msg = "Tidak ada data resume yang valid"
                logger.error(error_msg)
                return {"error": error_msg}
                
            validated_resume_data = []
            for data in resume_data:
                if not isinstance(data, tuple) or len(data) < 2:
                    logger.warning(f"Data resume tidak valid: {data}")
                    resume_text = data[0] if isinstance(data, tuple) and len(data) > 0 else str(data)
                    validated_resume_data.append((resume_text.strip(), ""))
                else:
                    validated_resume_data.append((data[0].strip(), data[1].strip()))
            logger.info(f"Validated {len(validated_resume_data)} resumes")
                
            if use_case == "Compare Multiple Candidates":
                logger.info("Running compare_candidates")
                return asyncio.run(rag_chain.compare_candidates(validated_resume_data, jd_text))
            
            elif use_case == "Compare with Scoring":
                criteria = inputs.get("criteria") if isinstance(inputs, dict) else None
                if not criteria:
                    standardizer = ResumeStandardizer(domain=domain)
                    criteria = standardizer.get_domain_specific_criteria()
                    logger.debug(f"Using default criteria: {criteria}")
                
                logger.info(f"Processing scoring with {len(validated_resume_data)} candidates, domain: {domain}")
                
                try:
                    results = rag_chain.score_and_rank_candidates(
                        validated_resume_data,
                        jd_text,
                        criteria
                    )
                    
                    if "error" in results:
                        logger.error(f"Scoring error: {results['error']}")
                        return {"error": results['error']}
                    
                    narrative = results.get("narrative_analysis", "")
                    logger.debug(f"Initial narrative length: {len(narrative)}")
                    if not narrative or len(narrative.strip()) < 50:
                        logger.warning("Narrative missing or too short, attempting to generate")
                        try:
                            narrative = asyncio.run(rag_chain.generate_llm_narrative_analysis(results, jd_text))
                            if narrative and len(narrative.strip()) > 50:
                                results["narrative_analysis"] = narrative
                                logger.info(f"Generated narrative successfully: {len(narrative)} chars")
                            else:
                                logger.warning("Failed to generate adequate narrative")
                        except Exception as e:
                            logger.error(f"Error generating narrative: {str(e)}\n{traceback.format_exc()}")
                    
                    # Simpan ke session state untuk memastikan hasil tersedia
                    st.session_state["last_narrative_analysis"] = narrative
                    st.session_state["last_scoring_results"] = results
                    st.session_state["show_scoring_results"] = True
                    
                    logger.info(f"Session state updated - narrative: {len(narrative)} chars, results: {len(results['ranking'])} candidates")
                    logger.debug(f"Show scoring results flag set to: {st.session_state['show_scoring_results']}")
                    return results
                
                except Exception as e:
                    error_msg = f"Kesalahan pemrosesan untuk {use_case}: {str(e)}"
                    logger.error(f"{error_msg}\n{traceback.format_exc()}")
                    return {"error": error_msg}
                
        error_msg = "Use case tidak valid"
        logger.error(error_msg)
        return {"error": error_msg}
    
    except ValueError as e:
        error_msg = f"Kesalahan input untuk {use_case}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Kesalahan pemrosesan untuk {use_case}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}
