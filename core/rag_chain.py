import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.retriever import get_retriever
from typing import List, Optional, Dict, Tuple
from core.scoring import ResumeScorer
from utils.resume_standardizer import ResumeStandardizer
from utils.name_extractor import NameExtractor
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Initialize components
llm = ChatGroq(
    temperature=0,
    model_name="deepseek-r1-distill-llama-70b",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=45
)

name_extractor = NameExtractor()
standardizer = ResumeStandardizer()

def _batch_process(func, items: List, batch_size: int = 5):
    """Helper for parallel processing"""
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        return list(executor.map(func, items))

def _get_candidate_name(resume_text: str, filename: str = "") -> str:
    """Wrapper for name extraction"""
    return name_extractor.extract_name_from_resume(resume_text, filename)

def resume_qa(resume_text: str, question: str, filename: str = "") -> str:
    """Enhanced resume Q&A with standardization"""
    try:
        std_resume = standardizer.standardize_resume(resume_text)
        candidate_name = _get_candidate_name(resume_text, filename)
        
        prompt = ChatPromptTemplate.from_template(
            """Answer this question about {name}'s resume:
            Question: {question}
            
            Resume Content:
            {resume_text}
            
            Rules:
            1. Refer to candidate as {name}
            2. Be specific about section references
            3. If unsure, say "Not specified in resume"
            4. Answer in Indonesian languange"""
        )
        
        chain = prompt | llm
        return chain.invoke({
            "question": question,
            "resume_text": std_resume[:3000],
            "name": candidate_name
        }).content
    except Exception:
        return "Unable to analyze resume"

def candidate_search(jd_text: str, resume_docs: Optional[List[Tuple[str, str]]] = None) -> str:
    """Optimized candidate search with name extraction"""
    retriever = get_retriever()
    
    if resume_docs is None:
        # If no resumes provided, use the retriever to find relevant ones
        relevant_docs = retriever.invoke(jd_text)
        processed = _batch_process(
            lambda doc: (
                standardizer.standardize_resume(doc.page_content),
                _get_candidate_name(doc.page_content, doc.metadata.get('filename', ''))
            ),
            relevant_docs
        )
    else:
        # Use the provided resumes
        processed = _batch_process(
            lambda data: (
                standardizer.standardize_resume(data[0]),
                _get_candidate_name(data[0], data[1])
            ),
            resume_docs
        )
    
    prompt = ChatPromptTemplate.from_template(
        """Job Description:
        {jd_text}
        
        Matching Candidates:
        {candidates}
        
        Provide in Indonesian language:
        1. Top 3 matches with rationale
        2. Missing skills
        3. Hiring recommendations"""
    )
    
    candidates_formatted = "\n\n".join(
        f"Candidate {i+1} ({name}):\n{text[:2000]}..."
        for i, (text, name) in enumerate(processed)
    )
    
    chain = prompt | llm
    return chain.invoke({
        "jd_text": jd_text,
        "candidates": candidates_formatted
    }).content

def candidate_profiling(resume_text: str, filename: str = "") -> str:
    """Standardized profiling with name"""
    std_resume = standardizer.standardize_resume(resume_text)
    level = standardizer.detect_resume_level(std_resume)
    candidate_name = _get_candidate_name(resume_text, filename)
    
    prompt = ChatPromptTemplate.from_template(
        """Analyze this resume for {name} ({level} level):
        {resume_text}
        
        Provide in Indonesian language:
        1. 5 key strengths
        2. 3 development areas  
        3. Recommended roles
        4. Interview questions"""
    )
    
    chain = prompt | llm
    return chain.invoke({
        "level": level,
        "resume_text": std_resume,
        "name": candidate_name
    }).content

def compare_candidates(resume_data: List[Tuple[str, str]], jd_text: Optional[str] = None) -> str:
    """Compare multiple candidates with names"""
    if not resume_data:
        return "No valid resumes to compare"
    
    # Process all resumes in parallel
    processed = _batch_process(
        lambda data: (
            standardizer.standardize_resume(data[0]),
            _get_candidate_name(data[0], data[1])
        ),
        resume_data
    )
    
    prompt_template = """Compare these {count} candidates{jd_context}:
    {candidates}
    
    Analysis in Indonesian languange and should include:
    1. Technical competence comparison
    2. Leadership potential  
    3. Cultural add
    4. Growth potential"""
    
    # Format with candidate names
    candidates_formatted = "\n\n---\n\n".join(
        f"{name}:\n{text[:2000]}..." 
        for text, name in processed
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template.format(
        count=len(processed),
        jd_context=" against job description" if jd_text else "",
        candidates=candidates_formatted
    ))
    
    chain = prompt | llm
    return chain.invoke({
        "jd_text": jd_text or ""
    }).content

def score_and_rank_candidates(resume_data: List[Tuple[str, str]], 
                            jd_text: Optional[str] = None,
                            criteria: Optional[Dict[str, int]] = None) -> Dict:
    """Scoring with name integration"""
    scorer = ResumeScorer(criteria or {
        "Technical Skills": 8,
        "Problem Solving": 7,
        "Leadership": 6,
        "Communication": 5,
        "Cultural Fit": 4
    })
    
    # Prepare input for scorer
    resume_texts = [data[0] for data in resume_data]
    filenames = [data[1] for data in resume_data]
    
    results = scorer.compare_resumes(resume_texts, jd_text)
    
    # Add names to results
    for i, result in enumerate(results["ranking"]):
        result["name"] = _get_candidate_name(resume_texts[i], filenames[i])
    
    return results