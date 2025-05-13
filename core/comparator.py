from typing import List
from langchain.prompts import ChatPromptTemplate
from utils.resume_standardizer import ResumeStandardizer

def compare_resumes(resume_texts: List[str], llm) -> str:
    """Compare multiple standardized resumes"""
    standardizer = ResumeStandardizer()
    
    # Standardize all resumes first
    standardized_resumes, _ = standardizer.standardize_multiple(resume_texts)
    
    prompt = ChatPromptTemplate.from_template(
        """Compare these standardized resumes and provide recommendations:
        1. Comparative summary
        2. Best candidate for technical roles
        3. Best candidate for leadership roles
        4. Cultural fit considerations
        
        Standardized Resumes:
        {resumes}
        
        Focus on:
        - Skills alignment
        - Experience relevance
        - Leadership potential
        - Career progression
        """
    )
    
    formatted_resumes = "\n\n---\n\n".join(
        f"Candidate {i+1} (Standardized):\n{text}" 
        for i, text in enumerate(standardized_resumes))
    
    chain = prompt | llm
    return chain.invoke({"resumes": formatted_resumes}).content