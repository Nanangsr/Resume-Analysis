from typing import List
from langchain.prompts import ChatPromptTemplate
from utils.resume_standardizer import ResumeStandardizer

def compare_resumes(resume_texts: List[str], llm) -> str:
    """Bandingkan beberapa resume yang sudah distandardisasi"""
    standardizer = ResumeStandardizer()
    
    # Standarisasi semua resume terlebih dahulu
    standardized_resumes, _ = standardizer.standardize_multiple(resume_texts)
    
    prompt = ChatPromptTemplate.from_template(
        """Bandingkan resume-resume yang sudah distandardisasi ini dan berikan rekomendasi:
        1. Ringkasan perbandingan
        2. Kandidat terbaik untuk peran teknis
        3. Kandidat terbaik untuk peran kepemimpinan
        4. Pertimbangan kecocokan budaya
        
        Resume yang Sudah Distandardisasi:
        {resumes}
        
        Fokus pada:
        - Kesesuaian keterampilan
        - Relevansi pengalaman
        - Potensi kepemimpinan
        - Progresi karier
        """
    )
    
    formatted_resumes = "\n\n---\n\n".join(
        f"Kandidat {i+1} (Standardisasi):\n{text}" 
        for i, text in enumerate(standardized_resumes))
    
    chain = prompt | llm
    return chain.invoke({"resumes": formatted_resumes}).content
