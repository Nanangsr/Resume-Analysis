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

# Inisialisasi komponen
llm = ChatGroq(
    temperature=0,
    model_name="deepseek-r1-distill-llama-70b",
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=45
)

name_extractor = NameExtractor()
standardizer = ResumeStandardizer()

def _batch_process(func, items: List, batch_size: int = 5):
    """Helper untuk pemrosesan paralel"""
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        return list(executor.map(func, items))

def _get_candidate_name(resume_text: str, filename: str = "") -> str:
    """Wrapper untuk ekstraksi nama"""
    return name_extractor.extract_name_from_resume(resume_text, filename)

def resume_qa(resume_text: str, question: str, filename: str = "") -> str:
    """Q&A resume yang ditingkatkan dengan standardisasi"""
    try:
        std_resume = standardizer.standardize_resume(resume_text)
        candidate_name = _get_candidate_name(resume_text, filename)
        
        prompt = ChatPromptTemplate.from_template(
            """Jawab pertanyaan tentang resume {name}:
            Pertanyaan: {question}
            
            Konten Resume:
            {resume_text}
            
            Aturan:
            1. Sebut kandidat sebagai {name}
            2. Spesifik tentang referensi bagian
            3. Jika tidak yakin, katakan "Tidak disebutkan di resume"
            4. Jawab dalam bahasa Indonesia"""
        )
        
        chain = prompt | llm
        return chain.invoke({
            "question": question,
            "resume_text": std_resume[:3000],
            "name": candidate_name
        }).content
    except Exception:
        return "Tidak dapat menganalisis resume"

def candidate_search(jd_text: str, resume_docs: Optional[List[Tuple[str, str]]] = None) -> str:
    """Pencarian kandidat yang dioptimalkan dengan ekstraksi nama"""
    retriever = get_retriever()
    
    if resume_docs is None:
        # Jika tidak ada resume yang diberikan, gunakan retriever untuk mencari yang relevan
        relevant_docs = retriever.invoke(jd_text)
        processed = _batch_process(
            lambda doc: (
                standardizer.standardize_resume(doc.page_content),
                _get_candidate_name(doc.page_content, doc.metadata.get('filename', ''))
            ),
            relevant_docs
        )
    else:
        # Gunakan resume yang diberikan
        processed = _batch_process(
            lambda data: (
                standardizer.standardize_resume(data[0]),
                _get_candidate_name(data[0], data[1])
            ),
            resume_docs
        )
    
    prompt = ChatPromptTemplate.from_template(
        """Deskripsi Pekerjaan:
        {jd_text}
        
        Kandidat yang Cocok:
        {candidates}
        
        Berikan dalam bahasa Indonesia:
        1. 3 kecocokan teratas dengan alasan
        2. Keterampilan yang kurang
        3. Rekomendasi perekrutan"""
    )
    
    candidates_formatted = "\n\n".join(
        f"Kandidat {i+1} ({name}):\n{text[:2000]}..."
        for i, (text, name) in enumerate(processed)
    )
    
    chain = prompt | llm
    return chain.invoke({
        "jd_text": jd_text,
        "candidates": candidates_formatted
    }).content

def candidate_profiling(resume_text: str, filename: str = "") -> str:
    """Profil standar dengan nama"""
    std_resume = standardizer.standardize_resume(resume_text)
    level = standardizer.detect_resume_level(std_resume)
    candidate_name = _get_candidate_name(resume_text, filename)
    
    prompt = ChatPromptTemplate.from_template(
        """Analisis resume ini untuk {name} (level {level}):
        {resume_text}
        
        Berikan dalam bahasa Indonesia:
        1. 5 kekuatan utama
        2. 3 area pengembangan  
        3. Peran yang direkomendasikan
        4. Pertanyaan wawancara"""
    )
    
    chain = prompt | llm
    return chain.invoke({
        "level": level,
        "resume_text": std_resume,
        "name": candidate_name
    }).content

def compare_candidates(resume_data: List[Tuple[str, str]], jd_text: Optional[str] = None) -> str:
    """Bandingkan beberapa kandidat dengan nama"""
    if not resume_data:
        return "Tidak ada resume yang valid untuk dibandingkan"
    
    # Proses semua resume secara paralel
    processed = _batch_process(
        lambda data: (
            standardizer.standardize_resume(data[0]),
            _get_candidate_name(data[0], data[1])
        ),
        resume_data
    )
    
    prompt_template = """Bandingkan {count} kandidat{jd_context}:
    {candidates}
    
    Analisis dalam bahasa Indonesia harus mencakup:
    1. Perbandingan kompetensi teknis
    2. Potensi kepemimpinan  
    3. Nilai tambah budaya
    4. Potensi pengembangan"""
    
    # Format dengan nama kandidat
    candidates_formatted = "\n\n---\n\n".join(
        f"{name}:\n{text[:2000]}..." 
        for text, name in processed
    )
    
    prompt = ChatPromptTemplate.from_template(prompt_template.format(
        count=len(processed),
        jd_context=" terhadap deskripsi pekerjaan" if jd_text else "",
        candidates=candidates_formatted
    ))
    
    chain = prompt | llm
    return chain.invoke({
        "jd_text": jd_text or ""
    }).content

def score_and_rank_candidates(resume_data: List[Tuple[str, str]], 
                            jd_text: Optional[str] = None,
                            criteria: Optional[Dict[str, int]] = None) -> Dict:
    """Scoring dengan integrasi nama"""
    scorer = ResumeScorer(criteria or {
        "Technical Skills": 8,
        "Problem Solving": 7,
        "Leadership": 6,
        "Communication": 5,
        "Cultural Fit": 4
    })
    
    # Siapkan input untuk scorer
    resume_texts = [data[0] for data in resume_data]
    filenames = [data[1] for data in resume_data]
    
    results = scorer.compare_resumes(resume_texts, jd_text)
    
    # Tambahkan nama ke hasil
    for i, result in enumerate(results["ranking"]):
        result["name"] = _get_candidate_name(resume_texts[i], filenames[i])
    
    return results
