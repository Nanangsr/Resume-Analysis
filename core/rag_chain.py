import asyncio
import os
# from dotenv import load_dotenv # DIHAPUS
import streamlit as st # DITAMBAH
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from core.retriever import get_retriever
from typing import List, Optional, Dict, Tuple
from core.scoring import ResumeScorer
from utils.resume_standardizer import ResumeStandardizer
from utils.name_extractor import NameExtractor
from concurrent.futures import ThreadPoolExecutor
import logging
import hashlib
import re

# load_dotenv() # DIHAPUS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ResumeRagChain:
    def __init__(self, domain: str = "general"):
        """
        Initialize RAG chain with domain flexibility
        
        Args:
            domain: Target domain (it, hr, finance, marketing, general, etc.)
        """
        self.domain = domain.lower()
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b",
            # DIUBAH: Ambil API Key dari Streamlit Secrets, bukan os.getenv
            api_key=st.secrets["GROQ_API_KEY"], 
            request_timeout=120
        )
        self.name_extractor = NameExtractor()
        self.standardizer = ResumeStandardizer(domain=domain)
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize domain-flexible prompts"""
        domain_context = self._get_domain_context()
        
        self.qa_prompt = ChatPromptTemplate.from_template(
            f"""Jawab pertanyaan tentang resume {self.domain.upper()} untuk {{name}}:
            Pertanyaan: {{question}}
            Konten Resume:
            {{resume_text}}
            
            {domain_context}
            
            Aturan:
            1. Sebut kandidat sebagai {{name}}
            2. Spesifik tentang referensi bagian
            3. Jika tidak yakin, katakan "Tidak disebutkan di resume"
            4. Jawab dalam bahasa Indonesia
            5. Jangan sertakan proses berpikir Anda dalam jawaban.
            """
        )
        
        self.search_prompt = ChatPromptTemplate.from_template(
            f"""Deskripsi Pekerjaan:
            {{jd_text}}
            Kandidat yang Cocok untuk domain {self.domain.upper()}:
            {{candidates}}
            
            {domain_context}
            
            Berikan dalam bahasa Indonesia:
            1. 3 kecocokan teratas dengan alasan
            2. Area yang perlu dikembangkan
            3. Rekomendasi perekrutan
            Jangan sertakan proses berpikir Anda dalam jawaban.
            """
        )
        
        self.profile_prompt = ChatPromptTemplate.from_template(
            f"""Analisis resume ini untuk {{name}} (level {{level}}) dalam domain {self.domain.upper()}:
            {{resume_text}}
            
            {domain_context}
            
            Berikan dalam bahasa Indonesia:
            1. 5 kekuatan utama
            2. 3 area pengembangan  
            3. Peran yang direkomendasikan
            4. Pertanyaan wawancara
            Jangan sertakan proses berpikir Anda dalam jawaban.
            """
        )
        
        self.compare_prompt = ChatPromptTemplate.from_template(
            f"""Bandingkan {{count}} kandidat untuk domain {self.domain.upper()}{{jd_context}}:
            {{candidates}}
            
            {domain_context}
            
            Analisis dalam bahasa Indonesia harus mencakup:
            1. Perbandingan keterampilan utama
            2. Potensi kepemimpinan  
            3. Nilai tambah budaya
            4. Potensi pengembangan
            Jangan sertakan proses berpikir Anda dalam jawaban.
            """
        )
    
    def _get_domain_context(self) -> str:
        """Get domain-specific context"""
        contexts = {
            "it": """
            Fokus pada keterampilan teknis, pengalaman proyek teknologi, dan sertifikasi.
            Pertimbangkan relevansi dengan teknologi terkini dan problem solving.
            """,
            "hr": """
            Fokus pada manajemen SDM, komunikasi, dan pengalaman talent management.
            Pertimbangkan kemampuan strategis dan employee engagement.
            """,
            "finance": """
            Fokus pada analisis keuangan, kepatuhan, dan pelaporan.
            Pertimbangkan sertifikasi keuangan dan risk management.
            """,
            "marketing": """
            Fokus pada kreativitas, strategi digital, dan analisis data.
            Pertimbangkan pengalaman brand management dan campaign execution.
            """,
            "sales": """
            Fokus pada pencapaian penjualan, hubungan klien, dan negosiasi.
            Pertimbangkan pengalaman CRM dan market expansion.
            """,
            "operations": """
            Fokus pada optimalisasi proses, manajemen proyek, dan efisiensi.
            Pertimbangkan supply chain dan quality control.
            """
        }
        return contexts.get(self.domain, "Fokus pada keterampilan profesional, pengalaman, dan potensi kepemimpinan.")
    
    def _clean_output(self, text: str) -> str:
        """Clean model output from thinking processes"""
        if '<think>' in text.lower():
            logger.warning("Detected thinking process in output")
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'\(thinking.*?\)', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def batchprocess(self, func, items: List, batch_size: int = 5):
        """Helper untuk pemrosesan paralel"""
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            return list(executor.map(func, items))
    
    def getcandidate_name(self, resume_text: str, filename: str = "") -> str:
        """Wrapper untuk ekstraksi nama"""
        try:
            if not resume_text:
                return f"Unknown Candidate {hashlib.md5(filename.encode()).hexdigest()[:8]}"
            return self.name_extractor.extract_name_from_resume(resume_text, filename) or f"Unknown Candidate {hashlib.md5(resume_text.encode()).hexdigest()[:8]}"
        except Exception as e:
            logger.error(f"Error extracting name: {str(e)}")
            return f"Unknown Candidate {hashlib.md5((resume_text or filename).encode()).hexdigest()[:8]}"
    
    async def resume_qa(self, resume_text: str, question: str, filename: str = "") -> str:
        """Q&A resume dengan konteks domain"""
        try:
            if not resume_text:
                return "Resume text is empty"
            std_resume = self.standardizer.standardize_resume(resume_text)
            candidate_name = self.getcandidate_name(resume_text, filename)
            chain = self.qa_prompt | self.llm
            result = await chain.ainvoke({
                "question": question,
                "resume_text": std_resume[:3000],
                "name": candidate_name
            })
            return self._clean_output(result.content)
        except Exception as e:
            logger.error(f"Error in resume_qa: {str(e)}")
            return f"⚠️ Error dalam Q&A resume: {str(e)}"
    
    async def candidate_search(self, jd_text: str, resume_docs: Optional[List[Tuple[str, str]]] = None) -> str:
        """Pencarian kandidat dengan konteks domain"""
        try:
            retriever = get_retriever()
            if resume_docs is None:
                relevant_docs = retriever.invoke(jd_text)
                processed = self.batchprocess(
                    lambda doc: (
                        self.standardizer.standardize_resume(doc.page_content),
                        self.getcandidate_name(doc.page_content, doc.metadata.get('filename', ''))
                    ),
                    relevant_docs
                )
            else:
                processed = []
                for data in resume_docs:
                    if not isinstance(data, tuple) or len(data) != 2:
                        logger.warning(f"Invalid resume data: {data}")
                        resume_text = data[0] if isinstance(data, tuple) and len(data) > 0 else str(data)
                        processed.append((
                            self.standardizer.standardize_resume(resume_text),
                            self.getcandidate_name(resume_text, "")
                        ))
                    else:
                        processed.append((
                            self.standardizer.standardize_resume(data[0]),
                            self.getcandidate_name(data[0], data[1])
                        ))
            
            candidates_formatted = "\n\n".join(
                f"Kandidat {i+1} ({name}):\n{text[:2000]}..."
                for i, (text, name) in enumerate(processed)
            )
            chain = self.search_prompt | self.llm
            result = await chain.ainvoke({
                "jd_text": jd_text,
                "candidates": candidates_formatted
            })
            return self._clean_output(result.content)
        except Exception as e:
            logger.error(f"Error in candidate_search: {str(e)}")
            return f"⚠️ Error dalam pencarian kandidat: {str(e)}"
    
    async def candidate_profiling(self, resume_text: str, filename: str = "") -> str:
        """Profil kandidat dengan konteks domain"""
        try:
            if not resume_text:
                return "⚠️ Error: Resume text is empty"
            std_resume = self.standardizer.standardize_resume(resume_text)
            level = self.standardizer.detect_resume_level(std_resume)
            candidate_name = self.getcandidate_name(resume_text, filename)
            chain = self.profile_prompt | self.llm
            result = await chain.ainvoke({
                "level": level,
                "resume_text": std_resume[:3000],
                "name": candidate_name
            })
            return self._clean_output(result.content)
        except Exception as e:
            logger.error(f"Error in candidate_profiling: {str(e)}")
            return f"⚠️ Error dalam profiling kandidat: {str(e)}"
    
    async def compare_candidates(self, resume_data: List[Tuple[str, str]], jd_text: Optional[str] = None) -> str:
        """Bandingkan kandidat dengan konteks domain"""
        try:
            if not resume_data:
                return "Tidak ada resume yang valid untuk dibandingkan"
            
            processed = []
            for data in resume_data:
                if not isinstance(data, tuple) or len(data) != 2:
                    logger.warning(f"Invalid resume data: {data}")
                    resume_text = data[0] if isinstance(data, tuple) and len(data) > 0 else str(data)
                    processed.append((
                        self.standardizer.standardize_resume(resume_text),
                        self.getcandidate_name(resume_text, "")
                    ))
                else:
                    processed.append((
                        self.standardizer.standardize_resume(data[0]),
                        self.getcandidate_name(data[0], data[1])
                    ))
            
            candidates_formatted = "\n\n---\n\n".join(
                f"{name}:\n{text[:2000]}..." 
                for text, name in processed
            )
            chain = self.compare_prompt | self.llm
            result = await chain.ainvoke({
                "count": len(processed),
                "jd_context": " terhadap deskripsi pekerjaan" if jd_text else "",
                "candidates": candidates_formatted,
                "jd_text": jd_text or ""
            })
            return self._clean_output(result.content)
        except Exception as e:
            logger.error(f"Error in compare_candidates: {str(e)}")
            return f"⚠️ Error dalam perbandingan kandidat: {str(e)}"
    
    async def generate_llm_narrative_analysis(self, scoring_results: Dict, jd_text: Optional[str] = None) -> str:
        """Generate narrative analysis dengan konteks domain"""
        try:
            if not scoring_results.get("ranking"):
                return "⚠️ Tidak ada data ranking untuk dianalisis"
            
            candidates_info = []
            for candidate in scoring_results["ranking"]:
                candidate_info = {
                    "name": candidate.get("name", f"Kandidat {candidate['candidate_id']}"),
                    "scores": candidate["scores"],
                    "ai_score": candidate["ai_score"],
                    "level": candidate["level"],
                    "strengths": [],
                    "weaknesses": []
                }
                
                for criterion, score in candidate["scores"].items():
                    normalized_score = score / (scoring_results["criteria"][criterion] / 10)
                    if normalized_score >= 7:
                        candidate_info["strengths"].append(criterion)
                    elif normalized_score <= 4:
                        candidate_info["weaknesses"].append(criterion)
                
                candidates_info.append(candidate_info)
            
            domain_context = self._get_domain_context()
            prompt = ChatPromptTemplate.from_template(
                f"""Anda adalah ahli HR untuk domain {self.domain.upper()}. 
                Berikut adalah hasil scoring kandidat:
                
                {{candidates_info}}
                
                {{jd_context}}
                
                {domain_context}
                
                Berikan analisis naratif dalam bahasa Indonesia dengan struktur:
                1. Ringkasan Eksekutif (maksimal 3 kalimat)
                2. Analisis Komparatif:
                   - Bandingkan kandidat berdasarkan skor dan level
                   - Soroti perbedaan utama dalam kompetensi
                3. Rekomendasi Perekrutan:
                   - Kandidat terbaik dan alasannya
                   - Potensi hidden gem
                4. Pertimbangan Budaya Organisasi
                5. Rencana Pengembangan
                Jangan sertakan proses berpikir Anda dalam jawaban.
                """
            )
            
            candidates_formatted = "\n\n".join(
                f"Kandidat: {info['name']}\n"
                f"- Level: {info['level']}\n"
                f"- AI Score: {info['ai_score']:.1f}/100\n"
                f"- Kekuatan: {', '.join(info['strengths']) if info['strengths'] else 'Tidak ada'}\n"
                f"- Area Pengembangan: {', '.join(info['weaknesses']) if info['weaknesses'] else 'Tidak ada'}"
                for info in candidates_info
            )
            
            jd_context = f"\nDeskripsi Pekerjaan:\n{jd_text}" if jd_text else ""
            chain = prompt | self.llm
            
            result = await chain.ainvoke({
                "candidates_info": candidates_formatted,
                "jd_context": jd_context
            })
            
            return self._clean_output(result.content)
        except Exception as e:
            logger.error(f"Error in narrative analysis: {str(e)}")
            return f"⚠️ Error dalam analisis: {str(e)}"
    
    def score_and_rank_candidates(self, resume_data: List[Tuple[str, str]], 
                                jd_text: Optional[str] = None,
                                criteria: Optional[Dict[str, int]] = None) -> Dict:
        """Score and rank candidates dengan konteks domain"""
        try:
            if not resume_data:
                return {"error": "No resume data provided"}
            
            validated_resume_data = []
            for data in resume_data:
                if not isinstance(data, tuple) or len(data) != 2:
                    resume_text = data[0] if isinstance(data, tuple) and len(data) > 0 else str(data)
                    validated_resume_data.append((resume_text.strip(), ""))
                else:
                    validated_resume_data.append((data[0].strip(), data[1].strip()))
            
            if not criteria:
                criteria = self.standardizer.get_domain_specific_criteria()
            
            scorer = ResumeScorer(domain=self.domain, criteria=criteria)
            scoring_results = scorer.compare_resumes([data[0] for data in validated_resume_data], jd_text)
            
            candidate_id_to_resume = {
                i+1: (text, name) for i, (text, name) in enumerate(validated_resume_data)
            }
            
            for candidate in scoring_results.get("ranking", []):
                cid = candidate["candidate_id"]
                resume_text, filename = candidate_id_to_resume.get(cid, ("", ""))
                candidate["name"] = self.getcandidate_name(resume_text, filename)
            
            narrative = asyncio.run(self.generate_llm_narrative_analysis(scoring_results, jd_text))
            scoring_results["narrative_analysis"] = narrative
            
            if jd_text:
                st.session_state["last_jd_text"] = jd_text
            st.session_state["last_narrative_analysis"] = narrative
            st.session_state["last_scoring_results"] = scoring_results
            
            return scoring_results
        except Exception as e:
            logger.error(f"Error in score_and_rank_candidates: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}
