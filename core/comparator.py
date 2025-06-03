from typing import List
from langchain.prompts import ChatPromptTemplate
from utils.resume_standardizer import ResumeStandardizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResumeComparator:
    def __init__(self, domain: str = "general"):
        """
        Initialize comparator with domain flexibility
        
        Args:
            domain: Target domain (it, hr, finance, marketing, general, etc.)
        """
        self.domain = domain.lower()
        self.standardizer = ResumeStandardizer(domain=domain)
        
        # Domain-specific comparison criteria
        self.domain_criteria_mapping = {
            "IT": ["Technical Skills", "Problem Solving", "Project Delivery", "Innovation"],
            "HR": ["People Management", "Communication", "Talent Acquisition", "Employee Engagement"],
            "Finance": ["Analytical Skills", "Compliance Knowledge", "Financial Reporting", "Risk Management"],
            "Marketing": ["Creativity", "Digital Marketing", "Brand Strategy", "Data Analysis"],
            "Sales": ["Relationship Building", "Sales Achievement", "Negotiation", "Market Knowledge"],
            "Operations": ["Process Optimization", "Leadership", "Supply Chain", "Project Management"],
            "General": ["Professional Skills", "Communication", "Leadership", "Adaptability"]
        }
        
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize domain-flexible comparison prompt"""
        domain_context = self._get_domain_context()
        
        self.comparison_prompt = ChatPromptTemplate.from_template(
            f"""Bandingkan resume-resume yang sudah distandardisasi untuk domain {self.domain.upper()}:
            
            Resume yang Sudah Distandardisasi:
            {{resumes}}
            
            {domain_context}
            
            Berikan analisis dalam bahasa Indonesia dengan struktur:
            1. Ringkasan perbandingan
            2. Kandidat dengan keterampilan paling relevan
            3. Kandidat dengan pengalaman paling relevan
            4. Pertimbangan kecocokan budaya
            
            Fokus pada:
            - Kesesuaian keterampilan dengan domain {self.domain}
            - Relevansi pengalaman kerja
            - Potensi kepemimpinan
            - Progresi karier
            """
        )
    
    def _get_domain_context(self) -> str:
        """Get domain-specific context for comparison"""
        contexts = {
            "it": """
            Bandingkan berdasarkan keterampilan teknis, pengalaman proyek teknologi, dan inovasi.
            Pertimbangkan sertifikasi teknologi dan kemampuan problem solving teknis.
            """,
            "hr": """
            Bandingkan berdasarkan kemampuan manajemen SDM, komunikasi, dan pengalaman talent management.
            Pertimbangkan pengalaman dalam employee relations dan strategic HR.
            """,
            "finance": """
            Bandingkan berdasarkan kemampuan analisis keuangan, kepatuhan, dan pelaporan.
            Pertimbangkan sertifikasi keuangan dan pengalaman risk management.
            """,
            "marketing": """
            Bandingkan berdasarkan kreativitas, strategi pemasaran, dan analisis data.
            Pertimbangkan pengalaman digital marketing dan brand management.
            """,
            "sales": """
            Bandingkan berdasarkan pencapaian penjualan, hubungan klien, dan negosiasi.
            Pertimbangkan pengalaman dalam CRM dan market expansion.
            """,
            "operations": """
            Bandingkan berdasarkan optimalisasi proses, manajemen proyek, dan efisiensi operasional.
            Pertimbangkan pengalaman dalam supply chain dan quality control.
            """
        }
        return contexts.get(self.domain, "Bandingkan berdasarkan keterampilan profesional, pengalaman kerja, dan potensi kepemimpinan.")
    
    def compare_resumes(self, resume_texts: List[str], llm) -> str:
        """Compare multiple standardized resumes"""
        try:
            standardized_resumes, _ = self.standardizer.standardize_multiple(resume_texts)
            
            formatted_resumes = "\n\n---\n\n".join(
                f"Kandidat {i+1} (Standardisasi):\n{text}" 
                for i, text in enumerate(standardized_resumes)
            )
            
            chain = self.comparison_prompt | llm
            result = chain.invoke({"resumes": formatted_resumes}).content
            return result
            
        except Exception as e:
            logger.error(f"Comparison failed: {str(e)}")
            return f"⚠️ Error dalam perbandingan resume: {str(e)}"
