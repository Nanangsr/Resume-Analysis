from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
import re
import logging
import pandas as pd
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class ResumeStandardizer:
    def __init__(self, domain: str = "general"):
        """
        Initialize standardizer with domain flexibility
        
        Args:
            domain: Target domain (it, hr, finance, marketing, general, etc.)
        """
        self.domain = domain.lower()
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b",
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30,
            model_kwargs={"seed": 42}
        )
        self._init_prompts()
        self.cache = {}
    
    def _init_prompts(self):
        """Initialize domain-flexible prompts"""
        
        # Generic skills mapping for different domains
        self.domain_skills_mapping = {
            "IT": ["programming", "software development", "database", "cloud", "AI/ML", "cybersecurity"],
            "HR": ["recruitment", "talent management", "performance management", "employee relations", "HRIS", "compensation"],
            "Finance": ["financial analysis", "budgeting", "accounting", "risk management", "compliance", "auditing"],
            "Marketing": ["digital marketing", "content creation", "SEO/SEM", "analytics", "brand management", "social media"],
            "Sales": ["lead generation", "client relationship", "negotiation", "CRM", "sales forecasting", "territory management"],
            "Operations": ["process improvement", "supply chain", "quality management", "logistics", "vendor management", "project management"],
            "General": ["leadership", "communication", "problem solving", "teamwork", "analytical thinking", "project management"]
        }
        
        domain_context = self._get_domain_context()
        
        self.standardization_prompt = ChatPromptTemplate.from_template(
            f"""Transformasikan resume ke format terstruktur berikut untuk domain {self.domain.upper()}. 
            Pastikan semua bagian ada dan diformat dengan benar:
            
            - NAME: Nama lengkap
            - SKILLS: Keterampilan profesional yang relevan dengan domain {self.domain}, dipisah koma
            - EXPERIENCE_YEARS: Total tahun pengalaman profesional
            - EXPERIENCE:
              - Format: "**Perusahaan** (*Posisi*) MM/YYYY-MM/YYYY: Pencapaian1; Pencapaian2"
            - EDUCATION:
              - Format: "Gelar, Institusi (MM/YYYY)"
            - CERTIFICATIONS: Sertifikasi profesional yang relevan, dipisah koma jika ada, jika tidak "None"
            - JOB_ROLE: Peran pekerjaan terbaru
            - PROJECTS_COUNT: Jumlah proyek yang disebutkan
            - SALARY_EXPECTATION: Jika disebutkan, jika tidak "Not specified"
            - DOMAIN_EXPERTISE: Keahlian spesifik dalam domain {self.domain}
            
            {domain_context}
            
            Jika format pengalaman tidak jelas, gunakan format sederhana seperti:
            "**Perusahaan** (*Posisi*) Tahun: Deskripsi pekerjaan"
            
            Input Resume:
            {{resume_text}}
            
            Keluarkan hanya teks yang diformat.
            """
        )
        
        self.level_detection_prompt = ChatPromptTemplate.from_template(
            f"""Analyze experience in this resume for {self.domain.upper()} domain (STRICT RULES):
            
            [LEVEL CRITERIA FOR {self.domain.upper()} DOMAIN]
            Entry/Junior:
            - 0-2 years relevant experience
            - Entry-level positions in {self.domain}
            - Education-focused with limited practical experience
            - Basic skills in domain fundamentals
            
            Mid-level:
            - 3-6 years relevant experience
            - Demonstrated competency in core {self.domain} functions
            - Some project leadership or specialization
            - Moderate achievements and responsibilities
            
            Senior:
            - 7+ years relevant experience
            - Clear career progression in {self.domain}
            - Leadership roles or senior positions
            - Strategic impact and advanced expertise
            - Mentoring or team management experience
            
            Expert/Executive:
            - 10+ years with significant {self.domain} achievements
            - Executive or director-level positions
            - Industry recognition or major accomplishments
            - Strategic business impact
            
            Resume Excerpt:
            {{resume_excerpt}}
            
            Return ONLY the level keyword: entry | mid | senior | expert
            """
        )
    
    def _get_domain_context(self) -> str:
        """Get domain-specific context for better standardization"""
        contexts = {
            "it": """
            Fokus pada keterampilan teknis seperti bahasa pemrograman, framework, database, cloud platforms, 
            metodologi development, dan sertifikasi teknologi.
            """,
            "hr": """
            Fokus pada keterampilan manajemen SDM seperti recruitment, talent development, employee relations,
            HRIS systems, labor law compliance, dan sertifikasi HR profesional.
            """,
            "finance": """
            Fokus pada keterampilan keuangan seperti financial analysis, accounting, budgeting, risk management,
            compliance, auditing, dan sertifikasi keuangan (CPA, CFA, dll).
            """,
            "marketing": """
            Fokus pada keterampilan pemasaran seperti digital marketing, content strategy, brand management,
            analytics, SEO/SEM, social media, dan sertifikasi marketing.
            """,
            "sales": """
            Fokus pada keterampilan penjualan seperti lead generation, relationship building, negotiation,
            CRM systems, sales forecasting, dan achievement metrics.
            """,
            "operations": """
            Fokus pada keterampilan operasional seperti process improvement, supply chain, quality management,
            project management, dan operational efficiency.
            """
        }
        return contexts.get(self.domain, "Fokus pada keterampilan profesional yang relevan dengan bidang kerja.")
    
    def standardize_resume(self, resume_text: str) -> str:
        """Standardize resume with domain awareness"""
        cache_key = hashlib.md5((resume_text + self.domain).encode()).hexdigest()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if not resume_text or len(resume_text.strip()) < 50:
                raise ValueError("Resume text too short")
                
            cleaned_text = self._preprocess_text(resume_text)
            
            chain = self.standardization_prompt | self.llm
            result = chain.invoke({"resume_text": cleaned_text}).content
            
            validated_result = self._validate_for_model_features(result)
            self.cache[cache_key] = validated_result
            return validated_result
            
        except Exception as e:
            logger.error(f"Standardization failed: {str(e)}")
            return self._fallback_format(resume_text)
    
    def detect_resume_level(self, resume_text: str) -> str:
        """Detect resume level with domain context"""
        try:
            excerpt = self._extract_structured_experience(resume_text)
            chain = self.level_detection_prompt | self.llm
            response = chain.invoke({"resume_excerpt": excerpt}).content.lower()
            
            valid_levels = {'entry', 'mid', 'senior', 'expert'}
            if response not in valid_levels:
                raise ValueError(f"Invalid level detected: {response}")
                
            return response
            
        except Exception as e:
            logger.warning(f"Level detection failed: {str(e)}")
            return self._estimate_level_from_dates(resume_text)
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'(?i)\b(email|phone|address):.*?\n', '', text)
        return text
    
    def _validate_for_model_features(self, text: str) -> str:
        """Validate standardized resume format"""
        required_sections = {
            'NAME': r'NAME:(.+?)(?=\n[A-Z_]+:|$)',
            'SKILLS': r'SKILLS:(.+?)(?=\n[A-Z_]+:|$)',
            'EXPERIENCE_YEARS': r'EXPERIENCE_YEARS:(.+?)(?=\n[A-Z_]+:|$)',
            'EXPERIENCE': r'EXPERIENCE:(.+?)(?=\n[A-Z_]+:|$)',
            'EDUCATION': r'EDUCATION:(.+?)(?=\n[A-Z_]+:|$)',
            'CERTIFICATIONS': r'CERTIFICATIONS:(.+?)(?=\n[A-Z_]+:|$)',
            'JOB_ROLE': r'JOB_ROLE:(.+?)(?=\n[A-Z_]+:|$)',
            'PROJECTS_COUNT': r'PROJECTS_COUNT:(.+?)(?=\n[A-Z_]+:|$)',
            'SALARY_EXPECTATION': r'SALARY_EXPECTATION:(.+?)(?=\n[A-Z_]+:|$)',
            'DOMAIN_EXPERTISE': r'DOMAIN_EXPERTISE:(.+?)(?=\n[A-Z_]+:|$)'
        }
        
        errors = []
        for section, pattern in required_sections.items():
            if not re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                errors.append(f"Missing {section} section")
        
        if errors:
            logger.warning(f"Validation errors: {'; '.join(errors)}")
            return self._fallback_format(text)
            
        return text
    
    def _extract_structured_experience(self, text: str) -> str:
        """Extract experience section for level detection"""
        exp_match = re.search(r'EXPERIENCE:(.+?)(?=\n[A-Z_]+:|$)', text, re.DOTALL | re.IGNORECASE)
        if exp_match:
            return exp_match.group(1)[:2000]
        return text[:1500]
    
    def _estimate_level_from_dates(self, text: str) -> str:
        """Fallback level estimation from dates"""
        date_ranges = re.findall(r'(\d{2}/\d{4})-(\d{2}/\d{4})|\d{4}-\d{4}|\d{4}\s*-\s*(?:Present|Current)', text, re.IGNORECASE)
        
        if not date_ranges:
            return 'entry'
            
        total_years = 0
        for date_range in date_ranges:
            if len(date_range) >= 2 and date_range[0] and date_range[1]:
                try:
                    start_year = int(date_range[0].split('/')[1]) if '/' in date_range[0] else int(date_range[0])
                    end_year = int(date_range[1].split('/')[1]) if '/' in date_range[1] else int(date_range[1])
                    total_years += end_year - start_year
                except:
                    continue
            
        avg_years = total_years / max(1, len(date_ranges))
        
        if avg_years <= 2:
            return 'entry'
        elif avg_years <= 6:
            return 'mid'
        elif avg_years <= 10:
            return 'senior'
        return 'expert'
    
    def _fallback_format(self, text: str) -> str:
        """Create fallback standardized format"""
        domain_skills = self.domain_skills_mapping.get(self.domain, self.domain_skills_mapping["general"])
        
        sections = [
            "NAME: Unknown Candidate",
            f"SKILLS: {', '.join(domain_skills[:5])}",
            "EXPERIENCE_YEARS: 0",
            "EXPERIENCE: **Unknown Company** (*Position*) 2020-2022: Professional experience",
            "EDUCATION: Bachelor's Degree, University (01/2020)",
            "CERTIFICATIONS: None",
            "JOB_ROLE: Not specified",
            "PROJECTS_COUNT: 0",
            "SALARY_EXPECTATION: Not specified",
            f"DOMAIN_EXPERTISE: Entry-level {self.domain} knowledge"
        ]
        return '\n'.join(sections)
    
    def standardize_multiple(self, resume_texts: List[str]) -> Tuple[List[str], List[str]]:
        """Standardize multiple resumes and detect their levels"""
        standardized = []
        levels = []
        
        for text in resume_texts:
            std_text = self.standardize_resume(text)
            level = self.detect_resume_level(std_text)
            standardized.append(std_text)
            levels.append(level)
        
        return standardized, levels
    
    def get_domain_specific_criteria(self) -> Dict[str, int]:
        """Get domain-specific scoring criteria"""
        criteria_mapping = {
            "it": {
                "Technical Skills": 9,
                "Problem Solving": 8,
                "Work Experience": 8,
                "Education": 6,
                "Certifications": 7,
                "Project Management": 6
            },
            "hr": {
                "People Management": 9,
                "Communication": 8,
                "Work Experience": 8,
                "Education": 7,
                "Certifications": 6,
                "Strategic Thinking": 7
            },
            "finance": {
                "Analytical Skills": 9,
                "Attention to Detail": 8,
                "Work Experience": 8,
                "Education": 8,
                "Certifications": 9,
                "Compliance Knowledge": 7
            },
            "marketing": {
                "Creativity": 8,
                "Digital Skills": 8,
                "Communication": 9,
                "Work Experience": 7,
                "Education": 6,
                "Data Analysis": 7
            },
            "sales": {
                "Relationship Building": 9,
                "Communication": 9,
                "Achievement Record": 8,
                "Work Experience": 8,
                "Negotiation Skills": 7,
                "Industry Knowledge": 6
            },
            "operations": {
                "Process Improvement": 8,
                "Leadership": 8,
                "Problem Solving": 8,
                "Work Experience": 8,
                "Education": 6,
                "Project Management": 9
            },
            "general": {
                "Professional Skills": 8,
                "Work Experience": 8,
                "Education": 7,
                "Leadership": 7,
                "Communication": 8,
                "Problem Solving": 7
            }
        }
        
        return criteria_mapping.get(self.domain, criteria_mapping["general"])
