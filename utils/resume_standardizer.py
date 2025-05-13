from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from typing import List, Tuple
import re
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ResumeStandardizer:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            model_name="deepseek-r1-distill-llama-70b",  # Model yang lebih hemat biaya
            api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=30
        )
        self._init_prompts()
        
    def _init_prompts(self):
        """Inisialisasi dan cache template prompt"""
        self.standardization_prompt = ChatPromptTemplate.from_template(
            """Transformasikan resume ini ke format yang ramah ATS dengan mengikuti aturan berikut:
            
            [ATURAN STRUKTUR]
            1. BAGIAN WAJIB (urutannya):
               - SUMMARY: Ringkasan profesional 3-5 kalimat
               - EXPERIENCE: Urutan kronologis terbalik, format MM/YYYY
               - EDUCATION: Gelar saja, format MM/YYYY
               - SKILLS: Dikelompokkan berdasarkan kategori (Teknis, Soft Skill, dll)
               - (Opsional) SERTIFIKASI, PROYEK jika ada
            
            2. FORMAT:
               - Hanya poin-poin (tidak ada paragraf)
               - Mulai setiap poin dengan kata kerja kuat
               - Kuantifikasi pencapaian (contoh: "Meningkatkan X sebesar 30%")
               - Hapus: kata ganti, info pribadi (alamat, tanggal lahir)
               - Nama perusahaan tebal: **Perusahaan**
               - Jabatan miring: *Jabatan*
            
            3. KONTEN:
               - Hanya informasi profesional
               - Hapus: foto, tabel, grafik
               - Standarisasi nama gelar (BSc bukan Bachelor of Science)
            
            Input Resume:
            {resume_text}
            
            Output HANYA teks resume yang sudah diformat, TANPA komentar tambahan.
            """
        )
        
        self.level_detection_prompt = ChatPromptTemplate.from_template(
            """Analisa pengalaman di resume dan tentukan level (ATURAN KETAT):
            
            [KRITERIA LEVEL]
            Junior:
            - 0-3 tahun pengalaman
            - Jabatan entry-level
            - Tidak ada kepemimpinan tim
            - Pendidikan mungkin ditonjolkan
            
            Mid:
            - 4-7 tahun pengalaman
            - Beberapa kepemimpinan proyek/tim
            - Pencapaian moderat
            - Keterampilan khusus
            
            Senior:
            - 8+ tahun pengalaman
            - Progresi karier jelas
            - Kepemimpinan signifikan
            - Bukti dampak strategis
            
            [ANALISIS]
            Cuplikan Resume:
            {resume_excerpt}
            
            Kembalikan HANYA kata kunci level: junior | mid | senior
            """
        )

    def standardize_resume(self, resume_text: str) -> str:
        """Standarisasi resume dengan validasi yang dioptimalkan"""
        try:
            if not resume_text or len(resume_text.strip()) < 50:
                raise ValueError("Teks resume terlalu pendek")
                
            # Pra-pemrosesan teks
            cleaned_text = self._preprocess_text(resume_text)
            
            chain = self.standardization_prompt | self.llm
            result = chain.invoke({"resume_text": cleaned_text}).content
            
            # Pasca-pemrosesan dan validasi
            return self._validate_standardized_resume(result)
            
        except Exception as e:
            logger.error(f"Gagal standarisasi: {str(e)}")
            return self._fallback_format(resume_text)

    def detect_resume_level(self, resume_text: str) -> str:
        """Deteksi level resume yang lebih andal dengan fallback"""
        try:
            excerpt = self._extract_relevant_excerpt(resume_text)
            chain = self.level_detection_prompt | self.llm
            response = chain.invoke({"resume_excerpt": excerpt}).content.lower()
            
            # Validasi respons
            if response not in {'junior', 'mid', 'senior'}:
                raise ValueError(f"Level tidak valid terdeteksi: {response}")
                
            return response
            
        except Exception as e:
            logger.warning(f"Deteksi level gagal: {str(e)}")
            return self._estimate_level_fallback(resume_text)

    def standardize_multiple(self, resume_texts: List[str]) -> Tuple[List[str], List[str]]:
        """Proses batch dengan pelacakan progres"""
        standardized = []
        errors = []
        
        for text in resume_texts:
            try:
                standardized.append(self.standardize_resume(text))
            except Exception as e:
                standardized.append("")
                errors.append(str(e))
                
        return standardized, errors
    
    def _preprocess_text(self, text: str) -> str:
        """Bersihkan teks input sebelum pemrosesan"""
        # Hapus spasi ganda, karakter aneh
        text = re.sub(r'\s+', ' ', text).strip()
        # Hapus karakter non-ASCII
        return text.encode('ascii', 'ignore').decode('ascii')

    def _validate_standardized_resume(self, text: str) -> str:
        """Periksa apakah output memenuhi persyaratan minimum"""
        required_sections = {'SUMMARY', 'EXPERIENCE', 'EDUCATION', 'SKILLS'}
        present_sections = {s for s in required_sections if s in text}
        
        if len(present_sections) < 3:
            raise ValueError("Bagian wajib tidak lengkap")
            
        # Periksa penggunaan poin-poin
        if text.count('•') < 5 and text.count('-') < 5:
            raise ValueError("Poin-poin tidak cukup")
            
        return text

    def _extract_relevant_excerpt(self, text: str) -> str:
        """Ekstrak bagian paling relevan untuk deteksi level"""
        # Prioritaskan bagian pengalaman
        experience_match = re.search(r'EXPERIENCE(.+?)(?=\n[A-Z]{3,}|$)', text, re.DOTALL|re.IGNORECASE)
        if experience_match:
            return experience_match.group(1)[:2000]
        return text[:1500]

    def _estimate_level_fallback(self, text: str) -> str:
        """Estimasi level sederhana sebagai fallback"""
        year_patterns = r'(\d{4}[-–]\d{4}|\d{4}[^\d]\w+ \d{4})'
        matches = re.findall(year_patterns, text)
        total_years = min(len(matches) * 2, 15)  # Estimasi 2 tahun per posisi
        
        if total_years <= 3:
            return 'junior'
        elif total_years <= 7:
            return 'mid'
        return 'senior'

    def _fallback_format(self, text: str) -> str:
        """Format dasar ketika LLM gagal"""
        sections = [
            "SUMMARY:\n[Ringkasan profesional yang diekstrak]",
            "\nEXPERIENCE:\n" + re.sub(r'(?<=\n)(?=\S)', '• ', text),
            "\nSKILLS:\n[Keterampilan yang diekstrak]"
        ]
        return '\n'.join(sections)
