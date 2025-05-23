import os
import re
from typing import Optional

class NameExtractor:
    def __init__(self):
        # Prefix yang mungkin digunakan sebelum nama dalam resume
        self.name_prefixes = ["nama:", "name:", "full name:", "nama lengkap:"]
        # Kata-kata yang tidak termasuk nama
        self.exclude_words = ["resume", "cv", "curriculum", "vitae"]
    
    def extract_name_from_resume(self, resume_text: str, filename: str = "") -> str:
        """Ekstraksi nama dari teks resume dengan fallback ke nama file"""
        try:
            # Pertama coba ekstrak dari teks resume
            text_name = self._extract_from_text(resume_text)
            if text_name:
                return text_name
                
            # Jika gagal, gunakan nama file
            if filename:
                file_name = self._extract_from_filename(filename)
                if file_name:
                    return file_name
                    
        except Exception:
            pass
            
        return self._generate_fallback_name(filename)
    
    def _extract_from_text(self, text: str) -> Optional[str]:
        """Ekstrak nama dari konten teks resume"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Pola 1: Cari kandidat nama di 10 baris pertama
        potential_names = [
            line for line in lines[:10] 
            if 2 <= len(line.split()) <= 4
            and any(c.isupper() for c in line)
            and not any(word in line.lower() for word in self.exclude_words)
        ]
        
        # Pola 2: Cek label nama yang eksplisit
        for line in lines[:15]:
            line_lower = line.lower()
            for prefix in self.name_prefixes:
                if prefix in line_lower:
                    extracted = line.split(prefix)[1].strip().title()
                    if len(extracted.split()) >= 2:
                        return extracted
        
        return potential_names[0].title() if potential_names else None
    
    def _extract_from_filename(self, filename: str) -> Optional[str]:
        """Ekstrak nama dari nama file"""
        if not filename:
            return None
            
        try:
            clean_name = re.sub(r'[^a-zA-Z\s]', '', os.path.splitext(filename)[0])
            parts = [p for p in clean_name.split() if len(p) > 1 and p[0].isupper()]
            if len(parts) >= 2:
                return ' '.join(parts[:3]).title()
        except:
            return None
    
    def _generate_fallback_name(self, filename: str = "") -> str:
        """Buat nama fallback dari nama file atau generic"""
        if filename:
            try:
                base = os.path.splitext(os.path.basename(filename))[0]
                clean = re.sub(r'[^a-zA-Z\s]', '', base)
                parts = [p for p in clean.split() if p[0].isupper()]
                if parts:
                    return parts[0].title() + " Candidate"
            except:
                pass
        return "Candidate " + (str(abs(hash(filename))[:4]) if filename else "")

# Buat instance di level modul
name_extractor = NameExtractor()

# Fungsi untuk backward compatibility
def extract_name_from_resume(resume_text: str, filename: str = "") -> str:
    """Fungsi standalone untuk kompatibilitas ke belakang"""
    return name_extractor.extract_name_from_resume(resume_text, filename)
