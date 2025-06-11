import os
import re
from typing import Optional
from pydantic import BaseModel, field_validator
import hashlib

class CandidateName(BaseModel):
    full_name: str

    @field_validator('full_name')
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Nama tidak boleh kosong')
        words = v.split()
        if len(words) < 2:
            raise ValueError('Nama harus terdiri dari setidaknya dua kata')
        # Validasi setiap kata: minimal 3 huruf untuk nama, atau 1-2 huruf untuk inisial
        for word in words:
            if len(word) < 3 and not re.match(r'^[A-Z][a-z]?$', word):
                raise ValueError(f'Kata "{word}" tidak valid sebagai nama atau inisial')
        return v.strip()

class NameExtractor:
    def __init__(self):
        self.name_prefixes = ["nama:", "name:", "full name:", "nama lengkap:", "candidate:", "applicant:"]
        self.exclude_words = ["resume", "cv", "curriculum", "vitae", "document", "file", "copy"]
        self.header_keywords = [
            "skills", "education", "experience", "employment", "history",
            "profile", "summary", "objective", "certifications", "projects"
        ]
        self.name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)$',
            r'^([A-Z][a-z]+(?:\s+[a-z]+)*\s+[A-Z][a-z]+)$',
        ]
    
    def extract_name_from_resume(self, resume_text: str, filename: str = "") -> str:
        """Ekstraksi nama dengan prioritas dari nama file, lalu dari teks resume, dengan validasi Pydantic"""
        try:
            if filename:
                file_name = self._extract_from_filename(filename)
                if file_name:
                    try:
                        candidate = CandidateName(full_name=file_name)
                        return candidate.full_name
                    except ValueError as e:
                        print(f"Validasi nama dari file gagal: {e}")
            
            text_name = self._extract_from_text(resume_text)
            if text_name:
                try:
                    candidate = CandidateName(full_name=text_name)
                    return candidate.full_name
                except ValueError as e:
                    print(f"Validasi nama dari teks gagal: {e}")
                
        except Exception as e:
            print(f"Error extracting name: {e}")
    
        return self._generate_fallback_name(filename)
    
    def _extract_from_text(self, text: str) -> Optional[str]:
        if not text:
            return None
            
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines[:20]:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in self.header_keywords):
                continue
            for prefix in self.name_prefixes:
                if prefix in line_lower:
                    name_part = line.split(':', 1)[-1].strip()
                    cleaned_name = self._clean_name_text(name_part)
                    if cleaned_name and len(cleaned_name.split()) >= 2:
                        return cleaned_name
        
        potential_names = []
        for line in lines[:10]:
            line_lower = line.lower()
            if len(line) > 50 or any(word in line_lower for word in self.exclude_words + self.header_keywords):
                continue
                
            for pattern in self.name_patterns:
                if re.match(pattern, line.strip()):
                    cleaned = self._clean_name_text(line)
                    if cleaned:
                        potential_names.append(cleaned)
                        break
        
        if not potential_names:
            for line in lines[:3]:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in self.header_keywords):
                    continue
                words = line.split()
                if 2 <= len(words) <= 4:
                    if all(word[0].isupper() and word.isalpha() for word in words):
                        cleaned = self._clean_name_text(line)
                        if cleaned:
                            potential_names.append(cleaned)
        
        return potential_names[0] if potential_names else None
    
    def _extract_from_filename(self, filename: str) -> Optional[str]:
        if not filename:
            return None
            
        try:
            base_name = os.path.splitext(os.path.basename(filename))[0].lower()
            
            base_name = re.sub(r'^(resume|cv|curriculum|vitae)[-_]?\s*', '', base_name)
            base_name = re.sub(r'\s*[-_]?(resume|cv|curriculum|vitae)$', '', base_name)
            
            base_name = re.sub(r'[-_]+', ' ', base_name).strip()
            
            clean_name = re.sub(r'[^a-z\s]', '', base_name).strip()
            
            words = clean_name.split()
            if len(words) >= 2:
                name_words = [word.capitalize() for word in words if len(word) >= 2]
                if len(name_words) >= 2:
                    return ' '.join(name_words[:3])
            
            if '_' in base_name or '-' in base_name:
                parts = re.split(r'[-_]', base_name)
                name_words = [part.capitalize() for part in parts if len(part) >= 1]
                if len(name_words) >= 2:
                    return ' '.join(name_words[:3])
            
            clean_base = re.sub(r'[^a-z]', '', base_name)
            if len(clean_base) >= 4:
                # Heuristik: nama depan biasanya lebih panjang, sisanya inisial atau nama belakang
                for split_point in range(3, len(clean_base) - 1):  # Mulai dari 3 untuk nama depan
                    first_name = clean_base[:split_point]
                    last_name = clean_base[split_point:]
                    if len(first_name) >= 3 and (1 <= len(last_name) <= 2 or len(last_name) >= 3):
                        return f"{first_name.capitalize()} {last_name.capitalize()}"
                
        except Exception as e:
            print(f"Error extracting from filename: {e}")
            
        return None
    
    def _clean_name_text(self, text: str) -> Optional[str]:
        if not text:
            return None
            
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'^(mr\.?|mrs\.?|ms\.?|dr\.?)\s*', '', text, flags=re.IGNORECASE)
        
        words = text.split()
        if 2 <= len(words) <= 4:
            valid_words = []
            for word in words:
                clean_word = re.sub(r'[^\w]$', '', word)
                if len(clean_word) >= 2 and clean_word.isalpha():
                    valid_words.append(clean_word.capitalize())
            
            if len(valid_words) >= 2:
                return ' '.join(valid_words)
        
        return None
    
    def _generate_fallback_name(self, filename: str = "") -> str:
        if filename:
            base = os.path.splitext(os.path.basename(filename))[0]
            clean = re.sub(r'(resume|cv|curriculum|vitae)', '', base, flags=re.IGNORECASE)
            clean = re.sub(r'[^a-zA-Z]', '', clean)
            
            if len(clean) >= 3:
                return clean[:10].capitalize() + " Candidate"
        
        hash_val = hashlib.md5(filename.encode() if filename else b"default").hexdigest()[:6]
        return f"Candidate {hash_val.upper()}"

name_extractor = NameExtractor()
