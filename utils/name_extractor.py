import os
import re
from typing import Optional

class NameExtractor:
    def __init__(self):
        # Prefix yang mungkin digunakan sebelum nama dalam resume
        self.name_prefixes = ["nama:", "name:", "full name:", "nama lengkap:", "candidate:", "applicant:"]
        # Kata-kata yang tidak termasuk nama
        self.exclude_words = ["resume", "cv", "curriculum", "vitae", "document", "file", "copy"]
        # Pattern untuk nama Indonesia dan internasional
        self.name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})$',  # Standard name format
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*[A-Z][a-z]+)$',  # Name with initials
            r'^([A-Z][a-z]+(?:\s+[a-z]+)*\s+[A-Z][a-z]+)$',  # Name with lowercase middle
        ]
    
    def extract_name_from_resume(self, resume_text: str, filename: str = "") -> str:
        """Ekstraksi nama dari teks resume dengan fallback ke nama file"""
        try:
            # Pertama coba ekstrak dari teks resume
            text_name = self._extract_from_text(resume_text)
            if text_name and len(text_name.split()) >= 2:
                return text_name
                
            # Jika gagal, gunakan nama file
            if filename:
                file_name = self._extract_from_filename(filename)
                if file_name:
                    return file_name
                    
        except Exception as e:
            print(f"Error extracting name: {e}")
            
        return self._generate_fallback_name(filename)
    
    def _extract_from_text(self, text: str) -> Optional[str]:
        """Ekstrak nama dari konten teks resume"""
        if not text:
            return None
            
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Pola 1: Cari label nama yang eksplisit
        for line in lines[:20]:
            line_lower = line.lower()
            for prefix in self.name_prefixes:
                if prefix in line_lower:
                    name_part = line.split(':', 1)[-1].strip()
                    cleaned_name = self._clean_name_text(name_part)
                    if cleaned_name and len(cleaned_name.split()) >= 2:
                        return cleaned_name
        
        # Pola 2: Cari kandidat nama di baris-baris awal
        potential_names = []
        for line in lines[:15]:
            # Skip lines yang terlalu panjang atau mengandung kata kunci umum
            if len(line) > 50 or any(word in line.lower() for word in self.exclude_words):
                continue
                
            # Check if line matches name patterns
            for pattern in self.name_patterns:
                if re.match(pattern, line.strip()):
                    cleaned = self._clean_name_text(line)
                    if cleaned:
                        potential_names.append(cleaned)
                        break
        
        # Pola 3: Cari nama di awal dokumen (3 baris pertama)
        if not potential_names:
            for line in lines[:3]:
                words = line.split()
                if 2 <= len(words) <= 4:
                    # Check if all words start with capital letter
                    if all(word[0].isupper() and word.isalpha() for word in words):
                        cleaned = self._clean_name_text(line)
                        if cleaned:
                            potential_names.append(cleaned)
        
        return potential_names[0] if potential_names else None
    
    def _extract_from_filename(self, filename: str) -> Optional[str]:
        """Ekstrak nama dari nama file dengan berbagai variasi"""
        if not filename:
            return None
            
        try:
            # Remove path and extension
            base_name = os.path.splitext(os.path.basename(filename))[0]
            
            # Remove common prefixes/suffixes
            base_name = re.sub(r'^(resume|cv|curriculum|vitae)[-_\s]*', '', base_name, flags=re.IGNORECASE)
            base_name = re.sub(r'[-_\s]*(resume|cv|curriculum|vitae)$', '', base_name, flags=re.IGNORECASE)
            
            # Replace separators with spaces
            base_name = re.sub(r'[-_]+', ' ', base_name)
            
            # Remove numbers and special characters, keep only letters and spaces
            clean_name = re.sub(r'[^a-zA-Z\s]', '', base_name)
            
            # Split and filter words
            words = [word for word in clean_name.split() if len(word) > 1]
            
            if len(words) >= 2:
                # Take first 2-3 meaningful words
                name_words = []
                for word in words[:4]:  # Check up to 4 words
                    if word[0].isupper() or len(name_words) == 0:  # First word or capitalized
                        name_words.append(word.capitalize())
                    if len(name_words) >= 3:  # Limit to 3 words max
                        break
                
                if len(name_words) >= 2:
                    return ' '.join(name_words)
            
            # Fallback: try to extract from original filename with different approach
            original_words = re.findall(r'[A-Z][a-z]+', filename)
            if len(original_words) >= 2:
                return ' '.join(original_words[:3])
                
        except Exception as e:
            print(f"Error extracting from filename: {e}")
            
        return None
    
    def _clean_name_text(self, text: str) -> Optional[str]:
        """Clean and validate name text"""
        if not text:
            return None
            
        # Remove extra whitespace and common prefixes
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'^(mr\.?|mrs\.?|ms\.?|dr\.?)\s*', '', text, flags=re.IGNORECASE)
        
        # Check if it's a valid name (2-4 words, each starting with capital)
        words = text.split()
        if 2 <= len(words) <= 4:
            # Each word should be mostly alphabetic and reasonable length
            valid_words = []
            for word in words:
                # Remove punctuation at the end
                clean_word = re.sub(r'[^\w]$', '', word)
                if len(clean_word) >= 2 and clean_word.isalpha():
                    valid_words.append(clean_word.capitalize())
            
            if len(valid_words) >= 2:
                return ' '.join(valid_words)
        
        return None
    
    def _generate_fallback_name(self, filename: str = "") -> str:
        """Buat nama fallback dari nama file atau generic"""
        if filename:
            try:
                # Try to extract any meaningful part from filename
                base = os.path.splitext(os.path.basename(filename))[0]
                # Remove common words and get first meaningful word
                clean = re.sub(r'(resume|cv|curriculum|vitae)', '', base, flags=re.IGNORECASE)
                clean = re.sub(r'[^a-zA-Z]', '', clean)
                
                if len(clean) >= 3:
                    return clean[:10].capitalize() + " Candidate"
            except:
                pass
        
        # Generate unique fallback
        import hashlib
        hash_val = hashlib.md5(filename.encode() if filename else b"default").hexdigest()[:6]
        return f"Candidate {hash_val.upper()}"

# Buat instance di level modul
name_extractor = NameExtractor()

def extract_name_from_resume(self, resume_text: str, filename: str) -> str:
        # Contoh implementasi aman
        match = re.search(r'NAME:\s*(.+?)\n', resume_text)
        if match:
            return match.group(1).strip()  # Return string, bukan tuple
        return name_extractor.extract_name_from_resume(resume_text, filename)
