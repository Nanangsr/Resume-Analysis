from utils.resume_parser import parse_resume
from typing import Tuple

def parse_jd(file) -> Tuple[str, str]:
    """Ekstrak teks dari file deskripsi pekerjaan (job description)"""
    return parse_resume(file, file.name if hasattr(file, 'name') else "JD File") 
