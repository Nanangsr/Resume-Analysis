from utils.resume_parser import parse_resume
from typing import Tuple

def parse_jd(file) -> Tuple[str, str]:
    """Extract text from job description file"""
    return parse_resume(file, file.name if hasattr(file, 'name') else "JD File") 