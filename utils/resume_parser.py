from pypdf import PdfReader
from typing import Union, List, BinaryIO, Tuple
import docx
import io
import os
import zipfile

def parse_resume(file: Union[BinaryIO, str], filename: str = "") -> Tuple[str, str]:
    """Extract text from resume file (PDF or DOCX)
    Returns: (text, error_message)"""
    try:
        if isinstance(file, str):  # Handle file path
            with open(file, 'rb') as f:
                return parse_resume(f, os.path.basename(file))
        
        if hasattr(file, 'type'):
            if file.type == "application/pdf":
                reader = PdfReader(io.BytesIO(file.read()))
                text = "\n".join([page.extract_text() for page in reader.pages])
                if not text.strip():
                    return "", f"File {filename}: PDF tidak mengandung teks (mungkin hasil scan)"
                return text, ""
            elif file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                doc = docx.Document(io.BytesIO(file.read()))
                text = "\n".join([para.text for para in doc.paragraphs])
                if not text.strip():
                    return "", f"File {filename}: DOCX kosong atau tidak mengandung teks"
                return text, ""
        else:  # Handle file-like object
            try:
                reader = PdfReader(file)
                text = "\n".join([page.extract_text() for page in reader.pages])
                if not text.strip():
                    return "", f"File {filename}: PDF tidak mengandung teks (mungkin hasil scan)"
                return text, ""
            except:
                file.seek(0)
                doc = docx.Document(file)
                text = "\n".join([para.text for para in doc.paragraphs])
                if not text.strip():
                    return "", f"File {filename}: DOCX kosong atau tidak mengandung teks"
                return text, ""
    except Exception as e:
        return "", f"File {filename}: Gagal diproses - {str(e)}"
    
    return "", f"File {filename}: Format tidak didukung (harus PDF/DOCX)"

def parse_uploaded_folder(uploaded_folder) -> Tuple[List[str], List[str]]:
    """Parse all resume files from uploaded folder (zip)
    Returns: (resume_texts, error_messages)"""
    resume_texts = []
    error_messages = []
    
    with zipfile.ZipFile(io.BytesIO(uploaded_folder.read()), 'r') as zip_ref:
        temp_dir = "temp_uploaded_folder"
        os.makedirs(temp_dir, exist_ok=True)
        zip_ref.extractall(temp_dir)
        
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(('.pdf', '.docx', '.doc')):
                    text, error = parse_resume(file_path, file)
                    if text:
                        resume_texts.append(text)
                    if error:
                        error_messages.append(error)
        
        # Clean up
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)
    
    return resume_texts, error_messages