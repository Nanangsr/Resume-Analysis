from importlib.metadata import PackageNotFoundError
import traceback
from venv import logger
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from typing import Union, List, BinaryIO, Tuple
import docx
import io
import os
import zipfile

def handle_parsing_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except PdfReadError as e:
            filename = kwargs.get('filename', args[1] if len(args) > 1 else "Unknown")
            error_msg = f"File {filename}: Kesalahan parsing PDF - {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return "", error_msg
        except PackageNotFoundError as e:
            filename = kwargs.get('filename', args[1] if len(args) > 1 else "Unknown")
            error_msg = f"File {filename}: File DOCX tidak valid - {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return "", error_msg
        except Exception as e:
            filename = kwargs.get('filename', args[1] if len(args) > 1 else "Unknown")
            error_msg = f"File {filename}: Gagal diproses - {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return "", error_msg
    return wrapper

@handle_parsing_errors
def parse_resume(file: Union[BinaryIO, str], filename: str = "") -> Tuple[str, str]:
    # Logika parsing seperti sebelumnya, tanpa try-except di dalamnya
    if isinstance(file, str):
        with open(file, 'rb') as f:
            return parse_resume(f, os.path.basename(file))
    
    if hasattr(file, 'type'):
        if file.type == "application/pdf":
            reader = PdfReader(io.BytesIO(file.read()))
            text = "\n".join([page.extract_text() for page in reader.pages])
            if not text.strip():
                error_msg = f"File {filename}: PDF tidak mengandung teks (mungkin hasil scan)"
                logger.warning(error_msg)
                return "", error_msg
            return text, ""
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file.read()))
            text = "\n".join([para.text for para in doc.paragraphs])
            if not text.strip():
                error_msg = f"File {filename}: DOCX kosong atau tidak mengandung teks"
                logger.warning(error_msg)
                return "", error_msg
            return text, ""
    return "", f"File {filename}: Format tidak didukung (harus PDF/DOCX)"

def parse_uploaded_folder(uploaded_folder) -> Tuple[List[str], List[str], List[str]]:
    """Parse semua file resume dari folder yang diupload (zip)
    Returns: (resume_texts, filenames, error_messages)"""
    resume_texts = []
    filenames = []
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
                        filenames.append(file)
                    if error:
                        error_messages.append(error)
        
        # Bersihkan direktori temporary
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)
    
    return resume_texts, filenames, error_messages
