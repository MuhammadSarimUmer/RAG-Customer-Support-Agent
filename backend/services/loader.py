import tempfile
import os
from typing import List
from fastapi import UploadFile
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document


async def load_file_content(file: UploadFile) -> List[Document]:
    filename = file.filename or "unknown"
    _, ext = os.path.splitext(filename)
    ext = ext.lower().strip()
    
    temp_file_path = None
    
    try:
   
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        if ext == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
        elif ext == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(temp_file_path, mode="elements")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        return loader.load()
        
    except Exception as e:
        raise Exception(f"Failed parsing '{file.filename}': {str(e)}")
        
    finally:
        
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
