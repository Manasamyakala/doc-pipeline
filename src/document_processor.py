import os
import tempfile
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_pdfs(uploaded_files: List[Any], chunk_size: int = 200, chunk_overlap: int = 2) -> List[Document]:
    """
    Saves uploaded PDF files from Streamlit temporarily, loads their content using PyPDFLoader, 
    and splits the extracted text into manageable chunks.
    
    Args:
        uploaded_files (List[UploadedFile]): A list of file objects uploaded via Streamlit.
        chunk_size (int): The maximum number of characters per chunk.
        chunk_overlap (int): The number of characters to overlap between adjacent chunks.
        
    Returns:
        List[Document]: A list of Langchain Document objects containing the chunked text and metadata.
    """
    if not uploaded_files:
        return []
        
    all_chunks = []
    
    # Configure the text splitter based on reference notebook defaults
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Streamlit UploadedFiles are in-memory, we need to save them temporarily for PyPDFLoader
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            # Load the PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Add metadata about source filename
            for doc in documents:
                doc.metadata["source_file"] = uploaded_file.name
                
            # Split the document into chunks
            chunks = splitter.split_documents(documents)
            all_chunks.extend(chunks)
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
                
    return all_chunks
