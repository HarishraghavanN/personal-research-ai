import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(uploaded_file):
    """
    Saves an uploaded file to a temporary location, loads it using PyMuPDF,
    and returns a list of text chunks.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()

        # Document chunking strategy: 1000 characters with 200 overlap to maintain semantic context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)

        # Enhance chunks with source metadata for citations
        for chunk in chunks:
            chunk.metadata['source'] = uploaded_file.name

        return chunks
    finally:
        os.remove(temp_file_path)
