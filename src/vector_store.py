import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIRECTORY = "./chroma_db"

def get_vector_store():
    # Use HuggingFace embeddings which run locally and don't require an API key by default
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize Chroma DB
    vector_store = Chroma(
        collection_name="research_papers",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    return vector_store

def add_documents_to_store(chunks):
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    return vector_store
