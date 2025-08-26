import os
import shutil
import pandas as pd
from pathlib import Path
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# A separate directory for the ArXiv vector database
ARXIV_DB_PATH = Path(__file__).resolve().parent.parent / "vector_db_arxiv"

def process_arxiv_papers(query: str, max_papers: int):
    """
    Fetches papers from ArXiv, chunks them, and stores them in a ChromaDB.
    """
    print(f"Fetching up to {max_papers} papers from ArXiv with query: '{query}'...")
    
    # Use ArxivLoader to automatically fetch and parse PDFs
    loader = ArxivLoader(query=query, load_max_docs=max_papers)
    docs = loader.load()
    
    if not docs:
        print("No documents found for the given query.")
        return

    print(f"Found {len(docs)} papers. Chunking text...")

    # Use a better text splitter for academic papers
    # It intelligently splits by paragraphs, sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    print(f"Created {len(chunks)} chunks.")

    # Clean up old database and create a new one
    if ARXIV_DB_PATH.exists():
        shutil.rmtree(ARXIV_DB_PATH)
    os.makedirs(ARXIV_DB_PATH, exist_ok=True)

    print(f"Building new ChromaDB at {ARXIV_DB_PATH}")

    # Create embeddings and save to the new database
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(ARXIV_DB_PATH)
    )

    print("ArXiv vector database built and saved.")

if __name__ == "__main__":
    # Example usage:
    # This will fetch papers about Retrieval-Augmented Generation
    process_arxiv_papers(query="Retrieval-Augmented Generation", max_papers=10)