import os
import urllib
import pandas as pd
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from tqdm import tqdm

# Define the vector DB path and the chunk file path
CHROMA_PATH = "vector_db_arxiv"
ARXIV_CHUNKS_PATH = "arxiv_chunks.csv"

def get_embedding_model():
    """Returns the SentenceTransformer embedding model."""
    return SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")

def process_arxiv_queries(queries: list):
    """
    Downloads, chunks, and adds documents for a list of queries to a vector DB.
    Also saves chunks to a CSV file.
    """
    documents = []
    print("Fetching new documents from ArXiv...")

    for query in queries:
        print(f"  - Searching for '{query}'...")
        try:
            loader = ArxivLoader(query=query, load_max_docs=50)
            documents.extend(loader.load())
        except urllib.error.HTTPError as e:
            print(f"    - Skipping document due to HTTP Error: {e}")
            continue
        except Exception as e:
            print(f"    - An unexpected error occurred: {e}")
            continue

    if not documents:
        print("No documents were successfully loaded. Exiting.")
        return

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks from {len(documents)} documents.")
    
    # Save chunks to a CSV file
    chunks_df = pd.DataFrame([
        {
            "page_content": chunk.page_content,
            "source": chunk.metadata.get("source", "N/A"),
            "title": chunk.metadata.get("title", "N/A"),
        }
        for chunk in chunks
    ])
    chunks_df.to_csv(ARXIV_CHUNKS_PATH, index=False)
    print(f"Chunks saved to '{ARXIV_CHUNKS_PATH}'.")

    # Initialize the embedding model
    embedding_model = get_embedding_model()

    # Create or load the vector database
    if not os.path.exists(CHROMA_PATH):
        print(f"Creating new vector database at '{CHROMA_PATH}'")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=CHROMA_PATH
        )
    else:
        print(f"Loading existing database from '{CHROMA_PATH}'")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
        print("Adding new documents to the existing database...")
        db.add_documents(chunks)
        
    db.persist()
    print("Database updated successfully!")

if __name__ == "__main__":
    new_topics = [
        "Large Language Models",
        "Natural Language Processing",
        #"Computer Vision",
        "Reinforcement Learning",
        "RAG",
        "Generative AI",
        "Machine Learning",
        "Deep Learning",
        "Transformers",
        "Mathematics",
        "Statistics",
        "Quantum physics",
        "Artificial Intelligence",
    ]
    process_arxiv_queries(new_topics)


# import os
# import shutil
# import pandas as pd
# from pathlib import Path
# from langchain_community.document_loaders import ArxivLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma

# # A separate directory for the ArXiv vector database
# ARXIV_DB_PATH = Path(__file__).resolve().parent.parent / "vector_db_arxiv"

# def process_arxiv_papers(query: str, max_papers: int):
#     """
#     Fetches papers from ArXiv, chunks them, and stores them in a ChromaDB.
#     """
#     print(f"Fetching up to {max_papers} papers from ArXiv with query: '{query}'...")
    
#     # Use ArxivLoader to automatically fetch and parse PDFs
#     loader = ArxivLoader(query=query, load_max_docs=max_papers)
#     docs = loader.load()
    
#     if not docs:
#         print("No documents found for the given query.")
#         return

#     print(f"Found {len(docs)} papers. Chunking text...")

#     # Use a better text splitter for academic papers
#     # It intelligently splits by paragraphs, sentences, etc.
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_documents(docs)

#     print(f"Created {len(chunks)} chunks.")

#     # Clean up old database and create a new one
#     if ARXIV_DB_PATH.exists():
#         shutil.rmtree(ARXIV_DB_PATH)
#     os.makedirs(ARXIV_DB_PATH, exist_ok=True)

#     print(f"Building new ChromaDB at {ARXIV_DB_PATH}")

#     # Create embeddings and save to the new database
#     embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         persist_directory=str(ARXIV_DB_PATH)
#     )

#     print("ArXiv vector database built and saved.")

# if __name__ == "__main__":
#     # Example usage:
#     # This will fetch papers about Retrieval-Augmented Generation
#     process_arxiv_papers(query="Retrieval-Augmented Generation", max_papers=10)