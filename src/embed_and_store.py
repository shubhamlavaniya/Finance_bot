# This script is to embed and store the chunked data into a vector database using LangChain and ChromaDB

import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Load chunked CSV
df = pd.read_csv("data/processed/chunks.csv")

# Convert rows to LangChain Documents with metadata
documents = [
    Document(page_content=row["text"], metadata={
        "section": row["section"],
        "chunk_id": row["chunk_id"]
    })
    for _, row in df.iterrows()
]

# Embed & store using LangChain's Chroma
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory="vector_db"
)

vectordb.persist()
print("Embedding and storing complete!")




# Uncomment the following lines to run the code directly without langchain


# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import chromadb
# #from chromadb.config import Settings
# from tqdm import tqdm

# # Load chunks
# df = pd.read_csv("data/processed/chunks.csv")

# # Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# persist_directory = "./vector_db" # Good practice to define it as a variable


# # Create ChromaDB client

# chroma_client = chromadb.PersistentClient(path=persist_directory)


# # chroma_client = chromadb.Client(Settings(
# #     chroma_db_impl="duckdb+parquet",
# #     persist_directory="vector_db"
# # ))

# # Create a collection for Apple 10K
# collection = chroma_client.get_or_create_collection(name="apple_10k")

# # Prepare documents and metadata
# texts = df["text"].tolist()
# ids = [f"chunk_{i}" for i in range(len(texts))]

# # Embed texts
# print("Generating embeddings...")
# embeddings = model.encode(texts, show_progress_bar=True).tolist()

# # Add to ChromaDB
# collection.add(
#     documents=texts,
#     embeddings=embeddings,
#     ids=ids
# )

# # Persist the DB
# print("Embeddings stored in ChromaDB (vector_db/)")
