# Standalone script to build a BM25 index for RAG (Retrieval-Augmented Generation)
# This script tokenizes text data, builds a BM25 index, and saves it along with metadata.

import pandas as pd
import pickle
import os
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer data (only once)
nltk.download("punkt")

# Load chunked data
df = pd.read_csv("data/processed/chunks.csv")

# Tokenize texts
texts = df["text"].tolist()
tokenized_corpus = [word_tokenize(text.lower()) for text in texts]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Extract metadata
metadata = df[["section", "chunk_id"]].to_dict(orient="records")

# Save index and metadata
os.makedirs("sparse_index_bm25", exist_ok=True)

with open("sparse_index_bm25/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open("sparse_index_bm25/tokenized_corpus.pkl", "wb") as f:
    pickle.dump(tokenized_corpus, f)

with open("sparse_index_bm25/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("BM25 index built and saved successfully!")
