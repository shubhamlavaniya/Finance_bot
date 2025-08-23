# This script is for RAG
# using LangChain with a hybrid retriever setup. it is using a MemoryRetriever to fetch relevant financial Q/A pairs.




import json
import os
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from typing import Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class MemoryRetriever:
    def __init__(self, memory_path: str = "../data/qa_pair.json", threshold: float = 0.9):
        self.threshold = threshold
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.memories = self._load_memory(memory_path)
        self._embed_questions()

    def _load_memory(self, path: str) -> list:
        with open(path, "r") as f:
            return json.load(f)

    def _embed_questions(self):
        questions = [item["question"] for item in self.memories]
        self.embeddings = self.embedding_model.embed_documents(questions)

    def query(self, user_query: str) -> Optional[Dict]:
        query_embedding = self.embedding_model.embed_query(user_query)
        sims = cosine_similarity([query_embedding], self.embeddings)[0]

        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]

        if best_score >= self.threshold:
            result = self.memories[best_idx]
            result["similarity"] = float(best_score)
            return result

        return None
