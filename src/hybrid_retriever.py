# This script is for Hybrid Retriever
# using LangChain with a hybrid retriever setup. it is using a combination of vector and BM25 retrievers.



import pickle
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any
from langchain_core.pydantic_v1 import Field

class HybridRetriever(BaseRetriever):
    vectordb: Any = Field(None)
    bm25_retriever: Any = Field(None)
    alpha: float = 0.5
    k: int = 4  # <-- Add a 'k' field to the class

    def __init__(self, vectordb: Any, bm25_retriever: Any, alpha=0.5, k=4):
        super().__init__()
        self.vectordb = vectordb
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha
        self.k = k

    def _get_score_dict(self, docs: List[Document]) -> dict:
        score_dict = {}
        for doc in docs:
            score_dict[doc.page_content] = doc.metadata.get("score", 1.0)
        return score_dict

    def _fuse_scores(
        self, sparse_docs: List[Document], dense_docs: List[Document]
    ) -> List[Document]:
        sparse_scores = self._get_score_dict(sparse_docs)
        dense_scores = self._get_score_dict(dense_docs)

        all_contents = set(sparse_scores.keys()).union(dense_scores.keys())
        fused_docs = []

        for content in all_contents:
            sparse_score = sparse_scores.get(content, 0.0)
            dense_score = dense_scores.get(content, 0.0)
            final_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score

            doc = next(
                (doc for doc in sparse_docs + dense_docs if doc.page_content == content),
                None,
            )
            if doc:
                doc.metadata["score"] = final_score
                fused_docs.append(doc)

        return sorted(fused_docs, key=lambda x: x.metadata["score"], reverse=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # You can set k to a higher value here for initial retrieval to get a wider pool
        dense_docs = self.vectordb.invoke(query)
        sparse_docs = self.bm25_retriever.invoke(query)
        
        # Fuse scores and return only the top 'k' documents
        fused_results = self._fuse_scores(sparse_docs, dense_docs)
        return fused_results[:self.k]  # <-- Truncate the final list here