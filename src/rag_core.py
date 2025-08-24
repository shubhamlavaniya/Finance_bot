# This script is for RAG
# using LangChain with a hybrid retriever setup.


import os
import shutil
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Any

# LangChain and ChromaDB imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.hybrid_retriever import HybridRetriever
from src.memory_retriever import MemoryRetriever

# Note: This is good practice for Streamlit to prevent sqlite3 issues
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)

# --- Cached Resources ---

@st.cache_resource
def get_rag_llm():
    """Loads the Hugging Face LLM via Inference API and caches it."""
    print("Loading Hugging Face LLM via Inference API...")
    llm = HuggingFaceHub(
        repo_id="microsoft/phi-2",
        huggingfacehub_api_token=api_token,
        temperature=0.1,
        max_new_tokens=512,
        do_sample=True
    )
    print("Successfully connected to Hugging Face Inference API.")
    return llm

@st.cache_resource
def get_vector_db():
    """
    Loads or creates a persistent ChromaDB instance.
    This function handles the ephemeral filesystem on Streamlit Cloud.
    """
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
    vector_db_path = script_dir.parent / "vector_db"

    print(f"Checking for existing vector database at: {vector_db_path}")

    if not vector_db_path.exists() or not os.listdir(vector_db_path):
        print("Vector database not found or is empty. Building from scratch...")
        
        if not data_path.exists():
            print(f"Error: The file {data_path} was not found. Cannot build vector database.")
            return None

        try:
            df = pd.read_csv(data_path)
            docs = [
                Document(
                    page_content=row["text"],
                    metadata={
                        "section": row["section"],
                        "chunk_id": row["chunk_id"],
                    },
                )
                for index, row in df.iterrows()
            ]
        except KeyError as e:
            print(f"Error reading CSV: {e}. Ensure columns are named 'text', 'section', and 'chunk_id'.")
            return None

        if vector_db_path.exists():
            shutil.rmtree(vector_db_path)

        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=str(vector_db_path)
        )
        print("Vector database built and saved.")
        return vectordb
    else:
        print("Vector database found. Loading from persistent storage.")
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma(
            persist_directory=str(vector_db_path), 
            embedding_function=embedding_model
        )
        return vectordb

@st.cache_resource
def get_bm25_retriever():
    """
    Loads documents from the CSV and builds a BM25 retriever, caching the result.
    """
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
    
    try:
        df = pd.read_csv(data_path)
        docs = [
            Document(page_content=row["text"], metadata={"section": row["section"], "chunk_id": row["chunk_id"]})
            for index, row in df.iterrows()
        ]
        bm25_retriever = BM25Retriever.from_documents(docs)
        print("BM25 retriever built and cached.")
        return bm25_retriever
    except Exception as e:
        print(f"Error building BM25 retriever: {e}")
        return None

# --- Guardrail Functions ---

def validate_query(llm: Any, query: str) -> str:
    """Classifies a user query as relevant, irrelevant, or harmful."""
    validation_prompt_template = """
    You are a financial query validator. Your task is to determine if a user's query is related to financial topics, a company's 10-K filing, or is a harmful request.

    <Instructions>
    - If the query is about a company's financial performance, a specific financial term (e.g., revenue, EPS), a financial document (e.g., 10-K), or investment advice, classify it as 'RELEVANT_FINANCIAL'.
    - If the query is a personal, non-financial, or general knowledge question, classify it as 'IRRELEVANT'.
    - If the query contains any harmful, unethical, dangerous, or illegal content, classify it as 'HARMFUL'.

    Output format should be a single line with the classification and a brief explanation.
    e.g., Classification: RELEVANT_FINANCIAL. Explanation: The query asks about a company's revenue.
    e.g., Classification: IRRELEVANT. Explanation: The query is a general knowledge question.
    e.g., Classification: HARMFUL. Explanation: The query contains harmful content.
    </Instructions>

    Query: {query}
    """
    validation_prompt = PromptTemplate.from_template(validation_prompt_template)
    validation_chain = validation_prompt | llm
    response = validation_chain.invoke({"query": query})
    return response.strip().split('\n')[0].split(':')[-1].strip()

def verify_answer(llm: Any, answer: str, source_documents: list) -> str:
    """Verifies if the answer is supported by the source documents."""
    source_texts = "\n\n".join([doc.page_content for doc in source_documents])
    verification_prompt_template = """
    You are a fact-checker. Your task is to determine if the given Answer is directly and entirely supported by the provided Source Documents.

    <Instructions>
    - If the Answer is fully supported by the Source Documents, output 'VERIFIED'.
    - If the Answer is not fully supported, contains information not in the Source Documents, or is a hallucination, output 'UNVERIFIED'.

    Output format should be a single line with the verification status and a brief explanation.
    e.g., Verification: VERIFIED. Explanation: The answer is directly from the source documents.
    e.g., Verification: UNVERIFIED. Explanation: The answer contains information not found in the source documents.
    </Instructions>

    Answer: {answer}
    Source Documents: {source_docs}
    """
    verification_prompt = PromptTemplate.from_template(verification_prompt_template)
    verification_chain = verification_prompt | llm
    response = verification_chain.invoke({"answer": answer, "source_docs": source_texts})
    return response.strip()


def validate_query_simple(query: str) -> str:
    """Simple rule-based query validation without API calls."""
    query_lower = query.lower()
    
    # Harmful patterns
    harmful_patterns = [
        "harm", "attack", "malware", "virus", "hack", "exploit",
        "self-harm", "suicide", "kill", "destroy", "bomb", "weapon"
    ]
    
    if any(pattern in query_lower for pattern in harmful_patterns):
        return "HARMFUL"
    
    # Financial keywords (from Apple 10-K context)
    financial_keywords = [
        "revenue", "income", "profit", "financial", "balance", "cash flow",
        "10-k", "apple", "financial statement", "earnings", "margin",
        "assets", "liabilities", "equity", "dividend", "investment",
        "stock", "share", "ipo", "market cap", "valuation", "growth",
        "sales", "expenses", "rd", "research", "development", "tax",
        "debt", "credit", "loan", "interest", "currency", "exchange",
        "segment", "geographic", "product", "service", "iphone", "mac",
        "ipad", "wearables", "app store", "cloud", "subscription"
    ]
    
    if any(keyword in query_lower for keyword in financial_keywords):
        return "RELEVANT_FINANCIAL"
    
    return "IRRELEVANT"

# --- Main RAG Pipeline Function ---

def get_rag_response(query: str):
    """The main RAG function to get a response with guardrails."""

    script_dir = Path(__file__).resolve().parent
    memory_path = script_dir.parent / "data" / "qa_pair.json"  
    memory_retriever = MemoryRetriever(memory_path=memory_path, threshold=0.9)

    memory_result = memory_retriever.query(query)
    
    if memory_result:
        return {
            "answer": memory_result["answer"],
            "source": memory_result["source"],
            "method": "Memory Bank",
            "confidence": memory_result.get("similarity", "N/A"),
            "verification": "VERIFIED"
        }
    else:
        # Get cached LLM for validation
        llm = get_rag_llm()
        if not llm:
            return {"answer": "Error: LLM not available.", "method": "Error"}
            
        validation_result = validate_query_simple(query)
        print(f"Validation: {validation_result}")
        if validation_result not in ["RELEVANT_FINANCIAL"]:
            return {
                "answer": "I can only answer relevant financial questions.",
                "source": "None",
                "method": "Guardrail",
                "confidence": "N/A",
                "verification": "N/A"
            }

        # Get cached resources
        vectordb = get_vector_db()
        bm25_retriever = get_bm25_retriever()

        if not vectordb or not bm25_retriever:
            return {"answer": "Error: Retrievers not available.", "method": "Error"}

        hybrid_retriever = HybridRetriever(
            vectordb=vectordb.as_retriever(),
            bm25_retriever=bm25_retriever,
            alpha=0.5,
            k=4
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=hybrid_retriever,
            return_source_documents=True
        )
        
        docs_with_scores = vectordb.similarity_search_with_score(query, k=1)
        confidence_score = 1 - docs_with_scores[0][1] if docs_with_scores else "N/A"
        
        result = qa_chain.invoke({"query": query})
        
        verification_status = verify_answer(llm, result["result"], result["source_documents"])
        
        return {
            "answer": result["result"],
            "source_docs": result["source_documents"],
            "method": "Hybrid RAG",
            "verification": verification_status,
            "confidence": confidence_score
        }

# uncomment this script if want to use ollama framework        

# import os
# import pandas as pd
# import time
# from pathlib import Path
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# #from langchain_chroma import Chroma
# #from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.llms import Ollama
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from typing import List, Any
# # from langfuse import Langfuse  <- This line was not present

# from src.hybrid_retriever import HybridRetriever
# from src.memory_retriever import MemoryRetriever

# # --- Guardrail Functions ---

# def validate_query(llm: Any, query: str) -> str:
#     """Classifies a user query as relevant, irrelevant, or harmful."""
#     validation_prompt_template = """
#     You are a financial query validator. Your task is to determine if a user's query is related to financial topics, a company's 10-K filing, or is a harmful request.

#     Classify the query into one of the following categories:
#     - RELEVANT_FINANCIAL: The query is about finance, a company, or a 10-K filing.
#     - IRRELEVANT: The query is unrelated to finance.
#     - HARMFUL: The query is a harmful, unethical, or dangerous request.

#     Query: {query}

#     Classification:
#     """
#     validation_prompt = PromptTemplate.from_template(validation_prompt_template)
#     validation_chain = validation_prompt | llm
#     response = validation_chain.invoke({"query": query})
#     return response.strip().split('\n')[0].split(':')[-1].strip()


# def verify_answer(llm: Any, answer: str, source_documents: list) -> str:
#     """Verifies if the answer is supported by the source documents."""
#     source_texts = "\n\n".join([doc.page_content for doc in source_documents])
#     verification_prompt_template = """
#     You are a fact-checker. Your task is to determine if the given Answer is directly and entirely supported by the provided Source Documents.

#     Instructions:
#     1. Read the Answer carefully.
#     2. Read all Source Documents.
#     3. Determine if every claim in the Answer can be found and verified in the Source Documents.
#     4. If the Answer is fully supported, respond with 'VERIFIED'.
#     5. If any part of the Answer is not supported, fabricated, or a contradiction, respond with 'HALLUCINATION'.

#     Answer: {answer}

#     Source Documents:
#     {source_docs}

#     Verification Status:
#     """
#     verification_prompt = PromptTemplate.from_template(verification_prompt_template)
#     verification_chain = verification_prompt | llm
#     response = verification_chain.invoke({"answer": answer, "source_docs": source_texts})
#     return response.strip()


# # --- Main RAG Pipeline Setup ---

# # Get the absolute path to the directory containing this script
# script_dir = Path(__file__).resolve().parent

# # Load and process documents to create the BM25Retriever
# data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
# if not data_path.exists():
#     raise FileNotFoundError(f"Ensure that {data_path} exists and contains your document chunks.")

# try:
#     df = pd.read_csv(data_path)
#     docs = [
#         Document(
#             page_content=row["text"],
#             metadata={
#                 "section": row["section"],
#                 "chunk_id": row["chunk_id"],
#             },
#         )
#         for index, row in df.iterrows()
#     ]
# except KeyError as e:
#     raise KeyError(f"Error reading CSV. Make sure your columns are named 'text', 'section', and 'chunk_id'. Original error: {e}")

# bm25_retriever = BM25Retriever.from_documents(docs)

# # Load embeddings and vector DB
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

# # Construct the absolute path to qa_pair.json
# memory_path = script_dir.parent / "data" / "qa_pair.json"    
# memory_retriever = MemoryRetriever(memory_path=memory_path, threshold=0.9)


# # Instantiate HybridRetriever with the correct objects
# hybrid_retriever = HybridRetriever(
#     vectordb=vectordb.as_retriever(),
#     bm25_retriever=bm25_retriever,
#     alpha=0.5,
#     k=4
# )

# # Load LLM
# #llm = Ollama(model="gemma:2b")
# #llm = Ollama(model="mistral")
# #llm = Ollama(model="phi3:mini")
# llm = Ollama(model="deepseek-r1:1.5b")

# # Setup QA chain with hybrid retriever
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=hybrid_retriever,
#     return_source_documents=True
# )

# def get_rag_response(query: str):
#     """The main RAG function to get a response with guardrails."""

#     # First check memory bank
#     memory_result = memory_retriever.query(query)
    
#     if memory_result:
#         return {
#             "answer": memory_result["answer"],
#             "source": memory_result["source"],
#             "method": "Memory Bank",
#             "confidence": memory_result.get("similarity", "N/A"),
#             "verification": "VERIFIED"
#         }
#     else:
#         docs_with_scores = vectordb.similarity_search_with_score(query, k=1)
#         confidence_score = "N/A"
#         if docs_with_scores:
#             # The score from similarity search is a distance, so we subtract from 1 to get a confidence
#             confidence_score = 1 - docs_with_scores[0][1]
#         result = qa_chain.invoke({"query": query})
        
#         # Output-side Guardrail
#         verification_status = verify_answer(llm, result["result"], result["source_documents"])
        
#         return {
#             "answer": result["result"],
#             "source_docs": result["source_documents"],
#             "method": "Hybrid RAG",
#             "verification": verification_status,
#             "confidence": confidence_score
#         }