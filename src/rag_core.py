# This script is for RAG
# using LangChain with a hybrid retriever setup.


import os
import time
from pathlib import Path
from typing import Any
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from src.hybrid_retriever import HybridRetriever

# --- Utility Functions ---

def get_openai_client():
    """Returns an OpenAI client instance."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Vector DB Loading ---

@st.cache_resource
def get_vector_db():
    """Loads the financial ChromaDB instance."""
    script_dir = Path(__file__).resolve().parent
    vector_db_path = script_dir.parent / "vector_db"

    if not vector_db_path.exists():
        st.error("Financial vector database not found. Please run the data ingestion script.")
        return None

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=str(vector_db_path),
        embedding_function=embedding_model
    )
    return vectordb

@st.cache_resource
def get_arxiv_vector_db():
    """Loads the ArXiv ChromaDB instance."""
    script_dir = Path(__file__).resolve().parent
    arxiv_db_path = script_dir.parent / "vector_db_arxiv"

    if not arxiv_db_path.exists():
        st.error("ArXiv vector database not found. Please run the arxiv_processor script.")
        return None

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=str(arxiv_db_path),
        embedding_function=embedding_model
    )
    return vectordb

# --- RAG Core Logic ---

def get_rag_response(query: str, chat_history: list = None) -> Any:
    """
    Retrieves, augments, and generates a response based on the user query.
    This function now acts as a router based on the query topic.
    """
    
    # 1. Route the query to the correct database
    query_topic = route_query_topic(query)
    
    docs = []
    response = ""
    source_url = "N/A"
    
    if query_topic == "FINANCIAL":
        vectordb = get_vector_db()
        if not vectordb:
            yield "The financial database is not available."
            return {"answer": "Error: Financial database not found.", "source": "N/A", "method": "RAG", "verification": "Error", "confidence": "Low"}
        
        # Use your existing HybridRetriever for financial data
        retriever = HybridRetriever(vectordb.as_retriever(), "data/processed/chunks.csv")
        docs = retriever.invoke(query)
        source_url = "Apple 10-K Filings"
        
    elif query_topic == "SCIENTIFIC":
        vectordb = get_arxiv_vector_db()
        if not vectordb:
            yield "The scientific database is not available."
            return {"answer": "Error: Scientific database not found.", "source": "N/A", "method": "RAG", "verification": "Error", "confidence": "Low"}
        
        # Use a standard vector retriever for scientific papers
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)
        source_url = "ArXiv API"

    else:
        # For general or irrelevant queries
        response = "I can only answer questions related to financial filings or scientific papers."
        yield response
        return {"answer": response, "source": "N/A", "method": "None", "verification": "None", "confidence": "N/A"}

    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Define the RAG prompt template
    template = """
    You are a professional financial or scientific assistant. Use the following context to answer the user's question.
    If you don't know the answer, say "I can't find that information in my knowledge base."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # 3. Create the RAG Chain with streaming
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    full_answer = ""
    
    # Stream the response back to the user
    for chunk in rag_chain.stream({"context": context, "question": query}):
        yield chunk
        full_answer += chunk

    # 4. Return the full response data
    return {
        "answer": full_answer,
        "source": source_url,
        "method": "RAG",
        "verification": "Context-based",
        "confidence": "High"
    }

# --- New Router Function ---

def route_query_topic(query: str) -> str:
    """Classifies a user query to determine the topic."""
    # Use a simple LLM call for classification
    client = get_openai_client()
    routing_prompt = f"""
    You are a query router. Your task is to classify a user's question into one of the following categories:
    - 'FINANCIAL' for questions about company financials (e.g., revenue, earnings, 10-K).
    - 'SCIENTIFIC' for questions about scientific papers or research concepts.
    - 'GENERAL' for any other type of question.

    Respond with only one word: FINANCIAL, SCIENTIFIC, or GENERAL.

    Query: {query}
    Classification:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": routing_prompt}],
            temperature=0.1,
            max_tokens=10,
        )
        classification = response.choices[0].message.content.strip().upper()
        if classification not in ["FINANCIAL", "SCIENTIFIC", "GENERAL"]:
            return "GENERAL" # Default to general if LLM response is unexpected
        return classification
    except Exception as e:
        print(f"Error in query routing: {e}")
        return "GENERAL" # Default to general in case of API error








# import os
# import shutil
# import pandas as pd
# import streamlit as st
# from pathlib import Path
# from typing import Any

# # LangChain and ChromaDB imports
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# # === NEW: Import OpenAI client directly for streaming ===
# from openai import OpenAI
# # =======================================================


# from src.hybrid_retriever import HybridRetriever
# from src.memory_retriever import MemoryRetriever

# # Note: This is good practice for Streamlit to prevent sqlite3 issues
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# # --- Cached Resources ---

# # === NEW: Use standard OpenAI client for streaming ===
# @st.cache_resource
# def get_openai_client():
#     """Loads and caches the OpenAI client."""
#     return OpenAI(
#         api_key=st.secrets["OPENAI_API_KEY"],
#     )
# # =======================================================


# @st.cache_resource
# def get_vector_db():
#     """
#     Loads or creates a persistent ChromaDB instance.
#     This function handles the ephemeral filesystem on Streamlit Cloud.
#     """
#     script_dir = Path(__file__).resolve().parent
#     data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
#     vector_db_path = script_dir.parent / "vector_db"

#     print(f"Checking for existing vector database at: {vector_db_path}")

#     if not vector_db_path.exists() or not os.listdir(vector_db_path):
#         print("Vector database not found or is empty. Building from scratch...")
        
#         if not data_path.exists():
#             print(f"Error: The file {data_path} was not found. Cannot build vector database.")
#             return None

#         try:
#             df = pd.read_csv(data_path)
#             docs = [
#                 Document(
#                     page_content=row["text"],
#                     metadata={
#                         "section": row["section"],
#                         "chunk_id": row["chunk_id"],
#                     },
#                 )
#                 for index, row in df.iterrows()
#             ]
#         except KeyError as e:
#             print(f"Error reading CSV: {e}. Ensure columns are named 'text', 'section', and 'chunk_id'.")
#             return None

#         if vector_db_path.exists():
#             shutil.rmtree(vector_db_path)

#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = Chroma.from_documents(
#             documents=docs,
#             embedding=embedding_model,
#             persist_directory=str(vector_db_path)
#         )
#         print("Vector database built and saved.")
#         return vectordb
#     else:
#         print("Vector database found. Loading from persistent storage.")
#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = Chroma(
#             persist_directory=str(vector_db_path), 
#             embedding_function=embedding_model
#         )
#         return vectordb

# @st.cache_resource
# def get_bm25_retriever():
#     """
#     Loads documents from the CSV and builds a BM25 retriever, caching the result.
#     """
#     script_dir = Path(__file__).resolve().parent
#     data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
    
#     try:
#         df = pd.read_csv(data_path)
#         docs = [
#             Document(page_content=row["text"], metadata={"section": row["section"], "chunk_id": row["chunk_id"]})
#             for index, row in df.iterrows()
#         ]
#         bm25_retriever = BM25Retriever.from_documents(docs)
#         print("BM25 retriever built and cached.")
#         return bm25_retriever
#     except Exception as e:
#         print(f"Error building BM25 retriever: {e}")
#         return None

# # --- Guardrail Functions ---

# def validate_query(llm_client: Any, query: str) -> str:
#     """Classifies a user query as relevant, irrelevant, or harmful using an LLM."""
#     validation_prompt_template = """
#     You are a financial query validator. Your task is to determine if a user's query is related to financial topics, a company's 10-K filing, or is a harmful request.

#     <Instructions>
#     - If the query is about a company's financial performance, a specific financial term (e.g., revenue, EPS), a financial document (e.g., 10-K), or investment advice, classify it as 'RELEVANT_FINANCIAL'.
#     - If the query is a personal, non-financial, or general knowledge question, classify it as 'IRRELEVANT'.
#     - If the query contains any harmful, unethical, dangerous, or illegal content, classify it as 'HARMFUL'.

#     Output format should be a single line with the classification and a brief explanation.
#     e.g., Classification: RELEVANT_FINANCIAL. Explanation: The query asks about a company's revenue.
#     e.g., Classification: IRRELEVANT. Explanation: The query is a general knowledge question.
#     e.g., Classification: HARMFUL. Explanation: The query contains harmful content.
#     </Instructions>

#     Query: {query}
#     """
    
#     response = llm_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": validation_prompt_template.format(query=query)}],
#         temperature=0.1,
#         max_tokens=256,
#     )
#     return response.choices[0].message.content.strip().split('\n')[0].split(':')[-1].strip()

# def verify_answer(llm: Any, answer: str, source_documents: list) -> str:
#     """Verifies if the answer is supported by the source documents."""
#     source_texts = "\n\n".join([doc.page_content for doc in source_documents])
#     verification_prompt_template = """
#     You are a fact-checker. Your task is to determine if the given Answer is directly and entirely supported by the provided Source Documents.

#     <Instructions>
#     - If the Answer is fully supported by the Source Documents, output 'VERIFIED'.
#     - If the Answer is not fully supported, contains information not in the Source Documents, or is a hallucination, output 'UNVERIFIED'.

#     Output format should be a single line with the verification status and a brief explanation.
#     e.g., Verification: VERIFIED. Explanation: The answer is directly from the source documents.
#     e.g., Verification: UNVERIFIED. Explanation: The answer contains information not found in the source documents.
#     </Instructions>

#     Answer: {answer}
#     Source Documents: {source_docs}
#     """
#     verification_prompt = PromptTemplate.from_template(verification_prompt_template)
#     verification_chain = verification_prompt | llm
#     response = verification_chain.invoke({"answer": answer, "source_docs": source_texts})
#     return response.strip()




# # --- Main RAG Pipeline Function ---

# # === get_rag_response now uses the LLM-based validation ===
# def get_rag_response(query: str):
#     """
#     The main RAG function to get a response with guardrails,
#     which yields chunks for streaming.
#     """
#     script_dir = Path(__file__).resolve().parent
#     memory_path = script_dir.parent / "data" / "qa_pair.json"  
#     memory_retriever = MemoryRetriever(memory_path=memory_path, threshold=0.9)

#     # Check memory bank (qa_pair.json)
#     memory_result = memory_retriever.query(query)
    
#     if memory_result:
#         yield memory_result["answer"]
#         return # Return to stop the generator
    
#     else:
#         # ONLY if not in memory bank: Use LLM-based validation
#         client = get_openai_client()
#         validation_result = validate_query(llm_client=client, query=query)
#         print(f"Validation: {validation_result}")
#         if validation_result not in ["RELEVANT_FINANCIAL"]:
#             yield "I can only answer relevant financial questions."
#             return # Stop the generator

#         # Get cached resources
#         vectordb = get_vector_db()
#         bm25_retriever = get_bm25_retriever()

#         if not vectordb or not bm25_retriever:
#             yield "Error: Retrievers not available."
#             return # Stop the generator

#         hybrid_retriever = HybridRetriever(
#             vectordb=vectordb.as_retriever(),
#             bm25_retriever=bm25_retriever,
#             alpha=0.5,
#             k=4
#         )
        
#         # Get documents and context
#         docs = hybrid_retriever.invoke(query)
#         docs = docs[:3]

#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         prompt_messages = [{
#             "role": "system",
#             "content": """You are a financial expert analyzing Apple's 10-K filings. 
# Use the provided context from the documents to answer the question accurately.
# ... [full prompt from your original code] ..."""
#         }, {
#             "role": "user",
#             "content": f"Context: {context}\nQuestion: {query}\nAnswer:"
#         }]
        
#         stream = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=prompt_messages,
#             temperature=0.1,
#             max_tokens=512,
#             stream=True
#         )

#         for chunk in stream:
#             content = chunk.choices[0].delta.content or ""
#             yield content

#         # === Note: We've removed the verification step for streaming. 
#         # You'll need to add it back for the final answer if needed.
#         # This can be done by capturing the full streamed response
#         # and then calling verify_answer on it, but it adds latency.
#         # This is a trade-off for the streaming effect.
#         # =======================================================


#---- without streaming code------
# import os
# import shutil
# import pandas as pd
# import streamlit as st
# from pathlib import Path
# from typing import Any

# # LangChain and ChromaDB imports
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# #from langchain_community.llms import HuggingFaceHub
# #from langchain_community.llms import HuggingFaceEndpoint
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_community.llms import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain


# from src.hybrid_retriever import HybridRetriever
# from src.memory_retriever import MemoryRetriever

# # Note: This is good practice for Streamlit to prevent sqlite3 issues
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# #api_token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)

# # --- Cached Resources ---

# @st.cache_resource
# def get_rag_llm():
#     """Loads OpenAI LLM for RAG"""
#     from langchain_community.llms import OpenAI
    
#     llm = OpenAI(
#         openai_api_key=st.secrets["OPENAI_API_KEY"],  # ← Your OpenAI key
#         model="gpt-4o-mini",  # ← OpenAI model
#         temperature=0.1,
#         max_tokens=512
#     )
#     return llm

# @st.cache_resource
# def get_vector_db():
#     """
#     Loads or creates a persistent ChromaDB instance.
#     This function handles the ephemeral filesystem on Streamlit Cloud.
#     """
#     script_dir = Path(__file__).resolve().parent
#     data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
#     vector_db_path = script_dir.parent / "vector_db"

#     print(f"Checking for existing vector database at: {vector_db_path}")

#     if not vector_db_path.exists() or not os.listdir(vector_db_path):
#         print("Vector database not found or is empty. Building from scratch...")
        
#         if not data_path.exists():
#             print(f"Error: The file {data_path} was not found. Cannot build vector database.")
#             return None

#         try:
#             df = pd.read_csv(data_path)
#             docs = [
#                 Document(
#                     page_content=row["text"],
#                     metadata={
#                         "section": row["section"],
#                         "chunk_id": row["chunk_id"],
#                     },
#                 )
#                 for index, row in df.iterrows()
#             ]
#         except KeyError as e:
#             print(f"Error reading CSV: {e}. Ensure columns are named 'text', 'section', and 'chunk_id'.")
#             return None

#         if vector_db_path.exists():
#             shutil.rmtree(vector_db_path)

#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = Chroma.from_documents(
#             documents=docs,
#             embedding=embedding_model,
#             persist_directory=str(vector_db_path)
#         )
#         print("Vector database built and saved.")
#         return vectordb
#     else:
#         print("Vector database found. Loading from persistent storage.")
#         embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#         vectordb = Chroma(
#             persist_directory=str(vector_db_path), 
#             embedding_function=embedding_model
#         )
#         return vectordb

# @st.cache_resource
# def get_bm25_retriever():
#     """
#     Loads documents from the CSV and builds a BM25 retriever, caching the result.
#     """
#     script_dir = Path(__file__).resolve().parent
#     data_path = script_dir.parent / "data" / "processed" / "chunks.csv"
    
#     try:
#         df = pd.read_csv(data_path)
#         docs = [
#             Document(page_content=row["text"], metadata={"section": row["section"], "chunk_id": row["chunk_id"]})
#             for index, row in df.iterrows()
#         ]
#         bm25_retriever = BM25Retriever.from_documents(docs)
#         print("BM25 retriever built and cached.")
#         return bm25_retriever
#     except Exception as e:
#         print(f"Error building BM25 retriever: {e}")
#         return None

# # --- Guardrail Functions ---

# def validate_query(llm: Any, query: str) -> str:
#     """Classifies a user query as relevant, irrelevant, or harmful."""
#     validation_prompt_template = """
#     You are a financial query validator. Your task is to determine if a user's query is related to financial topics, a company's 10-K filing, or is a harmful request.

#     <Instructions>
#     - If the query is about a company's financial performance, a specific financial term (e.g., revenue, EPS), a financial document (e.g., 10-K), or investment advice, classify it as 'RELEVANT_FINANCIAL'.
#     - If the query is a personal, non-financial, or general knowledge question, classify it as 'IRRELEVANT'.
#     - If the query contains any harmful, unethical, dangerous, or illegal content, classify it as 'HARMFUL'.

#     Output format should be a single line with the classification and a brief explanation.
#     e.g., Classification: RELEVANT_FINANCIAL. Explanation: The query asks about a company's revenue.
#     e.g., Classification: IRRELEVANT. Explanation: The query is a general knowledge question.
#     e.g., Classification: HARMFUL. Explanation: The query contains harmful content.
#     </Instructions>

#     Query: {query}
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

#     <Instructions>
#     - If the Answer is fully supported by the Source Documents, output 'VERIFIED'.
#     - If the Answer is not fully supported, contains information not in the Source Documents, or is a hallucination, output 'UNVERIFIED'.

#     Output format should be a single line with the verification status and a brief explanation.
#     e.g., Verification: VERIFIED. Explanation: The answer is directly from the source documents.
#     e.g., Verification: UNVERIFIED. Explanation: The answer contains information not found in the source documents.
#     </Instructions>

#     Answer: {answer}
#     Source Documents: {source_docs}
#     """
#     verification_prompt = PromptTemplate.from_template(verification_prompt_template)
#     verification_chain = verification_prompt | llm
#     response = verification_chain.invoke({"answer": answer, "source_docs": source_texts})
#     return response.strip()


# def validate_query_simple(query: str) -> str:
#     """Simple rule-based query validation without API calls."""
#     query_lower = query.lower()
    
#     # Harmful patterns
#     harmful_patterns = [
#         "harm", "attack", "malware", "virus", "hack", "exploit",
#         "self-harm", "suicide", "kill", "destroy", "bomb", "weapon"
#     ]
    
#     if any(pattern in query_lower for pattern in harmful_patterns):
#         return "HARMFUL"
    
#     # Financial keywords (from Apple 10-K context)
#     financial_keywords = [
#         "revenue", "income", "profit", "financial", "balance", "cash flow",
#         "10-k", "apple", "financial statement", "earnings", "margin",
#         "assets", "liabilities", "equity", "dividend", "investment",
#         "stock", "share", "ipo", "market cap", "valuation", "growth",
#         "sales", "expenses", "rd", "research", "development", "tax",
#         "debt", "credit", "loan", "interest", "currency", "exchange",
#         "segment", "geographic", "product", "service", "iphone", "mac",
#         "ipad", "wearables", "app store", "cloud", "subscription"
#     ]
    
#     if any(keyword in query_lower for keyword in financial_keywords):
#         return "RELEVANT_FINANCIAL"
    
#     return "IRRELEVANT"

# # --- Main RAG Pipeline Function ---

# def get_rag_response(query: str):
#     """The main RAG function to get a response with guardrails."""

#     script_dir = Path(__file__).resolve().parent
#     memory_path = script_dir.parent / "data" / "qa_pair.json"  
#     memory_retriever = MemoryRetriever(memory_path=memory_path, threshold=0.9)

#     # 1. FIRST: Check memory bank (qa_pair.json)
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
#         # 2. ONLY if not in memory bank: Use vector DB + OpenAI
#         # Get cached LLM for validation
#         llm = get_rag_llm()
#         if not llm:
#             return {"answer": "Error: LLM not available.", "method": "Error"}
            
#         validation_result = validate_query_simple(query)
#         print(f"Validation: {validation_result}")
#         if validation_result not in ["RELEVANT_FINANCIAL"]:
#             return {
#                 "answer": "I can only answer relevant financial questions.",
#                 "source": "None",
#                 "method": "Guardrail",
#                 "confidence": "N/A",
#                 "verification": "N/A"
#             }

#         # Get cached resources
#         vectordb = get_vector_db()
#         bm25_retriever = get_bm25_retriever()

#         if not vectordb or not bm25_retriever:
#             return {"answer": "Error: Retrievers not available.", "method": "Error"}

#         hybrid_retriever = HybridRetriever(
#             vectordb=vectordb.as_retriever(),
#             bm25_retriever=bm25_retriever,
#             alpha=0.5,
#             k=4
#         )
        
#         # === NEW: PROPER OPENAI PROMPTING ===
        
        
#         # Create optimized prompt template
#         prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""You are a financial expert analyzing Apple's 10-K filings. 
# Use the provided context from the documents to answer the question accurately.

# CONTEXT:
# {context}

# QUESTION: 
# {question}

# INSTRUCTIONS:
# - Answer based ONLY on the context provided
# - If the context doesn't contain the answer, say "I cannot find this information in the available documents"
# - Be concise and factual
# - Use financial terminology appropriately

# ANSWER:
# """
#         )
        
#         # Create LLM chain with proper prompting
#         qa_chain = LLMChain(
#             llm=llm,
#             prompt=prompt_template,
#             verbose=False
#         )
        
#         # Get documents and context
#         docs = hybrid_retriever.invoke(query)
#         docs = docs[:3]  # ← ONLY TAKE TOP 3 DOCUMENTS

#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         # Generate answer with proper context
#         result = qa_chain.invoke({
#             "context": context,
#             "question": query
#         })
        
#         docs_with_scores = vectordb.similarity_search_with_score(query, k=1)
#         confidence_score = 1 - docs_with_scores[0][1] if docs_with_scores else "N/A"
        
#         verification_status = verify_answer(llm, result["text"], docs)
        
#         return {
#             "answer": result["text"],
#             "source_docs": docs,
#             "method": "Hybrid RAG (OpenAI)",
#             "verification": verification_status,
#             "confidence": confidence_score
#         }

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