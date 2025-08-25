# This script is for Fine-Tuning (FT) and will be used to answer financial questions using fine tuned model



import streamlit as st
import time
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# --- Paths / defaults ---
script_dir = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "microsoft/phi-2"
QA_JSON = script_dir.parent / "data" / "qa_pair.json"
QA_TXT = script_dir.parent / "data" / "qa_gpt2.txt"

# --- Simple Validation Function (ADDED) ---
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

# --- Cached Model & Resource Loading ---


# In src/ft_core.py

@st.cache_resource
def load_ft_model_and_tokenizer():
    """Loads and caches the fine-tuned model and its tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {device}")

    try:
        # Define the path to your locally saved PEFT adapter
        adapter_path = Path(__file__).resolve().parent.parent / "models" / "financial_tiny2_v1"
        
        # 1. Load the base model (TinyLlama)
        # This will download the base model from Hugging Face Hub on the first run.
        base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

        # 2. Load the PEFT adapter from your local directory
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 3. Merge the adapter weights into the base model
        model = model.merge_and_unload()
        
        model.eval()
        print("Successfully loaded fine-tuned model from base + adapter")
        return model, tokenizer, device
    
    except Exception as e:
        st.error(f"Failed to load fine-tuned model: {e}")
        return None, None, None

@st.cache_resource
def load_memory_resources():
    """Loads and caches memory Q/A pairs and the Sentence Transformer."""
    def _load_qa_pairs():
        pairs = []
        if QA_JSON.exists():
            with open(QA_JSON, "r", encoding="utf-8") as f:
                pairs = json.load(f)
        elif QA_TXT.exists():
            with open(QA_TXT, "r", encoding="utf-8") as f:
                lines = f.read().split("\n")
                q, a = None, None
                for line in lines:
                    if line.startswith("Q:"):
                        q = line[2:].strip()
                    elif line.startswith("A:"):
                        a = line[2:].strip()
                    elif line.strip() == "" and q and a:
                        pairs.append({"question": q, "answer": a})
                        q, a = None, None
                if q and a:
                    pairs.append({"question": q, "answer": a})
        return pairs

    qa_pairs = _load_qa_pairs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    answer_corpus = [p["answer"] for p in qa_pairs] if qa_pairs else []
    answer_embeddings = embedder.encode(answer_corpus) if answer_corpus else None
    
    return qa_pairs, embedder, answer_embeddings, device

# --- Helper Functions (unchanged from your original code) ---
FINANCE_KEYWORDS = {
    "revenue", "net sales", "income", "operating income", "cash flow", "balance sheet", 
    "income statement", "assets", "liabilities", "eps", "earnings per share", "dividends", 
    "gross margin", "operating margin", "guidance", "10-k", "10q", "financial", "quarter", 
    "fiscal", "segment", "geographic", "capex", "r&d", "share repurchase", "buyback", 
    "opex", "cost of sales", "goodwill", "services", "iphone", "mac", "ipad", "spend", 
    "expenditure", "expense", "cost", "marketing", "advertising", "profit", "loss", 
    "amortization", "depreciation", "valuation", "debt", "equity", "stock", "share", 
    "Apple", "Microsoft", "Google", "investment", "portfolio", "risk", "return", "diversification"
}


DECODE = {
    "num_beams": 5, "max_new_tokens": 80, "early_stopping": True, 
    "no_repeat_ngram_size": 2, "repetition_penalty": 1.1, "temperature": 0.3
}

def _postprocess(text: str) -> str:
    sents = [s.strip() for s in text.split(".") if s.strip()]
    uniq = []
    for s in sents:
        if s not in uniq:
            uniq.append(s)
    out = ". ".join(uniq)
    return out + ("." if out and not out.endswith(".") else "")

def _verify_with_memory(generated: str, question: str = None):
    """Verify generated answer against training data."""
    qa_pairs, embedder, answer_embeddings, _ = load_memory_resources()
    
    if not generated.strip() or answer_embeddings is None:
        return "NOT_VERIFIED", 0.0, "Invalid input"
    
    try:
        gen_embedding = embedder.encode([generated])[0]
        best_sim = -1
        best_ref = None
        best_question = None
        
        similarities = util.cos_sim(gen_embedding, answer_embeddings)[0]
        
        for i, sim in enumerate(similarities):
            if sim > best_sim:
                best_sim = sim
                best_ref = qa_pairs[i]["answer"]
                best_question = qa_pairs[i]["question"]
        
        if question and any(q["question"].lower() == question.lower() for q in qa_pairs):
            return "VERIFIED", 1.0, "Exact training question match"
        
        if best_sim > 0.85:
            return "VERIFIED", float(best_sim), best_ref
        elif best_sim > 0.7:
            return "PARTIALLY_VERIFIED", float(best_sim), best_ref
        else:
            return "NOT_VERIFIED", float(best_sim), best_ref
            
    except Exception as e:
        return "NOT_VERIFIED", 0.0, f"Error: {str(e)}"

# --- Main FT Pipeline Function ---
def get_ft_response(question: str):
    """Uniform response shape with RAG for the UI."""
    # This will get the cached model and resources
    model, tokenizer, device = load_ft_model_and_tokenizer()
    qa_pairs, _, _, _ = load_memory_resources()
    
    if not all([model, tokenizer, qa_pairs]):
        return {
            "answer": "Error: A required model or resource could not be loaded.",
            "verification": "Error",
            "confidence": 0.0,
            "method": "Fine-tuned",
            "response_time": 0.0
        }
    
    guard = validate_query_simple(question)
    
    if guard in ("IRRELEVANT", "HARMFUL"):
        return {
            "answer": f"Your query was flagged as **{guard}**. Please ask a relevant financial question.",
            "verification": "BLOCKED",
            "confidence": 0.0,
            "method": "Fine-tuned",
            "response_time": 0.0
        }
    
    exact_match = None
    for qa in qa_pairs:
        if qa["question"].lower() == question.lower():
            exact_match = qa["answer"]
            break
    
    if exact_match:
        return {
            "answer": exact_match,
            "verification": "VERIFIED (exact match)",
            "confidence": 1.0,
            "method": "Fine-tuned",
            "response_time": 0.1,
            "source": "Training data"
        }
    
    prompt = f"Question: {question} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=DECODE["num_beams"],
            max_new_tokens=DECODE["max_new_tokens"],
            early_stopping=DECODE["early_stopping"],
            no_repeat_ngram_size=DECODE["no_repeat_ngram_size"],
            repetition_penalty=DECODE["repetition_penalty"],
            temperature=DECODE["temperature"],
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = round(time.time() - start, 2)
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if "Answer:" in full_text:
        raw_answer = full_text.split("Answer:")[-1].strip()
    else:
        raw_answer = full_text.replace(prompt, "").strip()
    
    answer = _postprocess(raw_answer)
    
    verification, sim, ref = _verify_with_memory(answer, question)
    
    return {
        "answer": answer if answer else "I couldn't generate a proper answer.",
        "verification": f"{verification} (similarity={sim:.2f})",
        "confidence": round(sim, 2),
        "method": "Fine-tuned",
        "response_time": elapsed,
        "source_docs": None,
        "source": ref if ref else "N/A"
    }




# import time
# from pathlib import Path
# import json
# import torch
# import numpy as np
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from sentence_transformers import SentenceTransformer, util
# from sklearn.metrics.pairwise import cosine_similarity

# # ---------- Paths / defaults ----------
# script_dir = Path(__file__).resolve().parent
# DEFAULT_MODEL_DIR = script_dir.parent / "models" / "financial_phi2_v1"
# QA_JSON = script_dir.parent / "data" / "qa_pair.json"
# QA_TXT = script_dir.parent / "data" / "qa_gpt2.txt"

# # ---------- Load FT model ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = GPT2Tokenizer.from_pretrained(DEFAULT_MODEL_DIR)
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained(DEFAULT_MODEL_DIR).to(device)
# model.eval()

# # ---------- Load memory Q/A & embeddings for verification ----------
# def _load_qa_pairs():
#     pairs = []
#     if QA_JSON.exists():
#         with open(QA_JSON, "r", encoding="utf-8") as f:
#             pairs = json.load(f)
#     elif QA_TXT.exists():
#         with open(QA_TXT, "r", encoding="utf-8") as f:
#             lines = f.read().split("\n")
#             q, a = None, None
#             for line in lines:
#                 if line.startswith("Q:"):
#                     q = line[2:].strip()
#                 elif line.startswith("A:"):
#                     a = line[2:].strip()
#                 elif line.strip() == "" and q and a:
#                     pairs.append({"question": q, "answer": a})
#                     q, a = None, None
#             if q and a:
#                 pairs.append({"question": q, "answer": a})
#     return pairs

# qa_pairs = _load_qa_pairs()
# embedder = SentenceTransformer("all-MiniLM-L6-v2", device=("cuda" if torch.cuda.is_available() else "cpu"))

# # Precompute embeddings for all answers
# answer_corpus = [p["answer"] for p in qa_pairs] if qa_pairs else []
# answer_embeddings = embedder.encode(answer_corpus) if answer_corpus else None

# # ---------- Simple input guardrail ----------
# FINANCE_KEYWORDS = {
#     "revenue", "net sales", "income", "operating income", "cash flow",
#     "balance sheet", "income statement", "assets", "liabilities", "eps",
#     "earnings per share", "dividends", "gross margin", "operating margin",
#     "guidance", "10-k", "10q", "financial", "quarter", "fiscal", "segment",
#     "geographic", "capex", "r&d", "share repurchase", "buyback", "opex",
#     "cost of sales", "goodwill", "services", "iphone", "mac", "ipad",
#     "spend", "expenditure", "expense", "cost", "marketing", "advertising",
#     "profit", "loss", "amortization", "depreciation", "valuation",
#     "debt", "equity", "stock", "share", "Apple", "Microsoft", "Google",
#     "investment", "portfolio", "risk", "return", "diversification"
# }

# def validate_query_ft(query: str) -> str:
#     q = (query or "").lower()
#     if any(bad in q for bad in ["attack", "explosive", "malware", "self-harm"]):
#         return "HARMFUL"
#     if not any(k in q for k in FINANCE_KEYWORDS):
#         return "IRRELEVANT"
#     return "OK"

# # ---------- FT answerer ----------
# DECODE = {
#     "num_beams": 5,
#     "max_new_tokens": 80,
#     "early_stopping": True,
#     "no_repeat_ngram_size": 2,
#     "repetition_penalty": 1.1,
#     "temperature": 0.3  # Added for better control
# }

# def _postprocess(text: str) -> str:
#     sents = [s.strip() for s in text.split(".") if s.strip()]
#     uniq = []
#     for s in sents:
#         if s not in uniq:
#             uniq.append(s)
#     out = ". ".join(uniq)
#     return out + ("." if out and not out.endswith(".") else "")

# def _verify_with_memory(generated: str, question: str = None):
#     """Verify generated answer against training data."""
#     if not generated.strip() or not answer_embeddings.any():
#         return "NOT_VERIFIED", 0.0, "Invalid input"
    
#     try:
#         # Generate embedding for the model's answer
#         gen_embedding = embedder.encode([generated])[0]
        
#         best_sim = -1
#         best_ref = None
#         best_question = None
        
#         # Compare with all training answers
#         similarities = util.cos_sim(gen_embedding, answer_embeddings)[0]
        
#         for i, sim in enumerate(similarities):
#             if sim > best_sim:
#                 best_sim = sim
#                 best_ref = qa_pairs[i]["answer"]
#                 best_question = qa_pairs[i]["question"]
        
#         # Also check if this is a direct match to training question
#         if question and any(q["question"].lower() == question.lower() for q in qa_pairs):
#             return "VERIFIED", 1.0, "Exact training question match"
        
#         if best_sim > 0.85:
#             return "VERIFIED", float(best_sim), best_ref
#         elif best_sim > 0.7:
#             return "PARTIALLY_VERIFIED", float(best_sim), best_ref
#         else:
#             return "NOT_VERIFIED", float(best_sim), best_ref
            
#     except Exception as e:
#         return "NOT_VERIFIED", 0.0, f"Error: {str(e)}"

# def get_ft_response(question: str):
#     """Uniform response shape with RAG for the UI."""
#     guard = validate_query_ft(question)
#     if guard in ("IRRELEVANT", "HARMFUL"):
#         return {
#             "answer": f"Your query was flagged as **{guard}**. Please ask a relevant financial question.",
#             "verification": "BLOCKED",
#             "confidence": 0.0,
#             "method": "Fine-tuned",
#             "response_time": 0.0
#         }
    
#     # Check if this is an exact training question
#     exact_match = None
#     for qa in qa_pairs:
#         if qa["question"].lower() == question.lower():
#             exact_match = qa["answer"]
#             break
    
#     if exact_match:
#         return {
#             "answer": exact_match,
#             "verification": "VERIFIED (exact match)",
#             "confidence": 1.0,
#             "method": "Fine-tuned",
#             "response_time": 0.1,
#             "source": "Training data"
#         }
    
#     # Use the SAME prompt format as training
#     prompt = f"Question: {question} Answer:"
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     start = time.time()
#     with torch.no_grad():
#         output_ids = model.generate(
#             **inputs,
#             num_beams=DECODE["num_beams"],
#             max_new_tokens=DECODE["max_new_tokens"],
#             early_stopping=DECODE["early_stopping"],
#             no_repeat_ngram_size=DECODE["no_repeat_ngram_size"],
#             repetition_penalty=DECODE["repetition_penalty"],
#             temperature=DECODE["temperature"],
#             pad_token_id=tokenizer.eos_token_id
#         )
#     elapsed = round(time.time() - start, 2)
    
#     full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
#     # Extract answer properly
#     if "Answer:" in full_text:
#         raw_answer = full_text.split("Answer:")[-1].strip()
#     else:
#         raw_answer = full_text.replace(prompt, "").strip()
    
#     answer = _postprocess(raw_answer)
    
#     verification, sim, ref = _verify_with_memory(answer, question)
    
#     return {
#         "answer": answer if answer else "I couldn't generate a proper answer.",
#         "verification": f"{verification} (similarity={sim:.2f})",
#         "confidence": round(sim, 2),
#         "method": "Fine-tuned",
#         "response_time": elapsed,
#         "source_docs": None,
#         "source": ref if ref else "N/A"
#     }
