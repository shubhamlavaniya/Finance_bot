# This script is for evaluation of RAG and Fine-Tuned models
# using LangChain with a hybrid retriever setup.    
# using a fine-tuned GPT-2 model for financial question answering.

import json
import time
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

# --- NEW: Imports for RAG (Hugging Face Hub) and Fine-tuned (PEFT) ---
import os
import torch
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------
# Paths
# -----------------------------
script_dir = Path(__file__).resolve().parent

QUESTIONS_FILE = script_dir / "questions.json"
CHUNKS_FILE = script_dir.parent / "data" / "processed" / "chunks.csv"
OUTPUT_FILE = script_dir / "eval_results.csv"

# --- NEW: Path to the fine-tuned PEFT adapter ---
FT_MODEL_DIR = script_dir.parent / "models" / "financial_phi2_v1"

# -----------------------------
# Evaluation Configuration
# -----------------------------
# Threshold for semantic correctness (0.0 to 1.0)
SEMANTIC_THRESHOLD = 0.80

# -----------------------------
# Load Data & Models
# -----------------------------
# You must set your Hugging Face API token as an environment variable
# export HUGGINGFACEHUB_API_TOKEN="hf_..."
# Or set it here for testing
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

try:
    with open(QUESTIONS_FILE, "r") as f:
        questions = json.load(f)
except FileNotFoundError:
    print(f"Questions file not found at {QUESTIONS_FILE}. Please create it.")
    exit()

try:
    chunks_df = pd.read_csv(CHUNKS_FILE)
except FileNotFoundError:
    print(f"Chunks file not found at {CHUNKS_FILE}. Please run data processing first.")
    exit()

# -----------------------------
# Evaluation & Embedding Model
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
chunk_embeddings = embed_model.encode(chunks_df["text"].tolist(), convert_to_tensor=True)

def evaluate_correctness(answer: str, ground_truth: str) -> str:
    """Calculates semantic similarity and returns 'Y' or 'N'."""
    if not ground_truth or not answer:
        return "N"
    answer_emb = embed_model.encode(answer, convert_to_tensor=True)
    gt_emb = embed_model.encode(ground_truth, convert_to_tensor=True)
    similarity = util.cos_sim(answer_emb, gt_emb).item()
    return "Y" if similarity >= SEMANTIC_THRESHOLD else "N"

def get_semantic_score(answer: str, ground_truth: str) -> float:
    """Calculates the raw semantic similarity score."""
    if not ground_truth or not answer:
        return 0.0
    answer_emb = embed_model.encode(answer, convert_to_tensor=True)
    gt_emb = embed_model.encode(ground_truth, convert_to_tensor=True)
    return util.cos_sim(answer_emb, gt_emb).item()

# -----------------------------
# RAG (Hugging Face API)
# -----------------------------
print("Loading RAG LLM...")
llm_rag = HuggingFaceHub(
    repo_id="microsoft/phi-2",
    model_kwargs={"temperature": 0.1, "max_new_tokens": 512}
)

def query_rag(question: str):
    start_time = time.time()

    # --- Use Sentence-Transformers for semantic search to find context ---
    q_emb = embed_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, chunk_embeddings)[0]
    top_scores, top_indices = torch.topk(scores, k=3)

    context = ""
    for i, idx in enumerate(top_indices):
        chunk_text = chunks_df.iloc[idx.item()]["text"]
        context += f"--- Source {i+1} ---\n{chunk_text}\n\n"

    # --- Craft the prompt for the LLM API call ---
    prompt = f"Based on the following context, answer the question. If the information is not present, say so.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
    try:
        response = llm_rag.invoke(prompt)
        answer = response
    except Exception as e:
        answer = f"Error during RAG API call: {e}"

    elapsed = round(time.time() - start_time, 2)
    confidence = float(top_scores[0].item())

    return answer, confidence, elapsed

# -----------------------------
# Fine-Tuned Phi-2 Model (Local)
# -----------------------------
print("Loading fine-tuned Phi-2 model...")
ft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW: Load the base model first, then the PEFT adapter ---
base_model_name = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

try:
    ft_model = PeftModel.from_pretrained(base_model, FT_MODEL_DIR)
    ft_model = ft_model.merge_and_unload() # Merge weights for inference
    ft_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    ft_tokenizer.pad_token = ft_tokenizer.eos_token
    ft_model.to(ft_device)
    ft_model.eval()
    print("Fine-tuned model loaded successfully.")
except Exception as e:
    print(f"Failed to load fine-tuned model: {e}")
    exit()


def query_finetuned(question: str):
    start_time = time.time()

    prompt_text = f"Q: {question} A:"
    input_ids = ft_tokenizer.encode(prompt_text, return_tensors="pt").to(ft_device)

    with torch.no_grad():
        output = ft_model.generate(
            input_ids,
            max_new_tokens=80,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=ft_tokenizer.eos_token_id
        )

    generated_text = ft_tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.replace(prompt_text, "").strip()

    elapsed = round(time.time() - start_time, 2)
    return answer, None, elapsed

# -----------------------------
# Evaluation Loop
# -----------------------------
results = []
for q in questions:
    q_text = q["question"]
    ground_truth = q.get("answer", "").strip()

    # --- RAG ---
    rag_ans, rag_conf, rag_time = query_rag(q_text)
    rag_semantic_score = get_semantic_score(rag_ans, ground_truth)
    rag_correct = evaluate_correctness(rag_ans, ground_truth)

    results.append({
        "question": q_text,
        "method": "RAG (Phi-2 API)",
        "ground_truth": ground_truth,
        "answer": rag_ans,
        "confidence": round(rag_conf, 2),
        "time_s": rag_time,
        "semantic_score": round(rag_semantic_score, 2),
        "correct": rag_correct
    })

    # --- Fine-Tuned ---
    ft_ans, _, ft_time = query_finetuned(q_text)
    ft_semantic_score = get_semantic_score(ft_ans, ground_truth)
    ft_correct = evaluate_correctness(ft_ans, ground_truth)

    results.append({
        "question": q_text,
        "method": "Fine-Tuned (Local Phi-2)",
        "ground_truth": ground_truth,
        "answer": ft_ans,
        "confidence": "N/A",
        "time_s": ft_time,
        "semantic_score": round(ft_semantic_score, 2),
        "correct": ft_correct
    })

# -----------------------------
# Save Results
# -----------------------------
df = pd.DataFrame(results)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Evaluation results saved to {OUTPUT_FILE}")






#older v2-----------------------------------------------

# import json
# import time
# import re
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# import ollama  # Ollama must be running
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from pathlib import Path

# # -----------------------------
# # Paths
# # -----------------------------
# script_dir = Path(__file__).resolve().parent

# QUESTIONS_FILE = script_dir / "questions.json"
# CHUNKS_FILE = script_dir.parent / "data" / "processed" / "chunks.csv"
# OUTPUT_FILE = script_dir / "eval_results.csv"

# FT_MODEL_DIR = script_dir.parent / "models" / "financial_gpt2_v1"

# # -----------------------------
# # Load Data
# # -----------------------------
# with open(QUESTIONS_FILE, "r") as f:
#     questions = json.load(f)

# chunks_df = pd.read_csv(CHUNKS_FILE)

# # -----------------------------
# # Embedding Model (for retrieval)
# # -----------------------------
# embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# chunk_embeddings = embed_model.encode(chunks_df["text"].tolist(), convert_to_tensor=True)

# # -----------------------------
# # Fine-Tuned GPT2 Model
# # -----------------------------
# print("Loading fine-tuned GPT2 model...")
# ft_tokenizer = GPT2Tokenizer.from_pretrained(FT_MODEL_DIR)
# ft_model = GPT2LMHeadModel.from_pretrained(FT_MODEL_DIR)

# ft_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ft_model.to(ft_device)
# ft_model.eval()

# # -----------------------------
# # RAG (Gemma2b via Ollama)
# # -----------------------------
# def query_rag(question: str):
#     start_time = time.time()

#     q_emb = embed_model.encode(question, convert_to_tensor=True)
#     scores = util.cos_sim(q_emb, chunk_embeddings)[0]
#     top_scores, top_indices = torch.topk(scores, k=3)

# # Combine the top chunks into a single context

#     context = ""
#     for i, idx in enumerate(top_indices):
#         chunk_text = chunks_df.iloc[idx.item()]["text"]
#         context += f"--- Source {i+1} ---\n{chunk_text}\n\n"
    
#     # Modify the prompt to guide the LLM to use the provided context

#     prompt = f"Based on the following context, answer the question. If the information is not present, say so.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    
#     response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": prompt}])
    
#     answer = response["message"]["content"]
#     elapsed = round(time.time() - start_time, 2)
    
#     # The confidence should be based on the highest score from the retrieved chunks

#     confidence = float(top_scores[0].item())
    
#     return answer, confidence, elapsed

# # -----------------------------
# # Fine-Tuned GPT2
# # -----------------------------
# def query_finetuned(question: str):
#     start_time = time.time()

#     prompt_text = f"Q: {question}\nA:"

#     input_ids = ft_tokenizer.encode(prompt_text, return_tensors="pt").to(ft_device)

#     with torch.no_grad():
#         output = ft_model.generate(
#             input_ids,
#             max_new_tokens=80,
#             do_sample=True,
#             top_k=50,
#             top_p=0.95,
#             temperature=0.7,
#             pad_token_id=ft_tokenizer.eos_token_id
#         )

#     generated_text = ft_tokenizer.decode(output[0], skip_special_tokens=True)

#     # Remove the prompt from output → get only answer

#     answer = generated_text.replace(prompt_text, "").strip()

#     elapsed = round(time.time() - start_time, 2)
#     return answer, 0.85, elapsed  # Dummy confidence

# # -----------------------------
# # Evaluation Loop
# # -----------------------------
# results = []
# for q in questions:
#     q_text = q["question"]
#     ground_truth = q.get("answer", "").strip()

#     # --- RAG ---
#     rag_ans, rag_conf, rag_time = query_rag(q_text)
#     rag_correct = "Y" if ground_truth.lower() in rag_ans.lower() else "N"

#     results.append({
#         "question": q_text,
#         "method": "RAG (Gemma2b)",
#         "ground_truth": ground_truth,
#         "answer": rag_ans,
#         "confidence": round(rag_conf, 2),
#         "time_s": rag_time,
#         "correct": rag_correct
#     })

#     # --- Fine-Tuned ---
#     ft_ans, ft_conf, ft_time = query_finetuned(q_text)
#     ft_correct = "Y" if ground_truth.lower() in ft_ans.lower() else "N"

#     results.append({
#         "question": q_text,
#         "method": "Fine-Tuned (financial_gpt2_v1)",
#         "ground_truth": ground_truth,
#         "answer": ft_ans,
#         "confidence": round(ft_conf, 2),
#         "time_s": ft_time,
#         "correct": ft_correct
#     })

# # -----------------------------
# # Save Results
# # -----------------------------
# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_FILE, index=False)
# print(f"Evaluation results saved to {OUTPUT_FILE}")

