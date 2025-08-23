# This script is for testing the new finetuned model for performance benchmark against the original distilgpt2 model.

import time
import json
import csv
import re
from pathlib import Path
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util

# --- Paths ---
script_dir = Path(__file__).resolve().parent
model_path = script_dir.parent / "models" / "financial_gpt2_v1"
qa_file = script_dir.parent / "data" / "qa_gpt2.txt"
output_csv = script_dir.parent / "distilgpt2_benchmark_results.csv"
log_file = script_dir.parent / "logs" / "distilgpt2_benchmark_log.json"

# --- Eval config / thresholds ---
SIM_THRESHOLD = 0.85  # counts as "correct" if similarity >= 0.85

# --- Load model & tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_str = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
model.eval()

# --- Load embedding model for semantic similarity ---
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device_str)

# --- Read Q/A pairs from the text file ---
qa_pairs = []
with open(qa_file, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    question, answer = None, None
    for line in lines:
        if line.startswith("Q:"):
            question = line[2:].strip()
        elif line.startswith("A:"):
            answer = line[2:].strip()
        elif line.strip() == "" and question and answer:
            qa_pairs.append({"question": question, "answer": answer})
            question, answer = None, None
    if question and answer:  # catch last block if no trailing blank line
        qa_pairs.append({"question": question, "answer": answer})

# --- Evaluate model ---
results = []
total_time = 0.0
sims = []
num_correct = 0

for pair in qa_pairs:
    #prompt = f"Question: {pair['question']} Answer:"
    prompt = f"Q: {pair['question']} A:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    output_ids = model.generate(
        **inputs,
        max_length=inputs.input_ids.shape[1] + 50,
        num_beams=3,
        max_new_tokens=100,
        early_stopping=True,
        #temperature=0.7,
        #do_sample=True,
        #top_p=0.9,
        #no_repeat_ngram_size=3,   # reduce repetition
        #repetition_penalty=1.2    # discourage loops
    )
    end_time = time.time()
    inference_time = round(end_time - start_time, 2)
    total_time += inference_time

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #generated_answer = generated_text.split("A:")[-1].strip()
    generated_answer = re.sub(r"^Q:.*?A:\s*", "", generated_text, flags=re.DOTALL).strip()


    # --- Post-process to remove repeated sentences ---
    sentences = [s.strip() for s in generated_answer.split(".") if s.strip()]
    unique = []
    for s in sentences:
        if s not in unique:
            unique.append(s)
    generated_answer = ". ".join(unique)
    if generated_answer and not generated_answer.endswith("."):
        generated_answer += "."

    # --- Semantic similarity ---
    emb_true = embedder.encode(pair["answer"], convert_to_tensor=True, normalize_embeddings=True)
    emb_pred = embedder.encode(generated_answer, convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb_true, emb_pred).item())
    sims.append(sim)

    if sim >= SIM_THRESHOLD:
        num_correct += 1

    results.append({
        "question": pair["question"],
        "true_answer": pair["answer"],
        "generated_answer": generated_answer,
        "similarity_score": round(sim, 3),
        "inference_time_sec": inference_time
    })

# --- Aggregate metrics ---
avg_time = round(total_time / max(len(qa_pairs), 1), 2)
avg_sim = round(sum(sims) / max(len(sims), 1), 3)
accuracy = round(100.0 * num_correct / max(len(qa_pairs), 1), 2)

# --- Save results to CSV ---
output_csv.parent.mkdir(exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["question", "true_answer", "generated_answer", "similarity_score", "inference_time_sec"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow(r)

# --- Save run metadata ---
metadata = {
    "model_path": str(model_path),
    "device": device_str,
    "decoding_params": {
        "num_beams": 5,
        "max_new_tokens": 100,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "early_stopping": True
    },
    "evaluation": {
        "num_questions": len(qa_pairs),
        "similarity_model": "all-MiniLM-L6-v2",
        "similarity_threshold": SIM_THRESHOLD,
        "avg_similarity": avg_sim,
        "accuracy_at_threshold_percent": accuracy,
        "total_inference_time_sec": round(total_time, 2),
        "avg_inference_time_sec": avg_time
    }
}
with open(log_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Benchmark completed. Results saved to {output_csv}")
print(f"Metadata log saved to {log_file}")
print(f"Accuracy @ {SIM_THRESHOLD}: {accuracy}% | Avg similarity: {avg_sim} | Avg time: {avg_time}s")



# --- End of Baseline Benchmark Script ---