#this file is part of the Financial RAG Chatbot project.
# It implements a baseline benchmark for the GPT-2 model on a Q/A dataset.


#================================================================#
# Note: This script assumes you have a text file with Q/A pairs in the format:
# Q: <question>
# A: <answer>
# Each pair is separated by a blank line.
# The output will be saved in a CSV file with columns for question, true answer, generated answer, and inference time.
# You can adjust the max_length and other parameters in the model.generate() call as needed.
# Make sure to install the required libraries: transformers, torch, pandas.
# You can run this script directly to benchmark the GPT-2 model on your Q/A dataset.
# ===============================================================#

import json
import time
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os
from pathlib import Path

# ====== CONFIG ======
script_dir = Path(__file__).resolve().parent

QA_JSON_PATH = script_dir.parent / "data" / "qa_pair.json"    
OUTPUT_CSV_PATH = script_dir.parent / "baseline_results.csv"
MODEL_NAME = "distilgpt2" #or give gpt2
MAX_LENGTH = 150  # max tokens to generate
NUM_SAMPLES = 10  # number of Qs to test
# ====================

# Load Q/A dataset
with open(QA_JSON_PATH, "r") as f:
    qa_data = json.load(f)

# Take only first NUM_SAMPLES
qa_samples = qa_data[:NUM_SAMPLES]

# Load model & tokenizer
print(f"Loading model: {MODEL_NAME}")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.eval()

results = []

# Run baseline inference
for sample in qa_samples:
    question = sample["question"]
    true_answer = sample["answer"]

    input_text = f"Q: {question}\nA:"
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    start_time = time.time()
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + MAX_LENGTH,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    end_time = time.time()

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = generated_answer.replace(input_text, "").strip()

    results.append({
        "question": question,
        "true_answer": true_answer,
        "generated_answer": generated_answer,
        "inference_time_sec": round(end_time - start_time, 3)
    })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"Baseline results saved to {OUTPUT_CSV_PATH}")


# Old version-----------------------------------------------------

# import time
# import json
# import csv
# from pathlib import Path
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # --- Paths ---
# script_dir = Path(__file__).resolve().parent
# model_path = script_dir.parent / "models" / "distilgpt2-finetuned"
# qa_file = script_dir.parent / "data" / "qa_gpt2.txt"  # text file in Q/A format
# output_csv = script_dir.parent / "distilgpt2_benchmark_results.csv"

# # --- Load fine-tuned model & tokenizer ---
# tokenizer = GPT2Tokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained(model_path)
# model.eval()

# # --- Read Q/A pairs from text file ---
# qa_pairs = []
# with open(qa_file, "r", encoding="utf-8") as f:
#     lines = f.read().split("\n")
#     question, answer = None, None
#     for line in lines:
#         if line.startswith("Q:"):
#             question = line[2:].strip()
#         elif line.startswith("A:"):
#             answer = line[2:].strip()
#         elif line.strip() == "" and question and answer:
#             qa_pairs.append({"question": question, "answer": answer})
#             question, answer = None, None
#     if question and answer:  # add last pair
#         qa_pairs.append({"question": question, "answer": answer})

# # --- Evaluate model ---
# results = []
# for pair in qa_pairs:
#     prompt = f"Q: {pair['question']}\nA:"
#     inputs = tokenizer(prompt, return_tensors="pt")

#     start_time = time.time()
#     output_ids = model.generate(
#         **inputs,
#         max_length=inputs.input_ids.shape[1] + 50,
#         num_beams=10,
#         max_new_tokens=100,
#         early_stopping=True,
#         #temperature=0.0
#     )
#     end_time = time.time()
#     inference_time = round(end_time - start_time, 2)

#     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     generated_answer = generated_text.split("A:")[-1].strip()

#     results.append({
#         "question": pair["question"],
#         "true_answer": pair["answer"],
#         "generated_answer": generated_answer,
#         "inference_time_sec": inference_time
#     })

# # --- Save results to CSV ---
# output_csv.parent.mkdir(exist_ok=True)
# with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
#     fieldnames = ["question", "true_answer", "generated_answer", "inference_time_sec"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for r in results:
#         writer.writerow(r)

# print(f"Benchmark completed. Results saved to {output_csv}")