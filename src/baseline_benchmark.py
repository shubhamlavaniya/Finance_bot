#this file is part of the Financial RAG Chatbot project.
# It implements a baseline benchmark for the phi-2 model on a Q/A dataset.


#================================================================#
# Note: This script assumes you have a text file with Q/A pairs in the format:
# Q: <question>
# A: <answer>
# Each pair is separated by a blank line.
# The output will be saved in a CSV file with columns for question, true answer, generated answer, and inference time.
# You can adjust the max_length and other parameters in the model.generate() call as needed.
# Make sure to install the required libraries: transformers, torch, pandas.
# You can run this script directly to benchmark the phi-2 model on your Q/A dataset.
# ===============================================================#

import json
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from pathlib import Path

# ====== CONFIG ======
script_dir = Path(__file__).resolve().parent

QA_JSON_PATH = script_dir.parent / "data" / "qa_pair.json"    
OUTPUT_CSV_PATH = script_dir.parent / "baseline_results.csv"
MODEL_NAME = "microsoft/phi-2"  # Changed to phi-2
MAX_LENGTH = 150  # max tokens to generate
NUM_SAMPLES = 10  # number of Qs to test
# ====================

# Load Q/A dataset
with open(QA_JSON_PATH, "r") as f:
    qa_data = json.load(f)

# Take only first NUM_SAMPLES
qa_samples = qa_data[:NUM_SAMPLES]

# Load model & tokenizer for phi-2
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token for phi-2

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use float16 to save memory
    device_map="auto"  # Automatically use GPU if available
)
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
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True  # Added for better generation
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

# import json
# import time
# import pandas as pd
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import os
# from pathlib import Path

# # ====== CONFIG ======
# script_dir = Path(__file__).resolve().parent

# QA_JSON_PATH = script_dir.parent / "data" / "qa_pair.json"    
# OUTPUT_CSV_PATH = script_dir.parent / "baseline_results.csv"
# MODEL_NAME = "distilgpt2" #or give gpt2
# MAX_LENGTH = 150  # max tokens to generate
# NUM_SAMPLES = 10  # number of Qs to test
# # ====================

# # Load Q/A dataset
# with open(QA_JSON_PATH, "r") as f:
#     qa_data = json.load(f)

# # Take only first NUM_SAMPLES
# qa_samples = qa_data[:NUM_SAMPLES]

# # Load model & tokenizer
# print(f"Loading model: {MODEL_NAME}")
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
# model.eval()

# results = []

# # Run baseline inference
# for sample in qa_samples:
#     question = sample["question"]
#     true_answer = sample["answer"]

#     input_text = f"Q: {question}\nA:"
#     inputs = tokenizer.encode(input_text, return_tensors="pt")

#     start_time = time.time()
#     outputs = model.generate(
#         inputs,
#         max_length=len(inputs[0]) + MAX_LENGTH,
#         temperature=0.7,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     end_time = time.time()

#     generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     generated_answer = generated_answer.replace(input_text, "").strip()

#     results.append({
#         "question": question,
#         "true_answer": true_answer,
#         "generated_answer": generated_answer,
#         "inference_time_sec": round(end_time - start_time, 3)
#     })

# # Save results to CSV
# df = pd.DataFrame(results)
# df.to_csv(OUTPUT_CSV_PATH, index=False)
# print(f"Baseline results saved to {OUTPUT_CSV_PATH}")



