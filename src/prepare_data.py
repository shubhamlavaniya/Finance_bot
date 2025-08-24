# This file is part of the Financial RAG Chatbot project.
# prepare dataset for fine-tuning
# It converts the Q/A pairs into a text format suitable for GPT-2 training.

import json
from pathlib import Path
import os

script_dir = Path(__file__).resolve().parent

input_file = script_dir.parent / "data" / "qa_pair.json"
output_file = script_dir.parent / "data" / "qa_gpt2.txt"

with open(input_file, "r") as f:
    data = json.load(f)

with open(output_file, "w") as f:
    for entry in data:
        q = entry["question"].strip()
        a = entry["answer"].strip()
        f.write(f"Q: {q}\nA: {a}\n\n")

print(f"Dataset saved to {output_file}")
