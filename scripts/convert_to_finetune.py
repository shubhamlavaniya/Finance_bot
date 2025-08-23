import json

# Paths
input_file = "data/qa_pair.json"  # your existing file
output_file = "data/finetune_dataset.jsonl"  # model-ready file

with open(input_file, "r") as f:
    qa_pairs = json.load(f)

with open(output_file, "w") as out_f:
    for pair in qa_pairs:
        prompt = f"Question: {pair['question']}\nAnswer:"
        completion = f" {pair['answer']}"  # space at start is common in HF training
        json.dump({"prompt": prompt, "completion": completion}, out_f)
        out_f.write("\n")

print(f"Converted {len(qa_pairs)} Q/A pairs to fine-tune format at {output_file}")
