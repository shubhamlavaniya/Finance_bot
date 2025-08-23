
# fine_tune_distil_gpt2.py it is a script to fine-tune a DistilGPT-2 model on a custom dataset of question-answer pairs.
# It uses the Hugging Face Transformers library for model and tokenizer handling, and PyTorch for dataset management.
# This is a reference script that can be used to fine-tune a DistilGPT-2 model on a custom dataset of question-answer pairs.




#from transformers import DistilGPT2Tokenizer, DistilGPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
from torch.utils.data import Dataset


# --- Paths ---
script_dir = Path(__file__).resolve().parent
input_file = script_dir.parent / "data" / "qa_gpt2.txt"
output_dir = script_dir.parent / "models" / "distilgpt2-finetuned"

# --- Load tokenizer & model ---
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# --- Read dataset ---
with open(input_file, "r") as f:
    lines = f.readlines()

# Join lines and split into samples by double newlines
samples = [line.strip() for line in "".join(lines).split("\n\n") if line.strip()]

# Prepare dataset as list of dicts
train_texts = [{"text": sample} for sample in samples]

# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)


class QADataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = [self.tokenizer(d["text"], truncation=True, padding="max_length", max_length=256) for d in data]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val) for key, val in self.inputs[idx].items()}
        return item


train_dataset = QADataset(train_texts, tokenizer)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=2,
    save_steps=200,
    save_total_limit=1,
    logging_dir=script_dir.parent / "logs",
    logging_steps=10
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Fine-tune
trainer.train()

# Save model & tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)



# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from pathlib import Path

# # --- Paths ---
# script_dir = Path(__file__).resolve().parent
# input_file = script_dir.parent / "data" / "qa_gpt2.txt"
# output_file = script_dir.parent / "models" / "gpt2_finetuned_model"
# logging_dir = script_dir.parent / "logs"

# # --- Load model & tokenizer ---
# model_name = "gpt2"
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # --- Read text file ---
# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # Remove empty lines and join all lines (optional: you can keep line by line too)
# texts = [line.strip() for line in lines if line.strip()]

# # --- Tokenize ---
# tokenized_texts = tokenizer(texts, truncation=True, padding="max_length", max_length=256, return_tensors="pt")

# # --- Dataset wrapper ---
# import torch
# from torch.utils.data import Dataset

# class TextDataset(Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings
#     def __len__(self):
#         return len(self.encodings["input_ids"])
#     def __getitem__(self, idx):
#         return {key: val[idx] for key, val in self.encodings.items()}

# train_dataset = TextDataset(tokenized_texts)

# # --- Data collator ---
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# # --- Training arguments ---
# training_args = TrainingArguments(
#     output_dir=output_file,
#     overwrite_output_dir=True,
#     num_train_epochs=10,
#     per_device_train_batch_size=2,
#     save_steps=200,
#     save_total_limit=1,
#     logging_dir=logging_dir,
#     logging_steps=10
# )

# # --- Trainer ---
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
#     data_collator=data_collator
# )

# # --- Fine-tune ---
# trainer.train()

# # --- Save model ---
# trainer.save_model(output_file)
# tokenizer.save_pretrained(output_file)
# print("Fine-tuning complete. Model saved to ../models/gpt2-finetuned")
# print("Tokenizer saved to ../models/gpt2-finetuned")
# print("You can now use this model for inference or further training.")