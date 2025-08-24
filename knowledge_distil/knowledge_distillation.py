# This script is for knowledge distiliation 
# Teacher-Student Architecture: fine-tuned Phi-2 teaches a smaller Phi-3-mini model

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import json
from pathlib import Path

# ====== CONFIG ======
script_dir = Path(__file__).resolve().parent
TEACHER_MODEL_PATH = script_dir.parent / "models" / "financial_phi2_v1"
STUDENT_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"  # Smaller student model
QA_JSON_PATH = script_dir.parent / "data" / "qa_pair.json"
DISTILLED_MODEL_PATH = script_dir.parent / "distilled_model"
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
# ====================

# Load teacher model (your fine-tuned Phi-2)
print("Loading teacher model...")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_PATH, trust_remote_code=True)
teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
teacher_model.eval()

# Load student model
print("Loading student model...")
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, trust_remote_code=True)
student_tokenizer.pad_token = student_tokenizer.eos_token

student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load QA data
print("Loading QA data...")
with open(QA_JSON_PATH, "r") as f:
    qa_data = json.load(f)

# Prepare dataset
def prepare_distillation_data(samples):
    formatted_texts = []
    for item in samples:
        text = f"Q: {item['question']}\nA: {item['answer']}"
        formatted_texts.append(text)
    return formatted_texts

distillation_texts = prepare_distillation_data(qa_data)
dataset = Dataset.from_dict({"text": distillation_texts})

# Tokenize function
def tokenize_function(examples):
    # Tokenize with teacher tokenizer (since teacher was trained with this)
    teacher_encodings = teacher_tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    # Get teacher logits (no gradients to save memory)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=teacher_encodings["input_ids"],
            attention_mask=teacher_encodings["attention_mask"]
        )
        teacher_logits = teacher_outputs.logits
    
    # Tokenize with student tokenizer
    student_encodings = student_tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    
    return {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "teacher_logits": teacher_logits,
        "labels": student_encodings["input_ids"].clone()  # For language modeling loss
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=BATCH_SIZE)

# Knowledge distillation loss
class DistillationLoss:
    def __init__(self, temperature=2.0, alpha=0.5):
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation vs student loss
        self.lm_loss = nn.CrossEntropyLoss()
    
    def __call__(self, student_logits, teacher_logits, labels):
        # Knowledge distillation loss
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Standard language modeling loss
        lm_loss = self.lm_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * lm_loss

# Custom trainer for distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss = DistillationLoss()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        teacher_logits = inputs.pop("teacher_logits")
        
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        loss = self.distillation_loss(student_logits, teacher_logits, labels)
        
        return (loss, outputs) if return_outputs else loss

# Training arguments
training_args = TrainingArguments(
    output_dir=str(DISTILLED_MODEL_PATH),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="no",
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # Use mixed precision
    dataloader_pin_memory=False,
)

# Create trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=student_tokenizer,
)

# Train
print("Starting knowledge distillation...")
trainer.train()

# Save distilled model
print("Saving distilled model...")
trainer.save_model()
student_tokenizer.save_pretrained(DISTILLED_MODEL_PATH)

print(f"Distilled model saved to {DISTILLED_MODEL_PATH}")