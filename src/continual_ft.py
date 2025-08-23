import os
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# --- Auto classes for general model loading ---
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import AdamW, Adam, SGD
from datetime import datetime
from pathlib import Path

# --- Import PEFT classes for QLoRA ---
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig

# -----------------------------
# Custom Dataset (no changes needed here)
# -----------------------------
script_dir = Path(__file__).resolve().parent

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=256):
        self.encodings = []
        for qa in qa_pairs:
            text = f"Q: {qa['question']} A: {qa['answer']}"
            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": enc["input_ids"].squeeze()
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]

# -----------------------------
# Load config
# -----------------------------
with open("/config/training_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

save_dir = Path(cfg["save_dir"])

# -----------------------------
# Tokenizer + Model
# -----------------------------
# --- CHANGE 3: Load tokenizer with AutoTokenizer ---
tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- CHANGE 4: Use QLoRA and PEFT for efficient fine-tuning ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if save_dir.exists():
    print(f"Running continual training from {save_dir} ...")
    # Load the base model and then the PEFT adapter
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Load PEFT model from the saved directory
    model = get_peft_model(base_model, PeftConfig.from_pretrained(save_dir))
    mode = "continual"
else:
    print("Running initial fine-tuning...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Prepare base model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Phi-2 target modules
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # --- REVISED CODE FOR GPT-2 ---
    #lora_config = LoraConfig(
        #r=16,
        #lora_alpha=32,
        #lora_dropout=0.05,
        #bias="none",
        #task_type="CAUSAL_LM",
    # GPT-2 specific target modules
      #target_modules=["c_attn", "c_proj"],
    #)
    model = get_peft_model(base_model, lora_config)
    mode = "initial"

# --- Remove redundant model.to(device) because device_map handles it ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

print(f"Number of trainable parameters: {model.print_trainable_parameters()}")


# -----------------------------
# Load dataset
# -----------------------------
with open(cfg["train_file"], "r") as f:
    qa_pairs = json.load(f)

dataset = QADataset(qa_pairs, tokenizer, max_length=cfg.get("max_length", 256))

# Train/val split
val_ratio = cfg.get("val_split", 0.1)
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"]) if val_size > 0 else None

# -----------------------------
# Optimizer (no changes needed here, it's generic)
# -----------------------------
opt_choice = cfg.get("optimizer", "AdamW")
if opt_choice == "Adam":
    optimizer = Adam(model.parameters(), lr=float(cfg["learning_rate"]))
elif opt_choice == "SGD":
    optimizer = SGD(model.parameters(), lr=float(cfg["learning_rate"]), momentum=0.9)
else:
    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=cfg.get("weight_decay", 0.01))

# -----------------------------
# Training Loop
# -----------------------------
model.train()
epoch_losses = []
early_stop_counter = 0
best_val_loss = float("inf")
patience = cfg.get("early_stopping_patience", None)

for epoch in range(cfg["epochs"]):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        # --- CHANGE 6: Move input tensors to the device ---
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{cfg['epochs']} - Loss: {avg_loss:.4f} - Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f}")

    # Validation
    if val_loader:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)
        print(f"  >> Validation Loss: {val_loss:.4f} - Val Perplexity: {torch.exp(torch.tensor(val_loss)):.2f}")

        # Early stopping
        if patience:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered!")
                    break
        model.train()

# -----------------------------
# Save model + tokenizer
# -----------------------------
os.makedirs(save_dir, exist_ok=True)
# --- CHANGE 7: Save only the PEFT adapter weights, not the whole model ---
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Training completed. PEFT adapter saved at {save_dir}")

# -----------------------------
# Logging metadata
# -----------------------------
os.makedirs("logs", exist_ok=True)
# Adjusted log file path to be relative to the current directory
log_file = Path("logs") / "continual_training.json"

log_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "mode": mode,
    "train_file": cfg["train_file"],
    "save_dir": str(save_dir),
    "epochs": cfg["epochs"],
    "batch_size": cfg["batch_size"],
    "learning_rate": cfg["learning_rate"],
    "optimizer": opt_choice,
    # --- CHANGE 8: Use the model's device
    "device": str(model.device),
    "epoch_losses": epoch_losses,
    "final_loss": epoch_losses[-1] if epoch_losses else None
}

if os.path.exists(log_file):
    with open(log_file, "r") as f:
        logs = json.load(f)
else:
    logs = []

logs.append(log_entry)

with open(log_file, "w") as f:
    json.dump(logs, f, indent=4)

print(f"Training metadata logged in {log_file}")