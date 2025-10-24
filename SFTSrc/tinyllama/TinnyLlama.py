# -*- coding: utf-8 -*-
"""
classification_finetune_tinyllama.py

This script fine-tunes the TinyLlama-1.1B model for a binary classification
task on preprocessed CAN bus data. It leverages PEFT with QLoRA for
memory-efficient training.

The process includes:
1.  Loading the Hugging Face API token from a .env file.
2.  Loading and preprocessing training data (Kia) and validation data (Tesla) from separate directories.
3.  Converting the numerical sequences into string format to be used as text input.
4.  Loading the TinyLlama base model with a new sequence classification head.
5.  Applying 4-bit quantization (QLoRA) to the base model.
6.  Configuring a LoRA adapter for parameter-efficient fine-tuning.
7.  Setting up and running the training process using the standard Hugging Face Trainer.
8.  Evaluating the model's classification performance on the validation set.
"""
import os
import glob
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import SegmentFromFile # Your provided preprocessing function

# --- 1. Load Environment Variables ---
# This will load the HF_TOKEN from your .env file.
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Warning: Hugging Face token not found. You may not be able to access private models.")
else:
    print("Hugging Face token loaded successfully.")


# --- 2. Configuration ---

# The base language model we will adapt for classification.
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Path to the directory containing your training data.
DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/Kia"
# Path to the directory containing your validation data.
VALIDATION_DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/Tesla"


# The name for the new, fine-tuned model adapter.
NEW_MODEL_NAME = "tinyllama-can-attack-classifier-adapter"

# Segmentation parameter from your utils file.
TIME_GAP_TRAIN = 100.0
TIME_GAP_TEST = 85.0

# --- 3. Load and Preprocess Data ---

def load_and_process_data(directory, time_gap):
    """Loads all CSVs from a directory, processes them, and returns a formatted DataFrame."""
    print(f"Loading data from: {directory}")
    all_chunks = []
    all_labels = []

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        # Note: SegmentFromFile expects the directory and filename separately
        chunks, labels = SegmentFromFile(directory, filename, time_gap=time_gap)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    print(f"Total segments processed from {directory}: {len(all_chunks)}")

    # Convert the numerical sequences into a string-based format for the LLM.
    def format_chunk_to_string(chunk):
        return " ".join([f"[{int(pair[0])} {int(pair[1])}]" for pair in chunk])

    formatted_texts = [format_chunk_to_string(chunk) for chunk in all_chunks]
    df = pd.DataFrame({'text': formatted_texts, 'label': all_labels})
    return df

# Load training and validation data from their respective directories
train_df = load_and_process_data(DATA_DIRECTORY, TIME_GAP_TRAIN)
test_df = load_and_process_data(VALIDATION_DATA_DIRECTORY, TIME_GAP_TEST)

num_labels = pd.concat([train_df['label'], test_df['label']]).nunique()
print(f"Data formatted. Total number of unique labels across datasets: {num_labels}")


# --- 4. Create Hugging Face Dataset ---

print("Creating Hugging Face dataset...")
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

# --- 5. Load Tokenizer and Tokenize Data ---

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    token=hf_token
)
# A pad token is required for classification batching.
# We use the EOS token as a pad token, a common practice for Llama models.
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False) # Padding handled by collator

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# The __index_level_0__ column is an artifact from pandas conversion, remove it.
if "__index_level_0__" in tokenized_datasets["train"].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
else:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Data collator will dynamically pad the batched sequences to the same length.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --- 6. Configure Quantization and PEFT ---

print("Configuring QLoRA and PEFT...")
# QLoRA configuration to load the model in 4-bit for memory efficiency.
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Specify the task type for classification
    r=2,                       # Rank of the update matrices
    lora_alpha=4,              # Alpha parameter for scaling
    lora_dropout=0.1,           # Dropout probability
    bias="none",
    target_modules=[            # Target the same modules as in the original script
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

# --- 7. Load Base Model ---

print(f"Loading base model: {MODEL_NAME}")
# We load the Causal LM but specify `num_labels`.
# `AutoModelForSequenceClassification` will automatically add a new, untrained
# classification head on top of the frozen Llama model.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
)

# The tokenizer and model need to agree on the padding token ID.
model.config.pad_token_id = tokenizer.pad_token_id

# Apply PEFT to the model. This freezes the base model and makes only the LoRA adapter trainable.
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 8. Define Training Arguments and Metrics ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_arguments = TrainingArguments(
    output_dir="./Llama_Rank_2",
    num_train_epochs=3,
    per_device_train_batch_size=8, # Reduced batch size for larger model
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=25,
    learning_rate=2e-4,
    fp16=False, # Disabled for 4-bit training
    bf16=False, # Disabled
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# --- 9. Initialize and Run Trainer ---

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator, # Use the data collator for padding
    compute_metrics=compute_metrics,
)

print("Starting the fine-tuning process for classification...")
trainer.train()
print("Fine-tuning completed.")

# --- 10. Save the Fine-Tuned Model Adapter ---

trainer.save_model(NEW_MODEL_NAME)
print(f"Fine-tuned classification model adapter saved to ./{NEW_MODEL_NAME}")