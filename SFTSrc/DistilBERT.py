# -*- coding: utf-8 -*-
"""
classification_finetune_distilbert.py

This script fine-tunes the distilbert-base-uncased model for a binary
classification task on preprocessed CAN bus data.

The process includes:
1.  Loading the Hugging Face API token from a .env file.
2.  Loading and preprocessing training data (Kia) and validation data (Tesla) from separate directories.
3.  Converting the numerical sequences into string format to be used as text input.
4.  Loading the DistilBERT base model with a new sequence classification head.
5.  Setting up and running the training process using the standard Hugging Face Trainer.
6.  Evaluating the model's classification performance on the validation set.
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
    DataCollatorWithPadding
)
# Note: BitsAndBytesConfig and PEFT imports are removed as they are not needed.
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

# MODIFIED: Changed model to DistilBERT.
MODEL_NAME = "distilbert-base-uncased"

# Path to the directory containing your training data.
DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/Kia"
# Path to the directory containing your validation data.
VALIDATION_DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/Tesla"

# MODIFIED: Changed new model name to reflect DistilBERT.
NEW_MODEL_NAME = "distilbert-can-attack-classifier"

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
# MODIFIED: Removed the line `tokenizer.pad_token = tokenizer.eos_token`
# DistilBERT's tokenizer already has a pad token.

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
if "__index_level_0__" in tokenized_datasets["train"].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
else:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# --- 6. REMOVED QLoRA and PEFT Configuration ---
# This entire section was removed as it is not needed for a model of DistilBERT's size.


# --- 7. Load Base Model ---

print(f"Loading base model: {MODEL_NAME}")
# MODIFIED: Simplified model loading for standard fine-tuning.
# Removed quantization_config, device_map, and PEFT-related code.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    token=hf_token,
)

# --- 8. Define Training Arguments and Metrics ---

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# MODIFIED: Training arguments adjusted for standard fine-tuning.
training_arguments = TrainingArguments(
    output_dir="./DistilBERT_Classifier",   # Updated output directory
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_torch",                    # Use standard AdamW optimizer
    logging_steps=25,
    learning_rate=2e-5,                     # Standard learning rate for BERT-like models
    fp16=True,                              # Enable mixed-precision for faster training
    bf16=False,
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
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting the fine-tuning process for classification...")
trainer.train()
print("Fine-tuning completed.")

# --- 10. Save the Fine-Tuned Model ---

# Note: This now saves the entire fine-tuned model, not just an adapter.
trainer.save_model(NEW_MODEL_NAME)
print(f"Fine-tuned classification model saved to ./{NEW_MODEL_NAME}")