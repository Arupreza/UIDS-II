# -*- coding: utf-8 -*-
"""
classification_finetune_albert.py

This script fine-tunes the albert/albert-base-v2 model for a binary
classification task on preprocessed CAN bus data.
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import SegmentFromFile

# --- 1. Load Environment Variables ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    print("Warning: Hugging Face token not found.")
else:
    print("Hugging Face token loaded successfully.")

# --- 2. Configuration ---
MODEL_NAME = "albert/albert-base-v2"
DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Split_data/Train/Gen"
VALIDATION_DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Split_data/Val"
NEW_MODEL_NAME = "albert-can-attack-classifier-experimental"
TIME_GAP_TRAIN = 105.0
TIME_GAP_TEST = 100.0

# --- 3. Load and Preprocess Data ---
def load_and_process_data(directory, time_gap):
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

    def format_chunk_to_string(chunk):
        tokens = []
        for pair in chunk:
            token1 = f"T{int(pair[0])}"
            token2 = f"G{int(pair[1])}"
            tokens.append(token1)
            tokens.append(token2)
        return " ".join(tokens)

    formatted_texts = [format_chunk_to_string(chunk) for chunk in all_chunks]
    df = pd.DataFrame({'text': formatted_texts, 'label': all_labels})
    return df

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
    token=hf_token
)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
if "__index_level_0__" in tokenized_datasets["train"].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
else:
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 6. Load Base Model ---
print(f"Loading base model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    token=hf_token,
)

# --- 7. Define Training Arguments and Metrics ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_arguments = TrainingArguments(
    output_dir="./ALBERT_Classifier_Experimental",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    logging_steps=25,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# --- 8. Initialize and Run Trainer ---
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

# --- 9. Save the Fine-Tuned Model ---
trainer.save_model(NEW_MODEL_NAME)
print(f"Fine-tuned classification model saved to ./{NEW_MODEL_NAME}")