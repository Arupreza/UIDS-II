# -*- coding: utf-8 -*-
"""
TiansClassifier.py

This script trains a custom Titans (Memory As Context) architecture from scratch 
for binary classification on CAN bus data, replacing MobileBERT.
"""
import os
import glob
import torch
import torch.nn as nn
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import SegmentFromFile  # Ensure this file exists in your directory

# --- 1. Load Environment Variables ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# --- 2. Configuration ---
# We use the MobileBERT tokenizer for input processing, but the model itself is custom
TOKENIZER_NAME = "google/mobilebert-uncased" 
DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Split_data/Train/Gen"
VALIDATION_DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Split_data/Val"
NEW_MODEL_NAME = "titans-mac-can-classifier"
TIME_GAP_TRAIN = 105.0
TIME_GAP_TEST = 98.0

# --- 3. Define Titans Architecture (MAC Variant) ---

class TitansConfig(PretrainedConfig):
    model_type = "titans"
    def __init__(
        self,
        vocab_size=30522,       # Matches MobileBERT tokenizer vocab
        hidden_size=256,        # Embedding & Memory dimension
        num_hidden_layers=4,    # Depth of Core Attention
        num_attention_heads=4,
        intermediate_size=1024,
        num_persistent_tokens=16, # Fixed "Knowledge" tokens
        segment_len=128,          # Chunk size for memory updates
        num_labels=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_persistent_tokens = num_persistent_tokens
        self.segment_len = segment_len
        self.num_labels = num_labels

class NeuralMemory(nn.Module):
    """
    Simulates the Neural Memory update. 
    For stable training in this classification script, we use a GRU Cell 
    to represent the stateful update mechanism M_t = f(M_{t-1}, x_t).
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.memory_cell = nn.GRUCell(input_dim, hidden_dim)
    
    def forward(self, x, h_prev):
        # x: Summary of current segment
        # h_prev: Previous memory state (weights abstraction)
        return self.memory_cell(x, h_prev)

class TitansMACClassifier(PreTrainedModel):
    config_class = TitansConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        
        # 1. Embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # 2. Persistent Memory (Learnable, fixed context vector)
        self.persistent_memory = nn.Parameter(torch.randn(config.num_persistent_tokens, config.hidden_size))
        
        # 3. Long-Term Memory (Neural Memory Module)
        self.neural_memory = NeuralMemory(config.hidden_size, config.hidden_size)
        
        # 4. Short-Term Memory (Core Attention)
        # Standard Transformer Encoder to process the [Persistent + Context + Segment]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads, 
            dim_feedforward=config.intermediate_size,
            batch_first=True,
            norm_first=True # Better stability for training from scratch
        )
        self.core_attention = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)
        
        # 5. Classifier Head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fct = nn.CrossEntropyLoss()

        # Initialize weights
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Shape: [Batch, Seq_Len]
        x = self.embedding(input_ids)
        batch_size, seq_len, hidden_size = x.shape
        
        # Padding for Segmentation
        # We must ensure the sequence length is divisible by segment_len
        remainder = seq_len % self.config.segment_len
        if remainder > 0:
            pad_len = self.config.segment_len - remainder
            padding = torch.zeros(batch_size, pad_len, hidden_size, device=x.device)
            x = torch.cat([x, padding], dim=1)
        
        segments = x.split(self.config.segment_len, dim=1)
        
        # Initialize Memory State (h_0)
        memory_state = torch.zeros(batch_size, hidden_size, device=x.device)
        
        # Iterate through segments (The Titans Recurrence)
        for segment in segments:
            # A. Retrieval: Use current memory state as context
            # Shape: [Batch, 1, Hidden]
            context = memory_state.unsqueeze(1)
            
            # B. Core Processing (MAC): 
            # Input = [Persistent Memory] + [Long-Term Context] + [Current Segment]
            persistent_batch = self.persistent_memory.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Concatenate: [Batch, N_persist + 1 + Seg_Len, Hidden]
            combined_input = torch.cat([persistent_batch, context, segment], dim=1)
            
            # Run Attention
            core_out = self.core_attention(combined_input)
            
            # Extract output corresponding to the segment (last segment_len tokens)
            segment_out = core_out[:, -self.config.segment_len:, :]
            
            # C. Memory Update:
            # Create a summary vector of the current segment (Mean Pooling)
            segment_summary = segment_out.mean(dim=1)
            # Update memory state
            memory_state = self.neural_memory(segment_summary, memory_state)
            
        # Final Prediction based on the final Memory State
        # The memory state now contains the compressed history of the entire sequence
        logits = self.classifier(memory_state)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )

# --- 4. Data Loading & Preprocessing ---

def load_and_process_data(directory, time_gap):
    print(f"Loading data from: {directory}")
    all_chunks = []
    all_labels = []

    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {directory}")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        # Using utils.SegmentFromFile
        chunks, labels = SegmentFromFile(directory, filename, time_gap=time_gap)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    print(f"Total segments processed: {len(all_chunks)}")

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

# Load Data
train_df = load_and_process_data(DATA_DIRECTORY, TIME_GAP_TRAIN)
test_df = load_and_process_data(VALIDATION_DATA_DIRECTORY, TIME_GAP_TEST)

num_labels = pd.concat([train_df['label'], test_df['label']]).nunique()
print(f"Number of labels: {num_labels}")

# Create Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
raw_datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Tokenization
print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=hf_token)

def tokenize_function(examples):
    # Truncate to 512. Titans splits this into chunks of 128 internally.
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=512)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
if "__index_level_0__" in tokenized_datasets["train"].column_names:
    tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 5. Initialize Model & Training Setup ---

print("Initializing Titans Model from scratch...")
config = TitansConfig(num_labels=num_labels)
model = TitansMACClassifier(config)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

training_arguments = TrainingArguments(
    output_dir="./Titans_Classifier_Output",
    num_train_epochs=15,             # Increased for training from scratch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_torch",
    learning_rate=1e-4,              # Higher LR for initialization
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    remove_unused_columns=False,     # CRITICAL: Required for custom models
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 6. Execution ---
print("Starting Titans training...")
trainer.train()
print("Training completed.")

trainer.save_model(NEW_MODEL_NAME)
print(f"Model saved to {NEW_MODEL_NAME}")