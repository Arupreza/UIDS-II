import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from safetensors.torch import save_file, load_file
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils import SegmentFromFile
from sklearn.model_selection import train_test_split # <-- ADDITION: For splitting data

# = a============================================================================
# SECTION 1: MODEL DEFINITION (WITH REGULARIZATION)
# ==============================================================================

class CANEncoder(nn.Module):
    def __init__(self, input_features, embedding_dim, hidden_units, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0 # <-- ADDITION: LSTM dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob), # <-- ADDITION: Dropout layer
            nn.Linear(in_features=hidden_units, out_features=2)
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_input)
        embedding = h_n[-1]
        output = self.classifier(embedding)
        return output

    def get_embedding(self, x, lengths):
        self.eval()
        with torch.no_grad():
            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed_input)
            embedding = h_n[-1]
            return embedding

# ==============================================================================
# SECTION 2: DATA HANDLING (PYTORCH DATASET AND COLLATE FUNCTION)
# ==============================================================================

class CANSegmentDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = [torch.tensor(chunk, dtype=torch.float32) for chunk in chunks]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded_sequences, lengths, labels

# ==============================================================================
# SECTION 4: TRAINING & VALIDATION LOGIC (WITH EARLY STOPPING)
# ==============================================================================

def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    """Handles training and validation, with early stopping."""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 3 # Number of epochs to wait for improvement before stopping

    print(f"Starting model training on {str(device).upper()}...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for sequences, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, lengths, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # --- Early Stopping Logic ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            save_file(model.state_dict(), save_path)
            print(f"Validation loss decreased. Saving best model to {save_path}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print("Training finished.")
    # Load the best model weights back
    model.load_state_dict(load_file(save_path))


# ==============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    # --- 1. Define Constants and Hyperparameters ---
    DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/"
    TIME_GAP = 100

    INPUT_FEATURES = 2
    EMBEDDING_DIM = 128
    HIDDEN_UNITS = 64
    NUM_EPOCHS = 20 # Increase epochs, early stopping will find the best one
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-5 # <-- ADDITION: L2 Regularization
    DROPOUT_PROB = 0.5 # <-- ADDITION: Dropout probability
    VALIDATION_SPLIT = 0.2 # <-- ADDITION: 20% of data for validation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")
    
    # --- 2. Discover and Process All Data Files ---
    all_chunks = []
    all_labels = []
    # ... (rest of your data loading code is fine) ...
    print(f"Scanning for data files in '{DATA_DIRECTORY}'...")
    try:
        filenames = sorted([f for f in os.listdir(DATA_DIRECTORY) if f.lower().endswith('.csv')])
        if not filenames:
            print("Error: No CSV files found. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{DATA_DIRECTORY}'.")
        return

    print("Loading and segmenting data...")
    for filename in filenames:
        print(f"Processing {filename}...")
        chunks, labels = SegmentFromFile(DATA_DIRECTORY, filename, TIME_GAP)
        all_chunks.extend(chunks)
        all_labels.extend(labels)
    
    if not all_chunks:
        print("No data was loaded. Exiting.")
        return

    print(f"\nTotal segments loaded: {len(all_chunks)}")

    # --- 3. Split Data and Create DataLoaders ---
    train_chunks, val_chunks, train_labels, val_labels = train_test_split(
        all_chunks, all_labels, test_size=VALIDATION_SPLIT, random_state=42, stratify=all_labels
    )

    train_dataset = CANSegmentDataset(train_chunks, train_labels)
    val_dataset = CANSegmentDataset(val_chunks, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Data split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # --- 4. Initialize Model, Loss, and Optimizer ---
    model = CANEncoder(INPUT_FEATURES, EMBEDDING_DIM, HIDDEN_UNITS, dropout_prob=DROPOUT_PROB)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- 5. Train the Model ---
    save_folder = "embedding_model"
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, "best_can_encoder.safetensors")
    
    train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, model_path)

    print(f"\nBest model saved at {model_path}.")

if __name__ == '__main__':
    main()