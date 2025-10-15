import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from safetensors.torch import load_file
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import SegmentFromFile # <-- CORRECT: Importing the function

# ==============================================================================
# SECTION 1: MODEL DEFINITION (Must match the trained model)
# ==============================================================================
class CANEncoder(nn.Module):
    """LSTM-based encoder with Dropout for regularization."""
    def __init__(self, input_features, embedding_dim, hidden_units, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_features, hidden_size=embedding_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_units),
            nn.ReLU(), nn.Dropout(p=dropout_prob),
            nn.Linear(in_features=hidden_units, out_features=2))

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_input)
        embedding = h_n[-1]
        return self.classifier(embedding)

    def get_embedding(self, x, lengths):
        """Method to get only the embedding vector during inference."""
        self.eval()
        with torch.no_grad():
            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed_input)
            embedding = h_n[-1]
            return embedding

# ==============================================================================
# SECTION 2: DATA HANDLING (Must match the training script)
# ==============================================================================
class CANSegmentDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = [torch.tensor(chunk, dtype=torch.float32) for chunk in chunks]
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.chunks)
    def __getitem__(self, idx): return self.chunks[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded_sequences, lengths, labels

# ==============================================================================
# SECTION 3: EVALUATION FUNCTIONS
# ==============================================================================
def evaluate_classifier(model, dataloader, device):
    """Calculates and prints a detailed classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, lengths, labels in tqdm(dataloader, desc="Quantitative Evaluation (Classifier)"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences, lengths)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Classification Report (Test Set) ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=['Normal (0)', 'Attack (1)']))

def visualize_embeddings(model, dataloader, device):
    """Generates a t-SNE plot to visualize embedding clusters."""
    model.eval()
    all_embeddings, all_labels = [], []
    with torch.no_grad():
        for sequences, lengths, labels in tqdm(dataloader, desc="Qualitative Evaluation (Embeddings)"):
            sequences = sequences.to(device)
            embeddings = model.get_embedding(sequences, lengths)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    all_embeddings = np.concatenate(all_embeddings)
    all_labels = np.concatenate(all_labels)
    
    print("\nEmbeddings generated. Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings) - 1))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    print("Plotting results...")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=all_labels,
        palette=sns.color_palette("hsv", 2), legend="full")
    plt.title("t-SNE Visualization of CAN Message Embeddings (Test Set)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title='Class', labels=['Normal', 'Attack'])
    plt.show()

# ==============================================================================
# SECTION 4: MAIN EXECUTION BLOCK FOR EVALUATION
# ==============================================================================
def main_evaluate():
    # --- 1. Define Constants and Hyperparameters ---
    DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/"
    MODEL_PATH = "/home/lisa/Arupreza/UIDS-II/EmbeddingModelSrc/embedding_model/best_can_encoder.safetensors"
    TIME_GAP = 100
    INPUT_FEATURES = 2
    EMBEDDING_DIM = 128
    HIDDEN_UNITS = 64
    DROPOUT_PROB = 0.5
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type.upper()}")

    # --- 2. Load and Process Data to get the Test Set ---
    all_chunks, all_labels = [], []
    print(f"Scanning data files to reconstruct the test set...")
    filenames = sorted([f for f in os.listdir(DATA_DIRECTORY) if f.lower().endswith('.csv')])
    for filename in tqdm(filenames, desc="Processing Files"):
        chunks, labels = SegmentFromFile(DATA_DIRECTORY, filename, TIME_GAP)
        all_chunks.extend(chunks)
        all_labels.extend(labels)

    if not all_chunks:
        print("CRITICAL ERROR: No data was loaded.")
        return

    # CRITICAL: Split data identically to training to isolate the exact same test set.
    # The random_state ensures this split is deterministic.
    # We create train/val splits but will not use them here.
    train_val_chunks, test_chunks, train_val_labels, test_labels = train_test_split(
        all_chunks, all_labels, test_size=0.20, random_state=42, stratify=all_labels)

    test_dataset = CANSegmentDataset(test_chunks, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Isolated {len(test_dataset)} samples for testing.")

    # --- 3. Load the Trained Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please run train.py first.")
        return
        
    model = CANEncoder(INPUT_FEATURES, EMBEDDING_DIM, HIDDEN_UNITS, dropout_prob=DROPOUT_PROB).to(device)
    model.load_state_dict(load_file(MODEL_PATH, device=str(device)))
    print("Model loaded successfully from file.")

    # --- 4. Run Evaluations ---
    evaluate_classifier(model, test_loader, device)
    visualize_embeddings(model, test_loader, device)

if __name__ == '__main__':
    main_evaluate()