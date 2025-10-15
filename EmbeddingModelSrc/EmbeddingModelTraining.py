import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau # <-- ADDITION: Learning rate scheduler
import numpy as np
import pandas as pd
import os
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
from safetensors.torch import save_file, load_file

# ==============================================================================
# SECTION 1: DATA HANDLING (TEXT-BASED)
# ==============================================================================

def load_and_prepare_as_text(directory, filename):
    """Loads a CSV and converts CAN messages into single strings."""
    try:
        df = pd.read_csv(os.path.join(directory, filename), dtype=str).fillna('')
        df['can_string'] = df['CAN_ID'] + ' ' + df['Data']
        return df['can_string'].tolist()
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

class CharVocabulary:
    """Manages the mapping between characters and integer tokens."""
    def __init__(self, texts):
        self.pad_token = "<PAD>"; self.unk_token = "<UNK>"
        unique_chars = sorted(list(set("".join(texts))))
        self.vocab = [self.pad_token, self.unk_token] + unique_chars
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab)}
    def __len__(self): return len(self.vocab)
    def save(self, filepath):
        with open(filepath, 'w') as f: json.dump({'char_to_int': self.char_to_int}, f)
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f: data = json.load(f)
        vocab = cls([]); vocab.char_to_int = data['char_to_int']
        vocab.int_to_char = {i: c for c, i in vocab.char_to_int.items()}
        return vocab

class CANSegmentTextDataset(Dataset):
    """Dataset for text-based CAN sequences."""
    def __init__(self, segments, vocab):
        self.segments = segments; self.vocab = vocab
    def __len__(self): return len(self.segments)
    def __getitem__(self, idx):
        segment = self.segments[idx]
        tokens = [self.vocab.char_to_int.get(c, 1) for c in segment]
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn_text(batch):
    sequences = batch
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences

# ==============================================================================
# SECTION 2: TRANSFORMER AUTOENCODER MODEL
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Injects position information into the token embeddings."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)].transpose(0, 1)

class TransformerEncoder(nn.Module):
    """Encodes a sequence of tokens into a single embedding vector."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        # We use the mean of all token embeddings as the final sequence embedding.
        # This is a simple and effective way to get a single vector for the whole sequence.
        return torch.mean(output, dim=1)

class TransformerDecoder(nn.Module):
    """Decodes an embedding back into a sequence of character logits."""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        # The 'memory' comes from the encoder and contains the contextual understanding of the input.
        output = self.transformer_decoder(tgt, memory, tgt_key_padding_mask=tgt_mask)
        return self.fc_out(output)

class CANAutoencoderTransformer(nn.Module):
    """
    The complete Transformer-based Autoencoder.
    This model's design choices mirror those in larger models like Llama.
    """
    def __init__(self, vocab_size, 
                 d_model=128,          # The main dimensionality of the model's vectors. Similar to LLAMA's 'hidden_size'.
                 nhead=4,              # Number of self-attention heads. LLAMA uses many more, but 4 is good for a small model.
                 num_encoder_layers=2, # Number of stacked encoder blocks. Deeper models learn more complex patterns.
                 num_decoder_layers=2, # Number of stacked decoder blocks.
                 dim_feedforward=256,  # The size of the internal feed-forward network.
                 dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
    
    def forward(self, src):
        src_padding_mask = (src == 0) # Mask for padding tokens.
        
        # 1. Encode the source sequence to get the context vector ('memory').
        memory = self.encoder(src, src_padding_mask)
        
        # 2. Prepare the memory for the decoder. It expects a sequence, so we expand our single vector.
        memory_for_decoder = memory.unsqueeze(1).repeat(1, src.size(1), 1)

        # 3. Decode the memory to reconstruct the original sequence.
        output_logits = self.decoder(src, memory_for_decoder, src_padding_mask)
        return output_logits

# ==============================================================================
# SECTION 3: TRAINING & INFERENCE
# ==============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_path, num_epochs=20, patience=3):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            sequences = sequences.to(device)
            optimizer.zero_grad()
            output_logits = model(sequences)
            loss = criterion(output_logits.view(-1, output_logits.size(2)), sequences.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                sequences = sequences.to(device)
                output_logits = model(sequences)
                loss = criterion(output_logits.view(-1, output_logits.size(2)), sequences.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}")

        # <-- ADDITION: Step the scheduler based on validation loss -->
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_file(model.encoder.state_dict(), save_path)
            print(f"Val loss improved. Saving best ENCODER to {save_path}")
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

def main():
    # --- Config ---
    DATA_DIRECTORY = "/home/lisa/Arupreza/UIDS-II/Input_data/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "saved_models/best_can_encoder_transformer.safetensors"
    os.makedirs("saved_models", exist_ok=True)

    # --- Data Loading ---
    all_texts = []
    for filename in tqdm(os.listdir(DATA_DIRECTORY), desc="Loading files"):
        if filename.endswith(".csv"):
            all_texts.extend(load_and_prepare_as_text(DATA_DIRECTORY, filename))
    
    vocab = CharVocabulary(all_texts)
    vocab.save("char_vocab_transformer.json")
    train_texts, val_texts = train_test_split(all_texts, test_size=0.2, random_state=42)
    train_ds = CANSegmentTextDataset(train_texts, vocab)
    val_ds = CANSegmentTextDataset(val_texts, vocab)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn_text)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn_text)
    
    # --- Model Training ---
    model = CANAutoencoderTransformer(vocab_size=len(vocab)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding token
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # <-- ADDITION: Initialize the scheduler -->
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, save_path)
    
    # --- Inference Demo ---
    print("\n--- INFERENCE DEMO ---")
    # Define the architecture for the inference model, matching the saved encoder
    inference_encoder = TransformerEncoder(vocab_size=len(vocab), d_model=128, nhead=4, num_layers=2, dim_feedforward=256).to(device)
    inference_encoder.load_state_dict(load_file(save_path, device=str(device)))
    inference_encoder.eval()

    sample_text = "3A1 FF 00 FF 00 FF 00"
    tokens = torch.tensor([[vocab.char_to_int.get(c, 1) for c in sample_text]], dtype=torch.long).to(device)
    mask = (tokens == 0) # Create the padding mask
    with torch.no_grad():
        embedding = inference_encoder(tokens, mask).cpu().numpy().flatten()

    print(f"Generated a {embedding.shape[0]}-dimensional embedding for the sample text.")
    print(f"Preview: {embedding[:10]}")

if __name__ == '__main__':
    main()