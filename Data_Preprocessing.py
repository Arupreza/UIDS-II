import pandas as pd
import numpy as np
import torch
import torch.nn as nn

########################################################################################
# Process CSV File
def process_csv_file(file_path):
    # Read CSV file
    data = pd.read_csv(file_path)

    # Remove the last row
    data = data.iloc[:-1]

    # Fill missing values with -1
    data = data.fillna(-1)

    return data

########################################################################################
# Divide CAN ID into three parts
def divide_into_parts(hex_values, num_parts=3):
    part_size = len(hex_values) // num_parts
    remainder = len(hex_values) % num_parts
    parts = []

    start = 0
    for i in range(num_parts):
        end = start + part_size + (1 if i < remainder else 0)
        parts.append(hex_values[start:end])
        start = end

    return parts

########################################################################################
# CAN ID Embedding
class CANIDEmbedding(nn.Module):
    def __init__(self, unique_ids, embedding_dim):
        super().__init__()
        self.id_to_index = {can_id: idx for idx, can_id in enumerate(unique_ids)}
        self.embedding = nn.Embedding(len(unique_ids), embedding_dim)

    def forward(self, can_ids):
        indices = torch.tensor([self.id_to_index.get(cid, 0) for cid in can_ids], dtype=torch.long)
        return self.embedding(indices)

########################################################################################
# Categorize CAN IDs using embeddings
def categorize_can_ids_embed(df, embedder):
    can_id_list = df['CAN_ID'].tolist()
    with torch.no_grad():
        embedded = embedder(can_id_list).numpy()

    # Attach embedding vectors to DataFrame (can replace this with PCA if dimensionality is high)
    for i in range(embedded.shape[1]):
        df[f'CAN_Embed_{i}'] = embedded[:, i]

    df = df[['Time_Offset', 'CAN_ID', 'Time_Gap'] + [f'CAN_Embed_{i}' for i in range(embedded.shape[1])]]
    return df

########################################################################################
# Chunk Compilation
def segment(df, time_gap):
    embed_cols = [col for col in df.columns if col.startswith('CAN_Embed_')]
    df = df[["Time_Offset", "Time_Gap"] + embed_cols]
    up = float(df['Time_Offset'][0])
    low = up + time_gap
    chunk = []

    for i in df.Time_Offset:
        if float(i) <= low:
            out = df[(df['Time_Offset'] >= float(up)) & (df['Time_Offset'] <= float(low))]
            out = np.array(out[["Time_Gap"] + embed_cols])
            chunk.append(out)

    return chunk