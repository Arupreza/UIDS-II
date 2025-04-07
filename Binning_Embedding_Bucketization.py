import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib


########################################################################################
def CAN_ID_Categorization(can_ids):
    decimal_values = [int(hex_value, 16) for hex_value in can_ids]
    min_can_id, max_can_id = min(decimal_values), max(decimal_values)
    range_step = (max_can_id - min_can_id) / 100

    categorized_ids = []
    for hex_value in can_ids:
        decimal_value = int(hex_value, 16)
        category_index = int((decimal_value - min_can_id) // range_step) + 1
        category_index = min(category_index, 100)  # clamp
        category = f"CAN_{category_index}"
        categorized_ids.append((hex_value, category))

    return pd.DataFrame(categorized_ids, columns=['CAN_ID', 'Cat_CAN_ID'])




def Time_Gap_Categorization(numbers):
    numbers = numbers.dropna()
    min_value, max_value = min(numbers), max(numbers)
    range_step = (max_value - min_value) / 1000

    categorized = []
    for value in numbers:
        category_index = int((value - min_value) // range_step) + 1
        category_index = min(category_index, 1000)  # clamp
        category = f"TG_{category_index}"
        categorized.append((value, category))

    return pd.DataFrame(categorized, columns=['Time_Gap', 'Cat_Time_Gap'])


########################################################################################

# ----------------------------
# Embedding Dataset
# ----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, df):
        self.can_ids = torch.tensor(df['CAN_ID_Encoded'].values, dtype=torch.long)
        self.time_gaps = torch.tensor(df['Time_Gap_Encoded'].values, dtype=torch.long)

    def __len__(self):
        return len(self.can_ids)

    def __getitem__(self, idx):
        return self.can_ids[idx], self.time_gaps[idx]

# ----------------------------
# Embedding Model
# ----------------------------
class CANEmbeddingModel(nn.Module):
    def __init__(self, can_vocab_size=100, gap_vocab_size=1000):
        super().__init__()
        self.can_embed = nn.Embedding(can_vocab_size, 6)
        self.gap_embed = nn.Embedding(gap_vocab_size, 11)

    def forward(self, can_id, gap_id):
        can_vec = self.can_embed(can_id)  # [batch, 6]
        gap_vec = self.gap_embed(gap_id)  # [batch, 11]
        return torch.cat([can_vec, gap_vec], dim=1)  # [batch, 17]


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