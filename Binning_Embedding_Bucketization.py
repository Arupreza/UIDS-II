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
# Data Loading and Preprocessing
########################################################################################
def load_data(df_name):
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    le = LabelEncoder()
    scaler = MinMaxScaler()

    path = "/home/lisa/Arupreza/Cognitive-Belief-Driven-Q-Learning-for-Vehicle-Model-Agnostic-Intrusion-Detection-in-V-Net/Input_data/"
    df = pd.read_csv(path + df_name)
    df['Intra_ID_Time_Gap'] = (df.groupby('CAN_ID')['Time_Offset'].diff().fillna(-1))
    df['Time_Gap'] = df['Time_Offset'].diff().fillna(-1)
    df = df[df['Intra_ID_Time_Gap'] != -1.0].reset_index(drop=True)
    df_ = CAN_ID_Categorization(df['CAN_ID'], 10)
    df__ = CAN_ID_Categorization(df['CAN_ID'], 10)
    df["Cat_CAN_ID"] = df_['Cat_CAN_ID']
    df = df[['Cat_CAN_ID', 'Intra_ID_Time_Gap', 'Time_Gap', 'Label']]
    df['Cat_CAN_ID'] = le.fit_transform(df['Cat_CAN_ID'])
    df[['Cat_CAN_ID', 'Intra_ID_Time_Gap', 'Time_Gap']] = scaler.fit_transform(df[['Cat_CAN_ID', 'Intra_ID_Time_Gap', 'Time_Gap']])
    
    return df

########################################################################################
# CAN ID Categorization
########################################################################################
def CAN_ID_Categorization(can_ids, div):
    decimal_values = [int(hex_value, 16) for hex_value in can_ids]
    min_can_id, max_can_id = min(decimal_values), max(decimal_values)
    range_step = (max_can_id - min_can_id) / div

    categorized_ids = []
    for hex_value in can_ids:
        decimal_value = int(hex_value, 16)
        category_index = int((decimal_value - min_can_id) // range_step) + 1
        category_index = min(category_index, div)  # clamp
        category = f"CAT_{category_index}"
        categorized_ids.append((hex_value, category))

    return pd.DataFrame(categorized_ids, columns=['CAN_ID', 'Cat_CAN_ID'])

########################################################################################
# Time Gap Categorization
########################################################################################

def Time_Gap_Categorization(time_div, div):
    min_value, max_value = min(time_div), max(time_div)
    range_step = (max_value - min_value) / div

    categorized = []
    for value in time_div:
        category_index = int((value - min_value) // range_step) + 1
        category_index = min(category_index, div)  # clamp
        category = f"CATT_{category_index}"
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


# ----------------------------
# Chunk Compilation
# ----------------------------
def segment(df, chunk_size=17, label=0):
    # This function will split the data into chunks of the given size (default 17 values)
    chunks = []
    
    # Iterate over the dataframe in steps of 'chunk_size'
    for i in range(0, len(df), chunk_size):
        # Get the chunk of the dataframe
        chunk_data = df.iloc[i:i+chunk_size]
        
        # Check if the chunk has the correct size
        if chunk_data.shape[0] == chunk_size and chunk_data.shape[1] == chunk_size:
            # Convert chunk data into a numpy array and append to the chunk list
            chunks.append([np.array(chunk_data), label])
    
    return chunks