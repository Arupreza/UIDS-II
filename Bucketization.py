import pandas as pd
import numpy as np
import torch
import torch.nn as nn


########################################################################################
def CAN_ID_Categorization(can_ids):
    # Convert all CAN IDs to decimal
    decimal_values = [int(hex_value, 16) for hex_value in can_ids]

    # Define the lowest and highest possible CAN ID from the list
    min_can_id = min(decimal_values)
    max_can_id = max(decimal_values)

    # Calculate the ranges for dividing the CAN ID into 30 categories
    range_step = (max_can_id - min_can_id) / 100  # Step for each category

    # List to hold the results
    categorized_ids = []

    # Process each CAN ID in the list
    for hex_value in can_ids:
        # Convert hex value to decimal
        decimal_value = int(hex_value, 16)

        # Determine the category based on the decimal value
        category_index = int((decimal_value - min_can_id) // range_step) + 1
        category = f"CAN_{category_index}"  # Format category as CAN_1, CAN_2, ..., CAN_30

        # Append the result as a tuple (original CAN ID, category)
        categorized_ids.append((hex_value, category))

    # Create a DataFrame from the list of categorized CAN IDs
    categorized_ids = pd.DataFrame(categorized_ids, columns=['CAN_ID', 'Cat_CAN_ID'])
    
    return categorized_ids



def Time_Gap_Categorization(numbers):
    # Remove NaN values from the list (if any)
    numbers = numbers.dropna()

    # Define the lowest and highest possible number from the list
    min_value = min(numbers)
    max_value = max(numbers)

    # Calculate the range step for dividing the values into 100 equal parts
    range_step = (max_value - min_value) / 1000  # Step for each category

    # List to hold the results
    categorized_numbers = []

    # Process each number in the list
    for value in numbers:
        # Determine the category based on the value
        category_index = int((value - min_value) // range_step) + 1
        category = f"TG_{category_index}"  # Format category as TG_1, TG_2, ..., TG_100

        # Append the result as a tuple (original number, category)
        categorized_numbers.append((value, category))

    # Create a DataFrame from the list of categorized numbers
    categorized_numbers = pd.DataFrame(categorized_numbers, columns=['Time_Gap', 'Cat_Time_Gap'])
    
    return categorized_numbers

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