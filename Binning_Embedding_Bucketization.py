import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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