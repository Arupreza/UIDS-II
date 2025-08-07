import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

########################################################################################
# Data Loading and Preprocessing
########################################################################################
Kia = ['0356', '0366', '0368', '0260', '0329', '0367'] #22813
Sil = ['018E', '00C9', '00D3', '00AA', '00BE', '01ED'] #18670
Tesla = ['0187', '0186', '02D5', '018A', '0256', '0142'] #19163
Gen = ['00B5', '0100', '007A', '00A5', '006A', '0040'] #23803

def load_data(df_name, IDs):
    path = "/home/lisa/Arupreza/Cognitive-Belief-Driven-Q-Learning-for-Vehicle-Model-Agnostic-Intrusion-Detection-in-V-Net/Input_data/"
    df = pd.read_csv(path + df_name)
    df['Intra_ID_Time_Gap'] = df.groupby('CAN_ID')['Time_Offset'].diff().fillna(-1)
    df = df[df["CAN_ID"].isin(IDs)]
    df = df[df['Intra_ID_Time_Gap'] != -1.0].reset_index(drop=True)
    df = df[['Intra_ID_Time_Gap']]
    df['Intra_ID_Time_Gap_Scaled'] = global_normalize(df['Intra_ID_Time_Gap'])
    
    return df

########################################################################################
# Global Normalization
########################################################################################
def global_normalize(data):
    X_global_min = 1.8
    X_global_max = 18.3
    
    # Avoid division by zero
    if X_global_max == X_global_min:
        return np.zeros_like(data)  # Return zeros with same shape as input
    
    # Apply normalization formula: ((X - X_global_min) / (X_global_max - X_global_min)) * 2 - 1
    normalized_data = ((data - X_global_min) / (X_global_max - X_global_min)) * 2 - 1
    
    return normalized_data


########################################################################################
# Chunk Compilation
########################################################################################

def segment_df(df, segment_size=30, drop_incomplete=True):
    arr = df[['Intra_ID_Time_Gap']].to_numpy()
    segments = []
    for start in range(0, arr.shape[0], segment_size):
        seg = arr[start : start + segment_size]
        if drop_incomplete and seg.shape[0] < segment_size:
            continue        # skip the short tail
        segments.append(seg)
    return segments

########################################################################################
# Heatmap Production
########################################################################################

def save_chunk_heatmaps(chunks, save_path, prefix='img'):
    os.makedirs(save_path, exist_ok=True)
    
    for idx, data_subset in enumerate(chunks):
        # ----- 1) grab data as a 2D array, shape = (n_rows, n_cols) = (3, segment_len)
        if hasattr(data_subset, 'to_numpy'):
            arr = data_subset[['Intra_ID_Time_Gap']].to_numpy().T
        else:
            # assume it’s already a NumPy array of shape (n_rows, n_cols) after .T
            arr = np.array(data_subset).T
        
        # ----- 2) normalize each row independently to [0,1]
        row_min = arr.min(axis=1, keepdims=True)
        row_max = arr.max(axis=1, keepdims=True)
        denom   = row_max - row_min
        denom[denom == 0] = 1  # avoid div0 for constant rows
        norm_arr = (arr - row_min) / denom
        
        # ----- 3) plot & save
        plt.figure(figsize=(8, 8))
        sns.heatmap(norm_arr, cmap="coolwarm", 
                    cbar=False,     # hide colorbar if you like
                    xticklabels=False,
                    yticklabels=['Intra_ID_Time_Gap'])
        plt.axis('off')
        
        filename = f"{prefix}_{idx:03d}.png"
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"Saved {len(chunks)} normalized heatmaps to {save_path}")