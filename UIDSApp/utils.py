import pandas as pd
import os

def IntraIDTimeGapNorm(v):
    if v is None: return -1
    if v <= 5.1: return 0
    elif v <= 10.1: return 1
    elif v <= 20.1: return 2
    elif v <= 30.1: return 3
    elif v <= 40.1: return 4
    elif v <= 50.1: return 5
    elif v <= 2010.1: return 6
    elif v <= 5010.1: return 7
    else: return 8

def TimeDeltaTimeGapNorm(v):
    if v is None: return -1
    if 0 <= v <= 0.05: return 0
    elif v <= 0.1: return 1
    elif v <= 0.2: return 2
    elif v <= 0.3: return 3
    elif v <= 0.4: return 4
    elif v <= 0.5: return 5
    else: return 6


# ---------- VALIDATION (with labels) ----------
def SegmentForValidation(data_dir, filename, time_gap):
    full_path = os.path.join(data_dir, filename)
    df = pd.read_csv(full_path)
    df['Time_Delta'] = df['Time_Offset'].diff()
    df['Intra_ID_Time_Gap'] = df.groupby('CAN_ID')['Time_Offset'].diff()
    df.dropna(subset=['Time_Delta','Intra_ID_Time_Gap'], inplace=True)

    df['Time_Delta_Norm'] = df['Time_Delta'].apply(TimeDeltaTimeGapNorm)
    df['Intra_ID_Time_Gap_Norm'] = df['Intra_ID_Time_Gap'].apply(IntraIDTimeGapNorm)
    if 'Label' not in df.columns: df['Label'] = 'Normal'

    chunks, labels = [], []
    start, end = df['Time_Offset'].min(), df['Time_Offset'].max()
    t = start
    while t <= end:
        window = df[(df['Time_Offset'] >= t) & (df['Time_Offset'] < t + time_gap)]
        if not window.empty:
            features = window[['Time_Delta_Norm','Intra_ID_Time_Gap_Norm']].values.astype(int)
            chunks.append(features)
            labels.append(1 if any(window['Label']!='Normal') else 0)
        t += time_gap
    return chunks, labels


# ---------- INFERENCE (no labels) ----------
def SegmentForInference(data_dir, filename, time_gap):
    full_path = os.path.join(data_dir, filename)
    df = pd.read_csv(full_path)
    df['Time_Delta'] = df['Time_Offset'].diff()
    df['Intra_ID_Time_Gap'] = df.groupby('CAN_ID')['Time_Offset'].diff()
    df.dropna(subset=['Time_Delta','Intra_ID_Time_Gap'], inplace=True)

    df['Time_Delta_Norm'] = df['Time_Delta'].apply(TimeDeltaTimeGapNorm)
    df['Intra_ID_Time_Gap_Norm'] = df['Intra_ID_Time_Gap'].apply(IntraIDTimeGapNorm)

    chunks = []
    start, end = df['Time_Offset'].min(), df['Time_Offset'].max()
    t = start
    while t <= end:
        window = df[(df['Time_Offset'] >= t) & (df['Time_Offset'] < t + time_gap)]
        if not window.empty:
            features = window[['Time_Delta_Norm','Intra_ID_Time_Gap_Norm']].values.astype(int)
            chunks.append(features)
        t += time_gap
    return chunks