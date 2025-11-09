import pandas as pd
import numpy as np
import os


# ===========================================================
# SECTION 1: FEATURE NORMALIZATION
# ===========================================================

def IntraIDTimeGapNorm(value):
    """Normalize 'Intra_ID_Time_Gap' (in ms) into discrete bins."""
    if value is None:
        return -1
    if value <= 5.1: return 0
    elif value <= 10.1: return 1
    elif value <= 20.1: return 2
    elif value <= 30.1: return 3
    elif value <= 40.1: return 4
    elif value <= 50.1: return 5
    elif value <= 2010.1: return 6
    elif value <= 5010.1: return 7
    else: return 8


def TimeDeltaTimeGapNorm(value):
    """Normalize 'Time_Delta' (in ms) into discrete bins."""
    if value is None:
        return -1
    if value <= 5.0: return 0
    elif value <= 10.0: return 1
    elif value <= 20.0: return 2
    elif value <= 30.0: return 3
    elif value <= 40.0: return 4
    elif value <= 50.0: return 5
    else: return 6


# ===========================================================
# SECTION 2: CORE DATA PREPROCESSING
# ===========================================================

def LoadPreprocessData(file_path: str):
    """
    Load and preprocess a CAN CSV file.
    Handles both labeled and unlabeled datasets.
    Returns a cleaned and normalized DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

    # Basic validation
    if "CAN_ID" not in df.columns or "Time_Offset" not in df.columns:
        print(f"⚠️ Missing 'CAN_ID' or 'Time_Offset' in {file_path}. Skipping.")
        return None

    # Temporal features
    df["Time_Delta"] = df["Time_Offset"].diff()
    df["Intra_ID_Time_Gap"] = df.groupby("CAN_ID")["Time_Offset"].diff()

    # Drop invalid rows
    df.dropna(subset=["Intra_ID_Time_Gap", "Time_Delta"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Apply normalization
    df["Intra_ID_Time_Gap_Norm"] = df["Intra_ID_Time_Gap"].apply(IntraIDTimeGapNorm)
    df["Time_Delta_Norm"] = df["Time_Delta"].apply(TimeDeltaTimeGapNorm)

    return df


# ===========================================================
# SECTION 3: SEGMENTATION FUNCTIONS
# ===========================================================

def SegmentForValidation(data_directory: str, filename: str, time_gap: float):
    """
    Segment CAN data for validation or testing.
    Returns (chunks, labels) if 'Label' exists in data.
    Each chunk corresponds to messages within a fixed time window.
    """
    full_path = os.path.join(data_directory, filename)
    df = LoadPreprocessData(full_path)
    if df is None or df.empty:
        return [], []

    if "Label" not in df.columns:
        print(f"⚠️ No 'Label' column found in {filename}. Defaulting all to 'Normal'.")
        df["Label"] = "Normal"

    df_subset = df[["Time_Offset", "Time_Delta_Norm", "Intra_ID_Time_Gap_Norm", "Label"]].copy()

    chunks, labels = [], []
    min_time, max_time = df_subset["Time_Offset"].min(), df_subset["Time_Offset"].max()
    window_start = min_time

    while window_start <= max_time:
        window_end = window_start + time_gap
        seg = df_subset[
            (df_subset["Time_Offset"] >= window_start) &
            (df_subset["Time_Offset"] < window_end)
        ]

        if not seg.empty:
            features = seg[["Time_Delta_Norm", "Intra_ID_Time_Gap_Norm"]].values.astype(int)
            chunks.append(features)
            is_attack = any(seg["Label"] != "Normal")
            labels.append(1 if is_attack else 0)

        window_start += time_gap

    return chunks, labels


def SegmentForInference(data_directory: str, filename: str, time_gap: float):
    """
    Segment CAN data for real-time deployment (no labels).
    Returns only feature chunks (no labels).
    """
    full_path = os.path.join(data_directory, filename)
    df = LoadPreprocessData(full_path)
    if df is None or df.empty:
        return []

    df_subset = df[["Time_Offset", "Time_Delta_Norm", "Intra_ID_Time_Gap_Norm"]].copy()

    chunks = []
    min_time, max_time = df_subset["Time_Offset"].min(), df_subset["Time_Offset"].max()
    window_start = min_time

    while window_start <= max_time:
        window_end = window_start + time_gap
        seg = df_subset[
            (df_subset["Time_Offset"] >= window_start) &
            (df_subset["Time_Offset"] < window_end)
        ]

        if not seg.empty:
            features = seg[["Time_Delta_Norm", "Intra_ID_Time_Gap_Norm"]].values.astype(int)
            chunks.append(features)

        window_start += time_gap

    return chunks


# ===========================================================
# SECTION 4: DYNAMIC CHUNK SIZE ESTIMATION
# ===========================================================

def ChunkSizeMatch(data_input, chunk_size: int = 265, max_time_ms: int = 3000):
    """
    Estimate average segment duration for time-based segmentation.

    Args:
        data_input: pd.DataFrame | list[str]
            Either a DataFrame or a list of CSV file paths.
        chunk_size: int
            Number of CAN messages per chunk (default=265).
        max_time_ms: int
            Maximum allowed chunk duration (default=3000 ms).

    Returns:
        float: Average time (in seconds) per valid chunk.
    """
    # --- Input handling ---
    if isinstance(data_input, list):
        if len(data_input) == 0:
            raise ValueError("Empty file list passed to ChunkSizeMatch()")
        first_file = data_input[0]
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"File not found: {first_file}")
        df = pd.read_csv(first_file)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError("data_input must be a DataFrame or list of CSV file paths")

    # --- Validation ---
    if "Time_Offset" not in df.columns:
        raise KeyError("Missing required column 'Time_Offset' in CAN data")

    total_msgs = len(df)
    valid_chunk_times = []

    # --- Segmentation iteration ---
    for start_idx in range(0, total_msgs, chunk_size):
        end_idx = start_idx + chunk_size
        segment_df = df.iloc[start_idx:end_idx]

        if len(segment_df) >= int(chunk_size * 0.8):
            start_time = segment_df["Time_Offset"].iloc[0]
            end_time = segment_df["Time_Offset"].iloc[-1]
            duration_ms = end_time - start_time

            if 0 < duration_ms <= max_time_ms:
                valid_chunk_times.append(duration_ms)

    # --- Average computation ---
    avg_time_ms = np.mean(valid_chunk_times) if valid_chunk_times else 0.0
    avg_time_s = round(avg_time_ms / 1000.0, 3)

    print(f"[ChunkSizeMatch] Computed average chunk time: {avg_time_s:.3f} sec")
    return avg_time_s