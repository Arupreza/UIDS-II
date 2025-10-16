import pandas as pd
import os

# ==============================================================================
# SECTION 1: NORMALIZATION AND FEATURE ENGINEERING HELPERS
# ==============================================================================

def normalize_can_id_by_frequency(df, column_name='CAN_ID'):
    """
    Engineers a feature by categorizing CAN IDs based on their frequency.
    """
    # Calculate the frequency of each unique CAN ID.
    id_counts = df[column_name].value_counts().reset_index()
    id_counts.columns = [column_name, 'Count']

    # Define the categorization logic with non-overlapping bins.
    def assign_category(count):
        if count >= 12000:
            return 1
        elif 5000 <= count < 12000:
            return 2
        elif 2500 <= count < 5000:
            return 3
        else:  # For counts < 2500
            return 4

    # Apply the logic to create a 'Category' column.
    id_counts['Category'] = id_counts['Count'].apply(assign_category)
    id_to_category_map = pd.Series(id_counts.Category.values, index=id_counts[column_name]).to_dict()

    # Use the map to create the new normalized feature column.
    normalized_column_name = f'{column_name}_Norm'
    df[normalized_column_name] = df[column_name].map(id_to_category_map)
    return df

def IntraIDTimeGapNorm(value):
    """
    Normalizes the 'Intra_ID_Time_Gap' value by binning it into categories.
    """
    if value is None: return -1 # Handle potential None values
    if value <= 5.1:
        return 0
    elif value <= 10.1:
        return 1
    elif value <= 20.1:
        return 2
    elif value <= 30.1:
        return 3
    elif value <= 40.1:
        return 4
    elif value <= 50.1:
        return 5
    elif value <= 2010.1:
        return 6
    elif value <= 5010.1:
        return 7
    else:
        return 8

def TimeDeltaTimeGapNorm(value):
    """
    Normalizes the 'Time_Delta' value by binning it into categories.
    """
    if value is None: return -1 # Handle potential None values
    if 0 <= value <= 0.05:
        return 0
    elif 0.05 < value <= 0.1:
        return 1
    elif 0.1 < value <= 0.2:
        return 2
    elif 0.2 < value <= 0.3:
        return 3
    elif 0.3 < value <= 0.4:
        return 4
    elif 0.4 < value <= 0.5:
        return 5
    else:
        return 6

# ==============================================================================
# SECTION 2: CORE DATA PREPROCESSING AND SEGMENTATION
# ==============================================================================

def LoadPreprocessData(file_path):
    """
    Loads and applies a full preprocessing pipeline to a single CSV file.
    """
    try:
        # Assuming CAN_ID is not always present, we won't force column types
        # and will handle potential errors gracefully.
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return None

    # --- Feature Engineering ---
    # Check if required columns exist
    if 'CAN_ID' not in df.columns or 'Time_Offset' not in df.columns:
        print(f"Warning: 'CAN_ID' or 'Time_Offset' not found in {file_path}. Skipping some preprocessing steps.")
        return None

    df = normalize_can_id_by_frequency(df)
    df['Time_Delta'] = df['Time_Offset'].diff()
    df['Intra_ID_Time_Gap'] = df.groupby('CAN_ID')['Time_Offset'].diff()

    # --- Data Cleaning ---
    df.dropna(subset=['Intra_ID_Time_Gap', 'Time_Delta'], inplace=True)
    df = df.reset_index(drop=True)

    # --- Normalization ---
    df['Intra_ID_Time_Gap_Norm'] = df['Intra_ID_Time_Gap'].apply(IntraIDTimeGapNorm)
    df['Time_Delta_Norm'] = df['Time_Delta'].apply(TimeDeltaTimeGapNorm)
    
    # Ensure Label column exists
    if 'Label' not in df.columns:
        print(f"Warning: 'Label' column not found in {file_path}. Assuming 'Normal'.")
        df['Label'] = 'Normal'


    return df

def SegmentFromFile(data_directory, filename, time_gap):
    """
    Loads, preprocesses, and segments data from a single file. This is the
    primary function used by the main training script.
    """
    full_path = os.path.join(data_directory, filename)
    df = LoadPreprocessData(full_path)

    if df is None or df.empty:
        return [], []

    df_subset = df[['Time_Offset', 'Time_Delta_Norm', 'Intra_ID_Time_Gap_Norm', 'Label']].copy()

    chunks, labels = [], []
    min_time, max_time = df_subset['Time_Offset'].min(), df_subset['Time_Offset'].max()
    window_start = min_time

    while window_start <= max_time:
        window_end = window_start + time_gap
        segment_df = df_subset[(df_subset['Time_Offset'] >= window_start) & (df_subset['Time_Offset'] < window_end)]
        if not segment_df.empty:
            features = segment_df[['Time_Delta_Norm', 'Intra_ID_Time_Gap_Norm']].values.astype(int)
            chunks.append(len(features))
            is_attack = any(segment_df['Label'] != 'Normal')
            labels.append(1 if is_attack else 0)
        window_start += time_gap

    return chunks, labels