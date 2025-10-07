import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def LoadPreprocessData(df):
    """
    Loads a CSV file and applies a full preprocessing pipeline.

    Args:
        file_path (str): The full path to the input CSV file.

    Returns:
        pd.DataFrame: The fully preprocessed DataFrame.
    """
    file_path = "/home/lisa/Arupreza/UIDS-II/Input_data/"
    # --- Load and perform initial feature engineering ---
    df = pd.read_csv(file_path + df)
    df = normalize_can_id_by_frequency(df)
    df['Time_Delta'] = df['Time_Offset'].diff()
    df['Intra_ID_Time_Gap'] = df.groupby('CAN_ID')['Time_Offset'].diff()

    # --- Correctly calculate the time delta WITHIN each normalized group ---
    # This single line replaces your four incorrect lines.
    df['CAN_ID_Norm_Time_Delta'] = df.groupby('CAN_ID_Norm')['Time_Offset'].diff()

    # --- Clean the data by removing rows with NaN deltas and reset the index ---
    # We combine the fillna and filtering logic for clarity
    df.fillna(-1, inplace=True)
    df = df[df['Intra_ID_Time_Gap'] != -1.0].reset_index(drop=True)
    df['Intra_ID_Time_Gap_Norm'] = df['Intra_ID_Time_Gap'].apply(IntraIDTimeGapNorm)
    df['Time_Delta_Norm'] = df['Intra_ID_Time_Gap'].apply(TimeDeltaTimeGapNorm)

    return df

def normalize_can_id_by_frequency(df, column_name='CAN_ID'):
    """
    Engineers a new feature by categorizing CAN IDs based on their frequency.

    This function calculates how often each unique CAN ID appears, assigns it to a
    category based on that frequency, and then adds a new column to the DataFrame
    reflecting this category.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain the specified CAN ID column.
        column_name (str): The name of the column containing the CAN IDs.

    Returns:
        pd.DataFrame: The original DataFrame with a new '{column_name}_Normalized' column.
    """
    # --- Step 1: Calculate Frequencies and Create Category Map ---

    # 1a. Calculate the frequency of each unique CAN ID.
    id_counts = df[column_name].value_counts().reset_index()
    id_counts.columns = [column_name, 'Count']

    # 1b. Define the categorization logic with non-overlapping bins.
    def assign_category(count):
        if count >= 12000:
            return 1
        elif 5000 <= count < 12000:
            return 2
        elif 2500 <= count < 5000:
            return 3
        else:  # For counts < 2500
            return 4

    # 1c. Apply the logic to create a 'Category' column.
    id_counts['Category'] = id_counts['Count'].apply(assign_category)

    # 1d. Create the final mapping dictionary: {CAN_ID: Category}.
    id_to_category_map = pd.Series(id_counts.Category.values, index=id_counts[column_name]).to_dict()


    # --- Step 2: Apply the Map to the Original DataFrame ---

    # 2a. Use the map to create the new normalized feature column.
    normalized_column_name = f'{column_name}_Norm'
    df[normalized_column_name] = df[column_name].map(id_to_category_map)

    return df

def IntraIDTimeGapNorm(value):
    """
    Normalizes the 'Intra_ID_Time_Gap' value based on predefined ranges.

    Args:
        value (float): The time gap value to normalize.

    Returns:
        int: The normalized category (0-6).
    """
    if 0 <= value <= 5.1:
        # Range for category 0
        return 0
    elif 5.1 < value <= 10.1:
        # Range for category 1
        return 1
    elif 10.1 < value <= 20.1:
        # Range for category 2
        return 2
    elif 20.1 < value <= 30.1:
        # Range for category 3
        return 3
    elif 30.1 < value <= 40.1:
        # Range for category 4
        return 4
    elif 40.1 < value <= 50.1:
        # Range for category 5
        return 5
    else:
        # This covers your rule for "50.2 to max" and anything else above 50.1
        return 6

def TimeDeltaTimeGapNorm(value):
    """
    Normalizes the 'Intra_ID_Time_Gap' value based on predefined ranges.

    Args:
        value (float): The time gap value to normalize.

    Returns:
        int: The normalized category (0-6).
    """
    if 0 <= value <= 0.05:
        # Range for category 0
        return 0
    elif 0.05 < value <=0.1:
        # Range for category 1
        return 1
    elif 0.1 < value <= 0.2:
        # Range for category 2
        return 2
    elif 0.2 < value <= 0.3:
        # Range for category 3
        return 3
    elif 0.3 < value <= 0.4:
        # Range for category 4
        return 4
    elif 0.4 < value <= 0.5:
        # Range for category 5
        return 5
    else:
        # This covers your rule for "50.2 to max" and anything else above 50.1
        return 6

def PlotCANIDFrequency(df, column_name='CAN_ID', Data_Name=""):
    """
    Generates and displays a bar plot of value counts for a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        Data_Name (str): The name of the dataset to be included in the title.
    """
    # --- Set a larger figure size for better readability ---
    plt.figure(figsize=(12, 7))

    # --- Generate and display the bar plot ---
    # 1. Select the specified column.
    # 2. Use value_counts() to count occurrences of each ID.
    # 3. Use .plot(kind='bar') to create a bar plot from the counts.
    df[column_name].value_counts().plot(kind='bar')

    # --- Add labels and title for clarity ---
    plt.title(f'Frequency of {column_name}s for {Data_Name}', fontsize=16, fontweight='bold')
    plt.xlabel(column_name, fontsize=12, fontweight='bold')
    plt.ylabel('Count', fontsize=12, fontweight='bold')
    # Rotate x-axis labels vertically (90 degrees) to prevent overlap
    plt.xticks(rotation=90)
    plt.tight_layout() # Adjust plot to ensure everything fits

    # --- Display the plot ---
    plt.show()


# # --- Call the function to generate the plot ---
# PlotCANIDFrequency(Kia_AF, Data_Name="Kia")