#Read CSV File
def process_csv_file(file_path):
    # Read CSV file
    data = pd.read_csv(file_path)
    
    # Remove the last row
    data = data.iloc[:-1]
    
    # Fill missing values with -1
    data = data.fillna(-1)
    
    return data
    
########################################################################################    
#Masking The CAN ID

#Devide CAN ID into three parts
def divide_into_parts(hex_values, num_parts=3):
    """
    Divides a list into 'num_parts' roughly equal parts.

    Parameters:
        hex_values (list): The list of values to be divided.
        num_parts (int): The number of parts to divide the list into.

    Returns:
        list: A list containing 'num_parts' lists, each representing a part.
    """
    part_size = len(hex_values) // num_parts
    remainder = len(hex_values) % num_parts
    parts = []

    start = 0
    for i in range(num_parts):
        # Add an extra item to some of the parts to account for remainder
        end = start + part_size + (1 if i < remainder else 0)
        parts.append(hex_values[start:end])
        start = end

    return parts
 
########################################################################################

#Devided CAN ID convert into Mask form
def categorize_can_ids_X(df, ID_G_1, ID_G_2, ID_G_3):
    """
    Categorizes CAN_IDs in the DataFrame based on the provided groups and adds a 'Category' column.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'CAN_ID' column to categorize.
        can_id_groups (list of lists): A list containing three lists of CAN_IDs corresponding to categories 'A', 'B', and 'C'.

    Returns:
        pd.DataFrame: The DataFrame with an added 'Category' column.
    """

    # Create a dictionary for mapping CAN_IDs to categories
    mapping_dict = {**{can_id: '0.33' for can_id in ID_G_1},
                    **{can_id: '0.66' for can_id in ID_G_2},
                    **{can_id: '1' for can_id in ID_G_3}}

    # Apply the mapping to create a new column 'Category'
    df['Category'] = df['CAN_ID'].map(mapping_dict)

    # Fill NaN values in 'Category' with 'Unknown' if any CAN_IDs don't match the predefined lists
    df['Category'].fillna('Unknown', inplace=True)
    df = df[['Time_Offset', 'CAN_ID', 'Time_Gap', 'Category']]

    return df
    
########################################################################################
#Chunk Compilation
def segment(df, time_gap):
    df = df[["Time_Offset","Time_Gap", "Category"]]
    up = float(df['Time_Offset'][0])
    low = up + time_gap
    chunk = []

        
    for i in df.Time_Offset:

        if  float(i) <= low:
            out = df[(df['Time_Offset'] >= float(up)) & (df['Time_Offset'] <= float(low))]
            out = np.array(out[["Time_Gap", "Category"]])
        chunk.append(out)
    return chunk


#Feature Selectoion

Kia_Korea_Normal = process_csv_file("Kia_Korea_Normal.csv")
Kia_Korea_Normal = Kia_Korea_Normal[:600000]
Kia_Korea_Normal = Kia_Korea_Normal.rename(columns={'Time_Ofset': 'Time_Offset'})
Kia_Korea_Normal.replace(-1, None, inplace=True)
Kia_Korea_Normal = Kia_Korea_Normal.astype(str)


Kia_Korea_DoS = process_csv_file("Kia_Korea_DoS.csv")
Kia_Korea_DoS = Kia_Korea_DoS[:600000]
Kia_Korea_DoS = Kia_Korea_DoS.rename(columns={'Time_Ofset': 'Time_Offset'})
Kia_Korea_DoS.replace(-1, None, inplace=True)
Kia_Korea_DoS = Kia_Korea_DoS.astype(str)


Kia_Korea_Fuzz = process_csv_file("Kia_Korea_Fuzz.csv")
Kia_Korea_Fuzz = Kia_Korea_Fuzz[:600000]
Kia_Korea_Fuzz = Kia_Korea_Fuzz.rename(columns={'Time_Ofset': 'Time_Offset'})
Kia_Korea_Fuzz.replace(-1, None, inplace=True)
Kia_Korea_Fuzz = Kia_Korea_Fuzz.astype(str)


Kia_Korea_Replay = process_csv_file("Kia_Korea_Replay.csv")
Kia_Korea_Replay = Kia_Korea_Replay[:600000]
Kia_Korea_Replay = Kia_Korea_Replay.rename(columns={'Time_Ofset': 'Time_Offset'})
Kia_Korea_Replay.replace(-1, None, inplace=True)
Kia_Korea_Replay = Kia_Korea_Replay.astype(str)


Kia_Korea_Normal = Kia_Korea_Normal[['Time_Offset', 'CAN_ID', 'Time_Gap']]
Kia_Korea_DoS = Kia_Korea_DoS[['Time_Offset', 'CAN_ID', 'Time_Gap']]
Kia_Korea_Fuzz = Kia_Korea_Fuzz[['Time_Offset', 'CAN_ID', 'Time_Gap']]
Kia_Korea_Replay = Kia_Korea_Replay[['Time_Offset', 'CAN_ID', 'Time_Gap']]

########################################################################################
Kia_Korea_Normal_IDs = divide_into_parts(np.unique(Kia_Korea_Normal['CAN_ID']))
Kia_Korea_DoS_IDs = divide_into_parts(np.unique(Kia_Korea_DoS['CAN_ID']))
Kia_Korea_Fuzz_IDs = divide_into_parts(np.unique(Kia_Korea_Fuzz['CAN_ID']))
Kia_Korea_Replay_IDs = divide_into_parts(np.unique(Kia_Korea_Replay['CAN_ID']))

########################################################################################
Kia_Korea_C_CAN_C = categorize_can_ids_X(Kia_Korea_Normal, Kia_Korea_Normal_IDs[0], Kia_Korea_Normal_IDs[1], Kia_Korea_Normal_IDs[2])
Kia_Korea_DoS_C = categorize_can_ids_X(Kia_Korea_DoS, Kia_Korea_DoS_IDs[0], Kia_Korea_DoS_IDs[1], Kia_Korea_DoS_IDs[2])
Kia_Korea_Fuzz_C = categorize_can_ids_X(Kia_Korea_Fuzz, Kia_Korea_Fuzz_IDs[0], Kia_Korea_Fuzz_IDs[1], Kia_Korea_Fuzz_IDs[2])
Kia_Korea_Replay_C = categorize_can_ids_X(Kia_Korea_Replay, Kia_Korea_Replay_IDs[0], Kia_Korea_Replay_IDs[1], Kia_Korea_Replay_IDs[2])