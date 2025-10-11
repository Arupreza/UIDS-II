import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# --- Column Definition for CAN FD ---
# Defines the structure for a CAN FD frame, which can hold up to 64 data bytes.
cols = ["Time_Offset", "CAN_ID", "Data_Length"] # Simplified core columns for injection
for i in range(1, 65):
    cols.append("Data" + str(i))

# --- Attack Generation Functions ---

def fuzzy_df_gen(orig_df):
    """
    Generates a single CAN FD frame for a Fuzz attack.
    It uses a random CAN ID, a random data length, and random data bytes.
    
    Args:
        orig_df (pd.DataFrame): The original DataFrame, used to copy structure.

    Returns:
        pd.DataFrame: A single-row DataFrame representing the attack message.
    """
    # Copy the structure from the original dataframe to ensure all columns exist
    data_df = orig_df.iloc[[0]].copy()
    for col in data_df.columns:
        data_df[col] = np.nan # Clear the copied data

    # 1. Generate a random CAN ID
    temp_id = str(hex(random.randrange(0, 2048))[2:]).upper().zfill(4)
    data_df["CAN_ID"] = temp_id # <-- CORRECTED LINE

    # 2. Generate a random data length (up to 64 bytes for CAN FD)
    data_length = 32 # Example fixed length for DoS
    data_df["Data_Length"] = data_length
    
    # 3. Fill data bytes with random hex values
    for i in range(data_length):
        col_name = "Data" + str(i + 1)
        data_field = random.randrange(0, 256)
        temp_data = str(hex(data_field)[2:]).upper().zfill(2)
        data_df[col_name] = temp_data
        
    data_df["Label"] = "Fuzzy"
    return data_df

def dos_df_gen(orig_df):
    """
    Generates a single CAN FD frame for a DoS attack.
    It uses a fixed CAN ID ('0000'), a fixed data length, and fixed data bytes.
    
    Args:
        orig_df (pd.DataFrame): The original DataFrame, used to copy structure.

    Returns:
        pd.DataFrame: A single-row DataFrame representing the attack message.
    """
    # Copy the structure from the original dataframe
    data_df = orig_df.iloc[[0]].copy()
    for col in data_df.columns:
        data_df[col] = np.nan # Clear the copied data
        
    # 1. Use a fixed CAN ID for DoS attack
    data_df["CAN_ID"] = "0000" # <-- CORRECTED LINE
    
    # 2. Set a fixed data length
    data_length = 32 # Example fixed length for DoS
    data_df["Data_Length"] = data_length
    
    # 3. Fill data bytes with a fixed pattern
    for i in range(data_length):
        col_name = "Data" + str(i + 1)
        data_field = i
        temp_data = str(hex(data_field)[2:]).upper().zfill(2)
        data_df[col_name] = temp_data
        
    data_df["Label"] = "DoS"
    return data_df

# --- Injection Logic Functions ---

def inject_attack_dataset(orig_df, attack_gen_func, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    """
    Injects generated attack messages into a DataFrame at random time intervals.
    """
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_messages = []
    current_time = start_time

    min_gap, max_gap = time_gap_range
    
    with tqdm(total=int(end_time), desc=f"Injecting '{label}' messages") as pbar:
        last_update_time = 0
        while current_time < end_time:
            # Generate one attack message, passing orig_df for structure
            attack_row = attack_gen_func(orig_df).copy()
            attack_row["Time_Offset"] = current_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)

            # Update time
            time_gap = random.uniform(min_gap, max_gap)
            current_time += time_gap

            # Update progress bar
            pbar.update(int(current_time - last_update_time))
            last_update_time = current_time
            
    print(f"\nDone – injected {len(attack_messages)} {label} messages.")

    # Merge and sort
    attack_df = pd.concat(attack_messages, ignore_index=True)
    merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
    merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
    df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)

    return df_sort_time

def inject_sequential_from_df(orig_df, injection_df, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    """
    Injects messages sequentially from another DataFrame (for Replay attacks).
    """
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_rows = []
    current_time = start_time

    min_gap, max_gap = time_gap_range
    injection_index = 0
    injection_len = len(injection_df)

    with tqdm(total=int(end_time), desc=f"Injecting '{label}' messages") as pbar:
        last_update_time = 0
        while current_time < end_time:
            # Get next injection row in order (loop if needed)
            attack_row = injection_df.iloc[[injection_index]].copy()
            attack_row["Time_Offset"] = current_time
            attack_row["Label"] = label
            attack_rows.append(attack_row)

            # Update index and time
            injection_index = (injection_index + 1) % injection_len
            time_gap = random.uniform(min_gap, max_gap)
            current_time += time_gap

            # Update progress bar
            pbar.update(int(current_time - last_update_time))
            last_update_time = current_time

    print(f"\nDone – injected {len(attack_rows)} sequential messages.")

    # Merge and sort
    attack_df = pd.concat(attack_rows, ignore_index=True)
    merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
    merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
    df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)

    return df_sort_time

# --- Main Wrapper Function ---

def CANFDDataInjection(type_of_attack, orig_df, injection_df=None, time_gap_range=(10, 50), label="Attack"):
    """
    Main function to select and execute the desired attack injection.

    Args:
        type_of_attack (str): The type of attack ("Fuzz", "DoS", "Replay").
        orig_df (pd.DataFrame): The original CAN FD DataFrame.
        injection_df (pd.DataFrame, optional): DataFrame for Replay attacks. Defaults to None.
        time_gap_range (tuple, optional): Min and max time between attacks. Defaults to (10, 50).
        label (str, optional): The label to assign to attack messages. Defaults to "Attack".

    Returns:
        pd.DataFrame: The original DataFrame with the injected attack messages.
    """
    if type_of_attack == "Fuzz":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=fuzzy_df_gen,
            time_gap_range=time_gap_range,
            label=label
        )
    
    elif type_of_attack == "DoS":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=dos_df_gen,
            time_gap_range=time_gap_range,
            label=label
        )
    
    elif type_of_attack == "Replay":
        if injection_df is None:
            raise ValueError("An 'injection_df' is required for a Replay attack.")
        return inject_sequential_from_df(
            orig_df=orig_df,
            injection_df=injection_df,
            time_gap_range=time_gap_range,
            label=label
        )
    else:
        raise ValueError(f"Unknown attack type: '{type_of_attack}'. Please use 'Fuzz', 'DoS', or 'Replay'.")


# # --- DoS Attack Injection Example ---
# print("--- Starting DoS Attack Injection ---")
# df_with_dos = CANFDDataInjection(
#     type_of_attack="DoS",
#     orig_df=simulation_df,
#     time_gap_range=(0.001, 0.005), # Injecting at a very high frequency for DoS
#     label="DoS"
# )
# print("DoS attack injection complete.")
# print(df_with_dos.tail())
# print(df_with_dos['Label'].value_counts())


# # --- Fuzz Attack Injection Example ---
# print("\n--- Starting Fuzz Attack Injection ---")
# df_with_fuzz = CANFDDataInjection(
#     type_of_attack="Fuzz",
#     orig_df=simulation_df,
#     time_gap_range=(10, 20),
#     label="Fuzz"
# )
# print("Fuzz attack injection complete.")
# print(df_with_fuzz.tail())
# print(df_with_fuzz['Label'].value_counts())


# # --- Replay Attack Injection Example ---
# # First, create a sample DataFrame to replay (e.g., capture some specific traffic)
# replay_attack_df = simulation_df[simulation_df['ID'] == '04B0'].head(5)

# print("\n--- Starting Replay Attack Injection ---")
# df_with_replay = CANFDDataInjection(
#     type_of_attack="Replay",
#     orig_df=simulation_df,
#     injection_df=replay_attack_df,
#     time_gap_range=(50, 100),
#     label="Replay"
# )
# print("Replay attack injection complete.")
# print(df_with_replay.tail())
# print(df_with_replay['Label'].value_counts())
