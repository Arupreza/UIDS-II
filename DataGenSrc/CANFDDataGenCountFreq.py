import pandas as pd
import numpy as np
import random

# --- 1. CAN FD Configuration ---

# Define the full column structure for a CAN FD frame with up to 64 data bytes.
# This structure is based on your provided CAN FD data sample.
DATA_COLUMN_NAMES = [f'Data{i}' for i in range(1, 65)]
COLS_FD = ['Time_Offset', 'CAN_ID', 'Data_Length'] + DATA_COLUMN_NAMES + ['Time_Gap', 'Label']

# Define the valid data lengths for CAN FD frames.
# Payloads are not linear from 0-64 bytes.
VALID_FD_DLC_SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]

# --- 2. Attack Generation Functions (for CAN FD) ---

def fuzzy_df_gen_fd(orig_df):
    """
    Generates a single fuzzy attack message row formatted for CAN FD.
    - Random CAN ID
    - Random valid CAN FD data length
    - Random data bytes up to the chosen length
    """
    # Initialize a new row by copying the structure from the original dataframe
    data_df = orig_df.iloc[[0]].copy()

    # Generate a random CAN_ID (11-bit ID, max 0x7FF or 2047)
    temp_id = hex(random.randrange(0, 2048))[2:].upper().zfill(4)
    data_df["CAN_ID"] = temp_id

    # Choose a random, valid data length for CAN FD
    data_l = random.choice(VALID_FD_DLC_SIZES)
    data_df["Data_Length"] = data_l

    # Populate data fields with random hex values
    for i in range(data_l):
        col_name = f'Data{i+1}'
        data_field = random.randrange(0, 256)
        data_df[col_name] = hex(data_field)[2:].upper().zfill(2)

    # Fill remaining data fields with a placeholder (-1 as in your sample)
    for i in range(data_l, 64):
        col_name = f'Data{i+1}'
        data_df[col_name] = -1

    data_df["Label"] = "Fuzzy"
    return data_df

def dos_df_gen_fd(orig_df):
    """
    Generates a single DoS attack message row formatted for CAN FD.
    - Fixed CAN ID (e.g., '0000')
    - Maximum data length (64 bytes) to maximize bus load
    - Fixed data payload
    """
    # Initialize a new row
    data_df = orig_df.iloc[[0]].copy()

    # Use a high-priority (low value) ID to dominate the bus
    data_df["CAN_ID"] = "0000"
    # Use the maximum payload size for a more effective DoS attack
    data_df["Data_Length"] = 64

    # Populate all 64 data fields with a simple, repeating pattern
    for i in range(64):
        col_name = f'Data{i+1}'
        data_field = i % 256  # Example pattern
        data_df[col_name] = hex(data_field)[2:].upper().zfill(2)

    data_df["Label"] = "DoS"
    return data_df

# --- 3. Core Injection Logic (Largely Unchanged) ---

def inject_attack_dataset(orig_df, attack_gen_func, injection_frequency=(5, 30), label="Attack"):
    """
    Injects generated attack messages into the original dataframe based on frequency.
    This function is generic and works with any attack generation function.
    """
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    num_groups = orig_len // group_size

    if num_groups == 0:
        print(f"Warning: Original data length ({orig_len}) is smaller than group size ({group_size}). No attacks injected.")
        return orig_df

    attack_messages = []
    print(f"Injecting '{label}' messages...")

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        
        # Select random, unique positions within the current group for injection
        group_positions = random.sample(range(group_start, group_end), attack_count)

        for pos in group_positions:
            attack_row = attack_gen_func(orig_df).copy()
            
            # Interpolate the timestamp to place the attack message between two normal messages
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            # Ensure we don't go out of bounds
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            
            if time_after > time_before:
                attack_time = random.uniform(time_before, time_after)
            else: # Handle the last message case
                attack_time = time_before + random.uniform(0.0001, 0.001)

            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)

    print(f"Generated {len(attack_messages)} '{label}' messages for injection.")

    # Merge original and attack dataframes
    if attack_messages:
        attack_df = pd.concat(attack_messages, ignore_index=True)
        merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
        
        # Sort by timestamp to correctly sequence all messages
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)
        
        # **Crucially, recalculate Time_Gap for the entire dataset after injection**
        df_sort_time['Time_Gap'] = df_sort_time['Time_Offset'].diff().fillna(0)
        
        return df_sort_time
    else:
        return orig_df

def inject_sequential_from_df(orig_df, injection_df, injection_frequency=(5, 30), label="Attack"):
    """
    Injects messages sequentially from another dataframe (for Replay attacks).
    """
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    injection_len = len(injection_df)
    num_groups = orig_len // group_size
    
    if num_groups == 0:
        print(f"Warning: Original data length ({orig_len}) is smaller than group size ({group_size}). No attacks injected.")
        return orig_df
        
    attack_rows = []
    injection_index = 0
    print(f"Injecting '{label}' messages sequentially...")

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        
        group_positions = random.sample(range(group_start, group_end), attack_count)
        
        for pos in group_positions:
            # Get the next row from the injection source, looping if necessary
            attack_row = injection_df.iloc[[injection_index % injection_len]].copy()
            
            # Interpolate timestamp
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            
            if time_after > time_before:
                attack_time = random.uniform(time_before, time_after)
            else:
                attack_time = time_before + random.uniform(0.0001, 0.001)

            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_rows.append(attack_row)
            injection_index += 1

    print(f"Generated {len(attack_rows)} '{label}' messages for injection.")

    if attack_rows:
        attack_df = pd.concat(attack_rows, ignore_index=True)
        merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)
        
        # Recalculate Time_Gap after injection and sorting
        df_sort_time['Time_Gap'] = df_sort_time['Time_Offset'].diff().fillna(0)
        
        return df_sort_time
    else:
        return orig_df

# --- 4. Main Wrapper Function (for CAN FD) ---

def CANFDDataInjectionCountFreq(type_of_attack, orig_df, injection_df=None, injection_frequency=(5, 30)):
    """
    Main function to inject CAN FD attacks based on message frequency.
    
    Args:
        type_of_attack (str): "Fuzz", "DoS", or "Replay".
        orig_df (pd.DataFrame): The original CAN FD dataframe.
        injection_df (pd.DataFrame, optional): Dataframe for Replay attacks.
        injection_frequency (tuple): (num_attacks, per_num_messages).
    """
    if type_of_attack == "Fuzz":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=fuzzy_df_gen_fd,
            injection_frequency=injection_frequency,
            label="Fuzz"
        )
    
    elif type_of_attack == "DoS":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=dos_df_gen_fd,
            injection_frequency=injection_frequency,
            label="DoS"
        )
    
    elif type_of_attack == "Replay":
        if injection_df is None:
            raise ValueError("An 'injection_df' is required for Replay attacks.")
        return inject_sequential_from_df(
            orig_df=orig_df,
            injection_df=injection_df,
            injection_frequency=injection_frequency,
            label="Replay"
        )
    else:
        raise ValueError(f"Unknown attack type: {type_of_attack}. Please use 'Fuzz', 'DoS', or 'Replay'.")


    # dos_result = CANFDDataInjectionCountFreq(
    #     type_of_attack="DoS",
    #     orig_df=simulation_df,
    #     injection_frequency=(5, 30)  # 5 DoS attacks per 30 messages
    # )
    
    # fuzz_result = CANFDDataInjectionCountFreq(
    #     type_of_attack="Fuzz",
    #     orig_df=simulation_df,
    #     injection_frequency=(5, 30)  # 5 Fuzz attacks per 30 messages
    # )

    # replay_result = CANFDDataInjectionCountFreq(
    #     type_of_attack="Replay",
    #     orig_df=simulation_df,
    #     injection_df=simulation_df,  # For replay, you need a source DataFrame
    #     injection_frequency=(6, 30)  # 6 Replay attacks per 30 messages
    # )