import pandas as pd
import numpy as np
import random

# ==============================================================================
# 1. CAN FD Configuration
# ==============================================================================

# Defines columns for the full 64-byte CAN FD standard
DATA_PAYLOAD_COLUMNS = [f'Data{i+1}' for i in range(64)]
COLS_FD = ['Time_Offset', 'CAN_ID', 'Data_Length', *DATA_PAYLOAD_COLUMNS, 'Label']
VALID_FD_PAYLOAD_SIZES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]

# ==============================================================================
# 2. Attack Generation Functions (for Fuzz/DoS)
# ==============================================================================

def fuzzy_df_gen_fd(existing_ids):
    """Generates a fuzzy attack message as a dictionary."""
    if not existing_ids:
        raise ValueError("Cannot generate a Fuzz attack with no existing CAN IDs.")
    new_row = {}
    new_row["CAN_ID"] = random.choice(existing_ids)
    payload_size = random.choice(VALID_FD_PAYLOAD_SIZES)
    new_row["Data_Length"] = payload_size
    for i in range(payload_size):
        new_row[f'Data{i+1}'] = f'{random.randrange(0, 256):02X}'
    for i in range(payload_size, 64):
        new_row[f'Data{i+1}'] = np.nan
    return new_row

def dos_df_gen_fd():
    """Generates a DoS attack message as a dictionary."""
    new_row = {"CAN_ID": "0000", "Data_Length": 64}
    for i in range(64):
        new_row[f'Data{i+1}'] = f'{i % 256:02X}'
    return new_row

# ==============================================================================
# 3. Core Injection Logic
# ==============================================================================

def inject_attack_dataset(orig_df, attack_gen_func, group_size, label):
    """Injects generated Fuzz/DoS attacks with a RANDOMIZED count per group."""
    orig_len = len(orig_df)
    num_groups = orig_len // group_size
    if num_groups == 0:
        print(f"Warning: Dataset with {orig_len} rows is too small for group_size={group_size}. No attacks injected.")
        return orig_df

    existing_ids = []
    if label == "Fuzz":
        existing_ids = [cid for cid in orig_df['CAN_ID'].unique() if isinstance(cid, str) and len(cid) > 0]
        if not existing_ids:
             print("Warning: No valid CAN IDs found for Fuzz attack. No attacks injected.")
             return orig_df

    attack_dicts = []
    for group_idx in range(num_groups):
        attack_count = random.randint(1, 8)
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_positions = random.sample(range(group_start, group_end), attack_count)

        for pos in group_positions:
            attack_dict = attack_gen_func(existing_ids) if label == "Fuzz" else attack_gen_func()
            
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            attack_time = random.uniform(time_before, time_after) if time_after > time_before else time_before + random.uniform(0.0001, 0.001)

            attack_dict["Time_Offset"] = attack_time
            attack_dict["Label"] = label
            attack_dicts.append(attack_dict)

    if not attack_dicts:
        return orig_df

    attack_df = pd.DataFrame(attack_dicts)
    merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
    merged_df.sort_values(by='Time_Offset', inplace=True, ignore_index=True)
    return merged_df

# ADDED: Function to handle Replay attacks specifically.
def inject_sequential_from_df(orig_df, injection_df, group_size, label):
    """Injects messages from a separate DataFrame for Replay attacks."""
    orig_len = len(orig_df)
    injection_len = len(injection_df)
    num_groups = orig_len // group_size
    if num_groups == 0:
        print(f"Warning: Dataset with {orig_len} rows is too small for group_size={group_size}. No attacks injected.")
        return orig_df

    attack_rows = []
    injection_index = 0
    for group_idx in range(num_groups):
        attack_count = random.randint(1, 8)
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_positions = random.sample(range(group_start, group_end), attack_count)

        for pos in group_positions:
            attack_row = injection_df.iloc[[injection_index % injection_len]].copy()
            injection_index += 1
            
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            attack_time = random.uniform(time_before, time_after) if time_after > time_before else time_before + random.uniform(0.0001, 0.001)

            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_rows.append(attack_row)

    if not attack_rows:
        return orig_df

    attack_df = pd.concat(attack_rows, ignore_index=True)
    merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
    merged_df.sort_values(by='Time_Offset', inplace=True, ignore_index=True)
    return merged_df

# ==============================================================================
# 4. Main Wrapper Function
# ==============================================================================

# UPDATED: The wrapper now accepts an injection_df for Replay attacks.
def CANFDDataInjection(type_of_attack, orig_df, injection_df=None, group_size=30, benign_label='Normal'):
    """Main function to inject all types of CAN FD attacks."""
    print(f"\nInjecting '{type_of_attack}' messages with 1 to 8 attacks per group of {group_size}...")
    
    clean_df = orig_df[orig_df['Label'] == benign_label].copy()
    
    if clean_df.empty:
        print(f"Error: No rows found with the label '{benign_label}'. Cannot inject attacks.")
        return pd.DataFrame()

    if type_of_attack == "Fuzz":
        return inject_attack_dataset(clean_df, fuzzy_df_gen_fd, group_size, "Fuzz")
    
    elif type_of_attack == "DoS":
        return inject_attack_dataset(clean_df, dos_df_gen_fd, group_size, "DoS")
    
    # ADDED: Logic to handle Replay attacks.
    elif type_of_attack == "Replay":
        if injection_df is None or injection_df.empty:
            raise ValueError("An 'injection_df' is required for Replay attacks.")
        return inject_sequential_from_df(clean_df, injection_df, group_size, "Replay")
        
    else:
        # UPDATED: Error message includes 'Replay'.
        raise ValueError(f"Unknown attack type: {type_of_attack}. Use 'Fuzz', 'DoS', or 'Replay'.")


# fuzz_df = CANFDDataInjection(
#     type_of_attack="Fuzz",
#     orig_df=original_dataframe
# )

#  # 2. Call the injection function with the replay data.
# replay_df = CANFDDataInjection(
#     type_of_attack="Replay",
#     orig_df=original_dataframe,
#     injection_df=replay_source_messages
# )