import pandas as pd
import numpy as np
import random

# ==============================================================================
# 1. Classical CAN Configuration
# ==============================================================================
CAN_DATA_COLUMNS = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
COLS_CAN = ['Time_Offset', 'CAN_ID', 'Data_Length'] + CAN_DATA_COLUMNS + ['Label']

# ==============================================================================
# 2. Fuzz Attack Generation Function
# ==============================================================================

def fuzzy_df_gen_can(existing_ids):
    """
    Generates a single fuzzy attack message as a dictionary.
    """
    if not existing_ids:
        raise ValueError("Cannot generate a Fuzz attack with no existing CAN IDs.")
    
    new_row = {}
    new_row["CAN_ID"] = random.choice(existing_ids)
    data_l = random.randrange(0, 9)
    new_row["Data_Length"] = data_l

    # --- FIX: Use the predefined CAN_DATA_COLUMNS list for consistent naming ---
    for i in range(data_l):
        # This now uses 'One', 'Two', etc. as column names
        new_row[CAN_DATA_COLUMNS[i]] = f'{random.randrange(0, 256):02X}'
    
    for i in range(data_l, 8):
        # This also uses the correct column names for placeholders
        new_row[CAN_DATA_COLUMNS[i]] = np.nan
    # --- END FIX ---
        
    return new_row

# ==============================================================================
# 3. Core Injection Logic
# ==============================================================================

def inject_attack_dataset(orig_df, attack_gen_func, attack_count, group_size, label):
    """
    Injects a fixed number of attacks per group into the dataframe.
    """
    orig_len = len(orig_df)
    num_groups = orig_len // group_size
    if num_groups == 0:
        print(f"Warning: Dataset with {orig_len} rows is too small for group_size={group_size}. No attacks injected.")
        return orig_df

    existing_ids = [cid for cid in orig_df['CAN_ID'].unique() if isinstance(cid, str) and len(cid) > 0]
    if not existing_ids:
       print("Warning: No valid CAN IDs found to target. No attacks injected.")
       return orig_df

    attack_dicts = []
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        
        actual_attack_count = min(attack_count, group_size)
        group_positions = random.sample(range(group_start, group_end), actual_attack_count)

        for pos in group_positions:
            attack_dict = attack_gen_func(existing_ids)
            
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

# ==============================================================================
# 4. Main Wrapper Function
# ==============================================================================

# FIX: Renamed function to follow standard Python conventions (snake_case)
def InjectFuzzAattacks(orig_df, injection_frequency=(5, 30), benign_label='Normal'):
    """
    Main function to inject Fuzz attacks with a user-defined fixed frequency.
    """
    attack_count, group_size = injection_frequency
    print(f"\nInjecting 'Fuzz' messages with a FIXED frequency: {attack_count} attacks per group of {group_size}...")
    
    clean_df = orig_df[orig_df['Label'] == benign_label].copy()
    
    if clean_df.empty:
        print(f"Error: No rows found with the label '{benign_label}'. Cannot inject attacks.")
        return pd.DataFrame()

    return inject_attack_dataset(
        orig_df=clean_df,
        attack_gen_func=fuzzy_df_gen_can,
        attack_count=attack_count,
        group_size=group_size,
        label="Fuzz"
    )

# fuzz_attack_df = InjectFuzzAattacks(
#     orig_df=benign_df,
#     injection_frequency=attack_frequency
# )