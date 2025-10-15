import pandas as pd
import numpy as np
import random

# --- 1. Classical CAN Configuration (FIXED COLUMNS) ---

COLS_CAN = ['Time_Offset', 'CAN_ID', 'Data_Length', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Label']
DATA_COLS = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']

# --- 2. Attack Generation Functions (MODIFIED) ---

# MODIFICATION: Returns a dictionary for efficiency and removed unused 'orig_df' parameter.
def fuzzy_df_gen_can(existing_ids):
    """
    (TARGETED) Generates a fuzzy attack message by targeting an EXISTING CAN ID.
    Returns a dictionary, not a DataFrame.
    """
    if not existing_ids:
        raise ValueError("Cannot generate a targeted Fuzz attack with no existing CAN IDs.")

    new_row = {}

    # 1. Choose a legitimate CAN_ID to target
    target_id = random.choice(existing_ids)
    new_row["CAN_ID"] = target_id

    # 2. Generate random data length (0-8)
    data_l = random.randrange(0, 9)
    new_row["Data_Length"] = data_l

    # 3. Populate data bytes with fuzzed (random) hex strings
    for i in range(data_l):
        new_row[DATA_COLS[i]] = f'{random.randrange(0, 256):02X}'
    
    # 4. CRITICAL FIX: Set unused data bytes to np.nan for data type consistency
    for i in range(data_l, 8):
        new_row[DATA_COLS[i]] = np.nan

    new_row["Label"] = "Fuzz" # Default label, can be overwritten
    
    return new_row

# MODIFICATION: Returns a dictionary for efficiency and removed unused 'orig_df' parameter.
def dos_df_gen_can():
    """
    Generates a DoS attack message for classical CAN.
    Returns a dictionary, not a DataFrame.
    """
    new_row = {}
    new_row["CAN_ID"] = "0000"
    new_row["Data_Length"] = 8
    
    for i in range(8):
        new_row[DATA_COLS[i]] = f'{i % 256:02X}'
        
    new_row["Label"] = "DoS" # Default label, can be overwritten

    return new_row

# --- 3. Core Injection Logic (MODIFIED FOR PERFORMANCE) ---

# MODIFICATION: Collects dictionaries and creates one DataFrame at the end.
def inject_attack_dataset(orig_df, attack_gen_func, group_size=30, label="Attack"):
    """
    Injects a random number of attacks (1-8) per group of a fixed size.
    """
    orig_len = len(orig_df)
    num_groups = orig_len // group_size

    if num_groups == 0:
        print("Warning: Original dataset too small for specified group size. No attacks injected.")
        return orig_df

    existing_ids = []
    if label == "Fuzz":
        existing_ids = [cid for cid in orig_df['CAN_ID'].unique() if isinstance(cid, str) and len(cid) > 0]
        if not existing_ids:
            print("Warning: No valid existing CAN IDs found to target for Fuzz attack. No attacks injected.")
            return orig_df
        print(f"Identified {len(existing_ids)} unique CAN IDs to target for Fuzz attack.")

    attack_dicts = []
    print(f"\nInjecting '{label}' messages...")

    for group_idx in range(num_groups):
        attack_count = random.randint(1, 8)
        group_start = group_idx * group_size
        group_end = group_start + group_size
        actual_attack_count = min(attack_count, group_size)
        group_positions = random.sample(range(group_start, group_end), actual_attack_count)

        for pos in group_positions:
            # Generate the attack row as a dictionary
            if label == "Fuzz":
                attack_dict = attack_gen_func(existing_ids)
            else: 
                attack_dict = attack_gen_func()
            
            # Determine injection time
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after_idx = min(pos + 1, orig_len - 1)
            time_after = float(orig_df.iloc[time_after_idx]["Time_Offset"])
            
            if time_after > time_before:
                attack_time = random.uniform(time_before, time_after)
            else:
                attack_time = time_before + random.uniform(0.000001, 0.00001)

            attack_dict["Time_Offset"] = attack_time
            attack_dict["Label"] = label
            attack_dicts.append(attack_dict)

    print(f"Generated {len(attack_dicts)} '{label}' messages for injection.")

    if attack_dicts:
        # PERFORMANCE FIX: Create a single DataFrame from the list of dicts
        attack_df = pd.DataFrame(attack_dicts)
        merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        
        df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)
        return df_sort_time
    return orig_df

def inject_sequential_from_df(orig_df, injection_df, group_size=30, label="Attack"):
    """
    Injects a random number of sequential attacks (1-8) per group from a given DataFrame.
    """
    orig_len = len(orig_df)
    injection_len = len(injection_df)
    num_groups = orig_len // group_size
    
    if num_groups == 0: 
        print("Warning: Original dataset too small for specified group size. No attacks injected.")
        return orig_df
        
    attack_rows = []
    injection_index = 0
    print(f"\nInjecting '{label}' messages sequentially...")

    for group_idx in range(num_groups):
        attack_count = random.randint(1, 8)
        group_start = group_idx * group_size
        group_end = group_start + group_size
        actual_attack_count = min(attack_count, group_size)
        group_positions = random.sample(range(group_start, group_end), actual_attack_count)
        
        for pos in group_positions:
            attack_row = injection_df.iloc[[injection_index % injection_len]].copy()

            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after_idx = min(pos + 1, orig_len - 1)
            time_after = float(orig_df.iloc[time_after_idx]["Time_Offset"])

            if time_after > time_before:
                attack_time = random.uniform(time_before, time_after)
            else:
                attack_time = time_before + random.uniform(0.000001, 0.00001)

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
        return df_sort_time
    return orig_df

# --- 4. Main Wrapper Function ---

def CANDataInjectionCountFreq(type_of_attack, orig_df, injection_df=None, group_size=30):
    """
    Wrapper function to inject attacks.
    """
    if type_of_attack == "Fuzz":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=fuzzy_df_gen_can,
            group_size=group_size,
            label="Fuzz"
        )
    elif type_of_attack == "DoS":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=dos_df_gen_can,
            group_size=group_size,
            label="DoS"
        )
    elif type_of_attack == "Replay":
        if injection_df is None or injection_df.empty:
            raise ValueError("An 'injection_df' is required for Replay attacks.")
        return inject_sequential_from_df(
            orig_df=orig_df,
            injection_df=injection_df,
            group_size=group_size,
            label="Replay"
        )
    else:
        raise ValueError(f"Unknown attack type: {type_of_attack}. Use 'Fuzz', 'DoS', or 'Replay'.")




# # 2. Inject a Fuzz attack (random 1-8 attacks per 30 messages)
# fuzz_result_df = CANDataInjectionCountFreq(
#     type_of_attack="Fuzz",
#     orig_df=simulation_df.copy(), # Work on a copy
#     group_size=30
# )

# # 3. Inject a DoS attack (random 1-8 attacks per 30 messages)
# dos_result_df = CANDataInjectionCountFreq(
#     type_of_attack="DoS",
#     orig_df=fuzz_result_df.copy(), # Inject into the fuzz-infected log
#     group_size=30
# )

# # 4. Inject a Replay attack
# replay_injection_df = create_dummy_replay_df()
# final_result_df = CANDataInjectionCountFreq(
#     type_of_attack="Replay",
#     orig_df=dos_result_df.copy(), # Inject into the DoS-infected log
#     injection_df=replay_injection_df,
#     group_size=30
# )