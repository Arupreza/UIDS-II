
import pandas as pd
import numpy as np
import random
import io

# --- 1. Classical CAN Configuration ---

CAN_DATA_COLUMNS = [f'Data{i}' for i in range(1, 9)]
COLS_CAN = ['Time_Offset', 'CAN_ID', 'Data_Length'] + CAN_DATA_COLUMNS + ['Time_Gap', 'Label']

# --- 2. Attack Generation Functions (Targeted Fuzz) ---

def fuzzy_df_gen_can(orig_df, existing_ids):
    """
    (TARGETED) Generates a fuzzy attack message by targeting an EXISTING CAN ID.
    """
    if not existing_ids:
        raise ValueError("Cannot generate a targeted Fuzz attack with no existing CAN IDs.")

    # Build the new row from scratch
    new_row = {col: np.nan for col in orig_df.columns}

    # 1. Choose a legitimate CAN_ID to target from the provided list
    target_id = random.choice(existing_ids)
    new_row["CAN_ID"] = target_id

    # 2. Generate random data length (0-8)
    data_l = random.randrange(0, 9)
    new_row["Data_Length"] = data_l

    # 3. Populate up to 8 data bytes with fuzzed (random) values
    for i in range(data_l):
        new_row[f'Data{i+1}'] = f'{random.randrange(0, 256):02X}'
    
    for i in range(data_l, 8):
        new_row[f'Data{i+1}'] = -1

    new_row["Label"] = "Fuzzy"
    new_row["Time_Offset"] = np.nan
    new_row["Time_Gap"] = np.nan

    return pd.DataFrame([new_row])

def dos_df_gen_can(orig_df):
    """
    (UNCHANGED) Generates a DoS attack message for classical CAN.
    """
    new_row = {col: np.nan for col in orig_df.columns}
    new_row["CAN_ID"] = "0000"
    new_row["Data_Length"] = 8
    
    for i in range(8):
        new_row[f'Data{i+1}'] = f'{i % 256:02X}'
        
    new_row["Label"] = "DoS"
    new_row["Time_Offset"] = np.nan
    new_row["Time_Gap"] = np.nan
    return pd.DataFrame([new_row])

# --- 3. Core Injection Logic (Modified for Targeting) ---

def inject_attack_dataset(orig_df, attack_gen_func, injection_frequency=(5, 30), label="Attack"):
    """
    (MODIFIED) Injects attacks and handles passing existing IDs for Fuzz attacks.
    """
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    num_groups = orig_len // group_size

    if num_groups == 0:
        return orig_df

    # **KEY LOGIC**: Get the list of IDs to target if this is a Fuzz attack
    existing_ids = []
    if label == "Fuzz":
        existing_ids = [cid for cid in orig_df['CAN_ID'].unique() if isinstance(cid, str) and len(cid) > 0]
        if not existing_ids:
            print("Warning: No valid existing CAN IDs found to target for Fuzz attack. No attacks injected.")
            return orig_df
        print(f"Identified {len(existing_ids)} unique CAN IDs to target for Fuzz attack.")

    attack_messages = []
    print(f"Injecting '{label}' messages...")

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_positions = random.sample(range(group_start, group_end), attack_count)

        for pos in group_positions:
            # Pass the list of IDs to the Fuzz generator
            if label == "Fuzz":
                attack_row = attack_gen_func(orig_df, existing_ids).copy()
            else: # For DoS, the original call signature is used
                attack_row = attack_gen_func(orig_df).copy()
            
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            attack_time = random.uniform(time_before, time_after) if time_after > time_before else time_before + random.uniform(0.0001, 0.001)

            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)

    print(f"Generated {len(attack_messages)} '{label}' messages for injection.")

    if attack_messages:
        attack_df = pd.concat(attack_messages, ignore_index=True)
        merged_df = pd.concat([orig_df, attack_df], ignore_index=True)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset').reset_index(drop=True)
        df_sort_time['Time_Gap'] = df_sort_time['Time_Offset'].diff().fillna(0)
        return df_sort_time
    return orig_df

def inject_sequential_from_df(orig_df, injection_df, injection_frequency=(5, 30), label="Attack"):
    """
    (UNCHANGED) This function is generic and does not need modification.
    """
    # This function's logic is correct and remains unchanged.
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    injection_len = len(injection_df)
    num_groups = orig_len // group_size
    
    if num_groups == 0: return orig_df
        
    attack_rows = []
    injection_index = 0
    print(f"Injecting '{label}' messages sequentially...")

    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = group_start + group_size
        group_positions = random.sample(range(group_start, group_end), attack_count)
        
        for pos in group_positions:
            attack_row = injection_df.iloc[[injection_index % injection_len]].copy()
            time_before = float(orig_df.iloc[pos]["Time_Offset"])
            time_after = float(orig_df.iloc[min(pos + 1, orig_len - 1)]["Time_Offset"])
            attack_time = random.uniform(time_before, time_after) if time_after > time_before else time_before + random.uniform(0.0001, 0.001)
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
        df_sort_time['Time_Gap'] = df_sort_time['Time_Offset'].diff().fillna(0)
        return df_sort_time
    return orig_df

# --- 4. Main Wrapper Function ---

def CANDataInjectionCountFreq(type_of_attack, orig_df, injection_df=None, injection_frequency=(5, 30)):
    """
    (MODIFIED) Wrapper function now calls the correct classical CAN generators.
    """
    if type_of_attack == "Fuzz":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=fuzzy_df_gen_can,
            injection_frequency=injection_frequency,
            label="Fuzz"
        )
    elif type_of_attack == "DoS":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=dos_df_gen_can,
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
        raise ValueError(f"Unknown attack type: {type_of_attack}. Use 'Fuzz', 'DoS', or 'Replay'.")



# # Inject a Fuzz attack into the classical CAN data
# fuzz_result_can = CANDataInjectionCountFreq(
#     type_of_attack="Fuzz",
#     orig_df=simulation_df,
#     injection_frequency=(5, 20)
# )
