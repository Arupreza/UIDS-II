import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# Column names
cols = ['Time_Offset', 'CAN_ID', 'Data_Length', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Label']

# Generate a hex value with padding
def generate_hex(value, length=4):
    temp = str(hex(value)[2:]).upper()
    return temp.zfill(length)

# Fuzz attack dataset generation
def fuzzy_df_gen(orig_df):
    data_df = orig_df[1:2].copy()
    
    # Generate a random CAN_ID
    temp = str(hex(random.randrange(0, 2048))[2:]).upper()
    for _ in range(4 - len(temp)):
        temp = '0' + temp    
    data_df["CAN_ID"] = temp

    data_l = random.randrange(1, 8)
    data_df["Data_Length"] = data_l
    for i in range(data_l):
        j = i + 5
        data_field = random.randrange(0, 255)
        temp = str(hex(data_field)[2:]).upper()
        for _ in range(2 - len(temp)):
            temp = '0' + temp
        data_df[cols[j]] = temp
    data_df["Label"] = "Fuzzy"
    return data_df

# DoS attack dataset generation
def dos_df_gen(orig_df):
    data_df = orig_df[1:2].copy()
    
    data_df["CAN_ID"] = "0000"
    data_df["Data_Length"] = 8
    
    # Data columns start from index 3 (One, Two, Three, etc.)
    data_columns = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']
    
    for i in range(8):
        data_field = i
        temp = str(hex(data_field)[2:]).upper()
        temp = temp.zfill(2)  # Pad to 2 characters
        data_df[data_columns[i]] = temp
    
    data_df["Label"] = "DoS"
    return data_df

# Frequency-based attack injection function
def inject_attack_dataset(orig_df, attack_gen_func, injection_frequency=(5, 30), label="Attack"):
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    
    # Calculate total number of complete groups
    num_groups = orig_len // group_size
    total_attacks = num_groups * attack_count
    
    attack_messages = []
    last_pct = -1
    
    # Process each group of messages
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, orig_len)
        
        # Generate random positions within this group for attacks
        group_positions = random.sample(range(group_start, group_end), 
                                    min(attack_count, group_end - group_start))
        
        for pos in group_positions:
            attack_row = attack_gen_func(orig_df).copy()
            
            # Interpolate timestamp between adjacent original messages
            if pos < orig_len - 1:
                time_before = float(orig_df.iloc[pos]["Time_Offset"])
                time_after = float(orig_df.iloc[pos + 1]["Time_Offset"])
                attack_time = time_before + random.uniform(0, time_after - time_before)
            else:
                attack_time = float(orig_df.iloc[pos]["Time_Offset"]) + random.uniform(0.001, 0.01)
            
            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)
        
        # Progress
        pct = int((group_idx + 1) / num_groups * 100)
        if pct != last_pct:
            print(f"\rInjecting '{label}' messages… {pct:3d}% complete", end="")
            last_pct = pct

    print(f"\nDone – injected {len(attack_messages)} {label} messages in {num_groups} groups.")

    # Merge and sort
    if attack_messages:
        attack_df = pd.concat(attack_messages)
        merged_df = pd.concat([orig_df, attack_df], axis=0)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset')
        return df_sort_time.reset_index(drop=True)
    else:
        return orig_df

# Sequential injection from another DataFrame with frequency control
def inject_sequential_from_df(orig_df, injection_df, injection_frequency=(5, 30), label="Attack"):
    attack_count, group_size = injection_frequency
    orig_len = len(orig_df)
    injection_len = len(injection_df)
    
    # Calculate total number of complete groups
    num_groups = orig_len // group_size
    total_attacks = num_groups * attack_count
    
    attack_rows = []
    injection_index = 0
    last_pct = -1
    
    # Process each group of messages
    for group_idx in range(num_groups):
        group_start = group_idx * group_size
        group_end = min(group_start + group_size, orig_len)
        
        # Generate random positions within this group for attacks
        group_positions = random.sample(range(group_start, group_end), 
                                    min(attack_count, group_end - group_start))
        
        for pos in group_positions:
            # Get next injection row in order (loop if needed)
            attack_row = injection_df.iloc[[injection_index % injection_len]].copy()
            
            # Interpolate timestamp
            if pos < orig_len - 1:
                time_before = float(orig_df.iloc[pos]["Time_Offset"])
                time_after = float(orig_df.iloc[pos + 1]["Time_Offset"])
                attack_time = time_before + random.uniform(0, time_after - time_before)
            else:
                attack_time = float(orig_df.iloc[pos]["Time_Offset"]) + random.uniform(0.001, 0.01)
            
            attack_row["Time_Offset"] = attack_time
            attack_row["Label"] = label
            attack_rows.append(attack_row)
            
            injection_index += 1
        
        # Progress
        pct = int((group_idx + 1) / num_groups * 100)
        if pct != last_pct:
            print(f"\rSequential injection… {pct:3d}% complete", end="")
            last_pct = pct

    print(f"\nDone – injected {len(attack_rows)} sequential messages in {num_groups} groups.")

    # Merge and sort
    if attack_rows:
        attack_df = pd.concat(attack_rows, ignore_index=True)
        merged_df = pd.concat([orig_df, attack_df], axis=0)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset')
        return df_sort_time.reset_index(drop=True)
    else:
        return orig_df

# Main function to select attack type
def CANDataInjectionCountFreq(type_of_attack, orig_df, injection_df=None, injection_frequency=(5, 30), label="Attack"):
    """
    Inject attacks based on message frequency rather than time intervals.
    
    Args:
        type_of_attack: "Fuzz", "DoS", or "Replay"
        orig_df: Original DataFrame
        injection_df: DataFrame for Replay attacks (required for Replay)
        injection_frequency: Tuple (attack_count, total_messages) 
                        e.g., (5, 30) means 5 attacks per 30 messages
        label: Label for attack messages
    """
    if type_of_attack == "Fuzz":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=fuzzy_df_gen,
            injection_frequency=injection_frequency,
            label=label
        )
    
    elif type_of_attack == "DoS":
        return inject_attack_dataset(
            orig_df=orig_df,
            attack_gen_func=dos_df_gen,
            injection_frequency=injection_frequency,
            label=label
        )
    
    elif type_of_attack == "Replay":
        if injection_df is None:
            raise ValueError("Injection DataFrame is required for Replay attack.")
        return inject_sequential_from_df(
            orig_df=orig_df,
            injection_df=injection_df,
            injection_frequency=injection_frequency,
            label="Replay"
        )
    else:
        raise ValueError(f"Unknown attack type: {type_of_attack}")

# dos_result = CANDataInjectionCountFreq(
#     type_of_attack="DoS",
#     orig_df=simulation_df,
#     injection_frequency=(5, 30),  # 5 DoS attacks per 30 messages
#     label="DoS"
# )

# fuzz_result = CANDataInjectionCountFreq(
#     type_of_attack="Fuzz",
#     orig_df=simulation_df,
#     injection_frequency=(5, 30),  # 5 fuzz attacks per 30 messages
#     label="Fuzz"
# )

# replay_result = CANDataInjectionCountFreq(
#     type_of_attack="Replay",
#     orig_df=simulation_df,
#     injection_df=simulation_df,  # This is required for Replay
#     injection_frequency=(6, 30),  # 5 replay attacks per 30 messages
#     label="Replay"
# )