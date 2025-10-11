import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import random

# Column names for standard CAN
cols = ['Time_Offset', 'CAN_ID', 'Data_Length', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Label']

# Generate a hex value with padding
def generate_hex(value, length=4):
    temp = str(hex(value)[2:]).upper()
    return temp.zfill(length)

# Fuzz attack dataset generation
def fuzzy_df_gen(orig_df):
    data_df = orig_df.iloc[[0]].copy()
    for col in data_df.columns:
        data_df[col] = np.nan

    temp = str(hex(random.randrange(0, 2048))[2:]).upper().zfill(4)
    data_df["CAN_ID"] = temp

    data_l = random.randrange(1, 9) # Up to 8 bytes for standard CAN
    data_df["Data_Length"] = data_l
    for i in range(data_l):
        j = i + 3
        data_field = random.randrange(0, 256)
        temp = str(hex(data_field)[2:]).upper().zfill(2)
        data_df[cols[j]] = temp
    data_df["Label"] = "Fuzzy"
    return data_df

# DoS attack dataset generation
def dos_df_gen(orig_df):
    data_df = orig_df.iloc[[0]].copy()
    for col in data_df.columns:
        data_df[col] = np.nan
    
    data_df["CAN_ID"] = "0000"
    data_df["Data_Length"] = 8
    
    for i in range(8):
        j = i + 3
        data_field = i
        temp = str(hex(data_field)[2:]).upper().zfill(2)
        data_df[cols[j]] = temp
    data_df["Label"] = "DoS"
    return data_df

# General attack injection function
def inject_attack_dataset(orig_df, attack_gen_func, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_messages = []
    current_time = start_time
    min_gap, max_gap = time_gap_range

    with tqdm(total=int(end_time), desc=f"Injecting '{label}' messages") as pbar:
        last_update_time = 0
        while current_time < end_time:
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

# Sequential injection from another DataFrame
def inject_sequential_from_df(orig_df, injection_df, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_rows = []
    current_time = start_time
    min_gap, max_gap = time_gap_range
    injection_index = 0
    injection_len = len(injection_df)

    with tqdm(total=int(end_time), desc=f"Injecting '{label}' messages") as pbar:
        last_update_time = 0
        while current_time < end_time:
            # Get next injection row in order
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

# Main function to select attack type
def CANDataInjection(type_of_attack, orig_df, injection_df=None, time_gap_range=(10, 50), label="Attack"):
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
            raise ValueError("Injection DataFrame is required for Replay attack.")
        return inject_sequential_from_df(
            orig_df=orig_df,
            injection_df=injection_df,
            time_gap_range=time_gap_range,
            label="Replay"
        )
    else:
        raise ValueError(f"Unknown attack type: {type_of_attack}")
    



# # Assuming 'simulation_df' is your original DataFrame

# # DoS Attack Injection
# df_with_dos = CANDataInjection(
#     type_of_attack="DoS",         # Specify the attack type as "DoS"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="DoS"                   # Attack label
# )

# # Fuzz Attack Injection
# df_with_fuzz = CANDataInjection(
#     type_of_attack="Fuzz",        # Specify the attack type as "Fuzz"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="Fuzz"                  # Attack label
# )

# # Replay Attack Injection
# # For Replay, you need to provide the 'injection_df' which is the attack data to replay
# # Assuming 'replay_attack_df' is the DataFrame with previous attack data you want to replay
# df_with_replay = CANDataInjection(
#     type_of_attack="Replay",      # Specify the attack type as "Replay"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     injection_df=replay_attack_df,  # Injection DataFrame (must be provided for Replay)
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="Replay"                # Attack label
# )