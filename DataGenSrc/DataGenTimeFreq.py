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
def fuzzy_df_gen(orig_df):  # Accept orig_df as an argument
    data_df = orig_df[1:2].copy()  # Use passed orig_df for attack generation
    
    # Generate a random CAN_ID
    temp = str(hex(random.randrange(0, 2048))[2:]).upper()
    for _ in range(4 - len(temp)):
        temp = '0' + temp    
    data_df["CAN_ID"] = temp

    data_l = random.randrange(1, 8)  # Random data length
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
def dos_df_gen(orig_df):  # Accept orig_df as an argument
    data_df = orig_df[1:2].copy()  # Use passed orig_df for attack generation
    
    data_df["CAN_ID"] = "0000"  # Fixed CAN_ID for DoS attack
    data_df["Data_Length"] = 8  # Fixed data length for DoS
    
    for i in range(8):  # Fixed length
        j = i + 5
        data_field = i
        temp = str(hex(data_field)[2:]).upper()
        for _ in range(2 - len(temp)):
            temp = '0' + temp
        data_df[cols[j]] = temp
    data_df["Label"] = "DoS"
    return data_df

# General attack injection function
def inject_attack_dataset(orig_df, attack_gen_func, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_messages = []
    current_time = start_time

    min_gap, max_gap = time_gap_range
    last_pct = -1

    while current_time < end_time:
        attack_row = attack_gen_func(orig_df).copy()  # Pass orig_df to attack generation function
        attack_row["Time_Offset"] = current_time
        attack_row["Label"] = label
        attack_messages.append(attack_row)

        # Update time
        time_gap = random.uniform(min_gap, max_gap)
        current_time += time_gap

        # Progress printing
        pct = int(current_time / end_time * 100)
        if pct != last_pct:
            print(f"\rInjecting '{label}' messages… {pct:3d}% complete", end="")
            last_pct = pct

    print(f"\nDone – injected {len(attack_messages)} {label} messages.")

    # Merge and sort
    attack_df = pd.concat(attack_messages)
    merged_df = pd.concat([orig_df, attack_df], axis=0)
    merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
    df_sort_time = merged_df.sort_values(by='Time_Offset')

    return df_sort_time[1:]

# Sequential injection from another DataFrame
def inject_sequential_from_df(orig_df, injection_df, time_gap_range=(10, 50), label="Attack", start_time=0.0):
    end_time = float(orig_df.iloc[-1]["Time_Offset"])
    attack_rows = []
    current_time = start_time

    min_gap, max_gap = time_gap_range
    last_pct = -1

    injection_index = 0
    total_injections = 0
    injection_len = len(injection_df)

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
        total_injections += 1

        # Progress
        pct = int(current_time / end_time * 100)
        if pct != last_pct:
            print(f"\rSequential injection… {pct:3d}% complete", end="")
            last_pct = pct

    print(f"\nDone – injected {total_injections} sequential messages.")

    # Merge and sort
    attack_df = pd.concat(attack_rows, ignore_index=True)
    merged_df = pd.concat([orig_df, attack_df], axis=0)
    merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
    df_sort_time = merged_df.sort_values(by='Time_Offset')

    return df_sort_time

# Main function to select attack type
def DataInjection_TimeFreq(type_of_attack, orig_df, injection_df=None, time_gap_range=(10, 50), label="Attack"):
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

# # Fuzz Attack Injection
# df_with_fuzz = DataInjection(
#     type_of_attack="Fuzz",        # Specify the attack type as "Fuzz"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="Fuzz"                  # Attack label
# )

# # DoS Attack Injection
# df_with_dos = DataInjection(
#     type_of_attack="DoS",         # Specify the attack type as "DoS"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="DoS"                   # Attack label
# )

# # Replay Attack Injection
# # For Replay, you need to provide the 'injection_df' which is the attack data to replay
# # Assuming 'replay_attack_df' is the DataFrame with previous attack data you want to replay
# df_with_replay = DataInjection(
#     type_of_attack="Replay",      # Specify the attack type as "Replay"
#     orig_df=simulation_df,        # The original DataFrame you are working with
#     injection_df=replay_attack_df,  # Injection DataFrame (must be provided for Replay)
#     time_gap_range=(5, 15),       # Specify time gap range for attack injection
#     label="Replay"                # Attack label
# )
