# AttackInjector.py

import pandas as pd
import numpy as np
import random
from tqdm.auto import tqdm

class FuzzInjecCAN:
    def __init__(self, orig_df, time_gap_range=(10, 50), label="Attack", start_time=0.0, num_attacks=100):
        """
        Initialize the Fuzz attack generator class.

        :param orig_df: The original DataFrame with the simulation data.
        :param time_gap_range: The range of time intervals between injected attack messages (not used here, kept for compatibility).
        :param label: The label for the attack (default is "Attack").
        :param start_time: The start time for the attack message injection.
        :param num_attacks: The number of attack messages to inject (default is 100).
        """
        self.orig_df = orig_df
        self.time_gap_range = time_gap_range  # This is not used anymore, as we're distributing evenly
        self.label = label
        self.start_time = start_time
        self.num_attacks = num_attacks
        self.cols = ["No", "Time_Offset", "Type", "CAN_ID", "Data_Length", 'One', 'Two', 
                    'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']

    def fuzzy_df_gen(self):
        """
        Generate a fuzz attack message based on the original DataFrame.
        """
        data_df = self.orig_df[1:2].copy()  # Take a small slice of the original DataFrame (adjust as needed)

        # Randomize Data Length
        data_l = random.randrange(1, 8)
        data_df["Data_Length"] = data_l
        for i in range(data_l):
            j = i + 5
            data_field = random.randrange(0, 255)
            temp = str(hex(data_field)[2:]).upper()
            for _ in range(2 - len(temp)):
                temp = '0' + temp
            data_df[self.cols[j]] = temp
        data_df["Label"] = "Fuzzy"
        return data_df

    def inject_attack_dataset(self, attack_gen_func, time_gap_range=(10, 50), label="Attack", start_time=0.0):
        """
        Inject attack messages into the original dataset, ensuring equal distribution of attack messages over time.
        """
        end_time = float(self.orig_df.iloc[-1]["Time_Offset"])

        # Calculate total time available for injections
        total_injection_time = end_time - start_time

        # Calculate the equal time gap between each injection
        equal_time_gap = total_injection_time / self.num_attacks

        attack_messages = []
        current_time = start_time

        last_pct = -1

        for _ in range(self.num_attacks):
            # Generate one attack message
            attack_row = attack_gen_func().copy()
            attack_row["Time_Offset"] = current_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)

            # Update time for the next attack message
            current_time += equal_time_gap

            # Progress printing
            pct = int(current_time / end_time * 100)
            if pct != last_pct:
                print(f"\rInjecting '{label}' messages… {pct:3d}% complete", end="")
                last_pct = pct

        print(f"\nDone – injected {self.num_attacks} {label} messages.")

        # Merge and sort
        attack_df = pd.concat(attack_messages)
        merged_df = pd.concat([self.orig_df, attack_df], axis=0)
        
        # Reset index after concatenation to avoid issues
        merged_df = merged_df.reset_index(drop=True)
        
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset')

        return df_sort_time
    
# # Create an instance of the FuzzInjecCAN class
# attack_injector = FuzzInjecCAN(
#     orig_df=simulation_df,  # Your original DataFrame
#     time_gap_range=(10, 50),  # Time gap range (not used for equal distribution)
#     label="Fuzz",  # Label for the attack
#     start_time=0.0,  # Start time for attack message injection
#     num_attacks=100  # Number of fuzz attacks to inject
# )

# # Inject attack messages into the original dataset with equal distribution
# df_with_fuzzy = attack_injector.inject_attack_dataset(
#     attack_gen_func=attack_injector.fuzzy_df_gen,  # Using the fuzz attack generator
#     time_gap_range=(10, 50),  # Time gap range (not used here)
#     label="Fuzz"
# )



class DoSInjecCAN:
    def __init__(self, orig_df, time_gap_range=(10, 50), label="DoS", start_time=0.0, num_attacks=8000):
        """
        Initialize the DoS attack generator class.

        :param orig_df: The original DataFrame with the simulation data.
        :param time_gap_range: The range of time intervals between injected attack messages (not used here, kept for compatibility).
        :param label: The label for the attack (default is "DoS").
        :param start_time: The start time for the attack message injection.
        :param num_attacks: The number of DoS attack messages to inject (default is 8000).
        """
        self.orig_df = orig_df
        self.time_gap_range = time_gap_range  # This is not used anymore, as we're distributing evenly
        self.label = label
        self.start_time = start_time
        self.num_attacks = num_attacks
        self.cols = ["No", "Time_Offset", "Type", "CAN_ID", "Data_Length", 'One', 'Two', 
                    'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight']

    # DoS attack dataset generation without generating 'ID' column
    def dos_df_gen(self):
        """
        Generate a fuzz attack message based on the original DataFrame.
        """
        data_df = self.orig_df[1:2].copy()  # Take a small slice of the original DataFrame (adjust as needed)

        # Randomize Data Length
        data_l = random.randrange(1, 8)
        data_df["Data_Length"] = data_l
        for i in range(data_l):
            j = i + 5
            data_field = random.randrange(0, 255)
            temp = str(hex(data_field)[2:]).upper()
            for _ in range(2 - len(temp)):
                temp = '0' + temp
            data_df[self.cols[j]] = temp
        data_df["Label"] = "DoS"
        return data_df

    # Attack message injection method with equal distribution
    def inject_attack_dataset(self, attack_gen_func, time_gap_range=(10, 50), label="DoS", start_time=0.0):
        """
        Inject attack messages into the original dataset, ensuring equal distribution of attack messages over time.
        """
        end_time = float(self.orig_df.iloc[-1]["Time_Offset"])

        # Calculate the total time available for injections
        total_injection_time = end_time - start_time

        # Calculate the equal time gap between each injection
        equal_time_gap = total_injection_time / self.num_attacks

        attack_messages = []
        current_time = start_time

        last_pct = -1

        for _ in range(self.num_attacks):
            # Generate one attack message
            attack_row = attack_gen_func().copy()
            attack_row["Time_Offset"] = current_time
            attack_row["Label"] = label
            attack_messages.append(attack_row)

            # Update time for the next attack message
            current_time += equal_time_gap

            # Progress printing
            pct = int(current_time / end_time * 100)
            if pct != last_pct:
                print(f"\rInjecting '{label}' messages… {pct:3d}% complete", end="")
                last_pct = pct

        print(f"\nDone – injected {self.num_attacks} {label} messages.")

        # Merge and sort
        attack_df = pd.concat(attack_messages)
        merged_df = pd.concat([self.orig_df, attack_df], axis=0)
        
        # Reset index after concatenation to avoid issues
        merged_df = merged_df.reset_index(drop=True)
        
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset')

        return df_sort_time
    
# # Create an instance of the DoSInjecCAN class with a large number of attacks (e.g., 8000)
# attack_injector = DoSInjecCAN(
#     orig_df=simulation_df,  # Your original DataFrame
#     time_gap_range=(10, 50),  # Time gap range for attack message intervals (not used for equal distribution)
#     label="DoS",  # Label for the attack
#     start_time=0.0,  # Start time for attack message injection
#     num_attacks=8000  # Number of DoS attacks to inject (8000 attacks)
# )

# # Inject attack messages into the original dataset with equal distribution
# df_with_dos = attack_injector.inject_attack_dataset(
#     attack_gen_func=attack_injector.dos_df_gen,  # Using the DoS attack generator
#     time_gap_range=(10, 50),  # Time gap range (not used here)
#     label="DoS"
# )


class ReplayInjecCAN:
    def __init__(self, orig_df, injection_df, time_gap_range=(10, 50), label="Replay", start_time=0.0):
        """
        Initialize the Replay attack generator class.

        :param orig_df: The original DataFrame with the simulation data.
        :param injection_df: The DataFrame containing the attack data to be injected sequentially.
        :param time_gap_range: The range of time intervals between injected attack messages.
        :param label: The label for the attack (default is "Replay").
        :param start_time: The start time for the attack message injection.
        """
        self.orig_df = orig_df
        self.injection_df = injection_df
        self.time_gap_range = time_gap_range
        self.label = label
        self.start_time = start_time

    def inject_replay_sequential_from_df(self, time_gap_range=None, label=None, start_time=None):
        """
        Inject attack messages sequentially from the injection DataFrame, distributing them equally across the timeline.
        """
        if time_gap_range is None:
            time_gap_range = self.time_gap_range
        if label is None:
            label = self.label
        if start_time is None:
            start_time = self.start_time

        end_time = float(self.orig_df.iloc[-1]["Time_Offset"])
        attack_rows = []
        current_time = start_time

        # Calculate total available time for attack injections
        total_injection_time = end_time - start_time

        # Calculate the number of attack injections to be made
        num_injections = len(self.injection_df)

        # Calculate the equal time gap for each attack message
        equal_time_gap = total_injection_time / num_injections

        last_pct = -1
        injection_index = 0
        total_injections = 0

        while injection_index < num_injections:
            # Get next injection row in order
            attack_row = self.injection_df.iloc[[injection_index]].copy()
            attack_row["Time_Offset"] = current_time
            attack_row["Label"] = label
            attack_rows.append(attack_row)

            # Update time for the next attack message
            current_time += equal_time_gap
            total_injections += 1
            injection_index += 1

            # Progress
            pct = int(current_time / end_time * 100)
            if pct != last_pct:
                print(f"\rSequential injection… {pct:3d}% complete", end="")
                last_pct = pct

        print(f"\nDone – injected {total_injections} sequential messages.")

        # Merge and sort
        attack_df = pd.concat(attack_rows, ignore_index=True)
        merged_df = pd.concat([self.orig_df, attack_df], axis=0)
        merged_df["Time_Offset"] = merged_df["Time_Offset"].astype(float)
        df_sort_time = merged_df.sort_values(by='Time_Offset')

        return df_sort_time
    
# # Create an instance of the ReplayInjecCAN class
# replay_injector = ReplayInjecCAN(
#     orig_df=simulation_df,  # Your original DataFrame
#     injection_df=attack_df,  # The attack DataFrame to inject sequentially
#     time_gap_range=(10, 50),  # Time gap range for attack message intervals (no longer used)
#     label="Replay Attack",  # Label for the attack
#     start_time=0.0  # Start time for attack message injection
# )

# # Inject attack messages sequentially from the injection DataFrame with equal distribution
# df_with_replay_attack = replay_injector.inject_replay_sequential_from_df(
#     time_gap_range=(10, 50),  # Time gap range (no longer used here, only for backward compatibility)
#     label="Replay Attack"  # Label for the injected attack
# )