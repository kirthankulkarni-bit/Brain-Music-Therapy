import numpy as np
import time

# baseline tracking across chunks
# we will simulate 6 consecutive chunks of data
smoothed_history = [0.95, 0.88, 0.82, 0.75, 0.78, 0.85]

# default prompt parameters
base_relaxation_prompt = "calm ambient drone music"
intensity_modifier = ""

print("--- starting closed-loop feedback tracking test ---")

for chunk_idx, current_val in enumerate(smoothed_history):
    print(f"\nChunk {chunk_idx + 1}: Current Smoothed Focus = {current_val:.2f}")
    
    # check if we have at least two data points to compute a trend
    if chunk_idx > 0:
        previous_val = smoothed_history[chunk_idx - 1]
        
        # calculate the trend slope as the difference between current and previous values
        trend = current_val - previous_val
        
        # evaluation logic for a user who is under the 1.0 focus threshold
        if current_val < 1.0:
            if trend < 0:
                # focus is low and dropping further. escalate the intervention.
                intensity_modifier = ", ultra low frequency binaural beats, deep relaxation"
                print("  -> Trend: DROPPING. Increasing relaxation prompt intensity.")
            elif trend > 0:
                # focus is low but recovering. maintain standard tracking.
                intensity_modifier = ""
                print("  -> Trend: RECOVERING. Maintaining standard baseline prompt.")
            else:
                intensity_modifier = ""
                print("  -> Trend: STABLE. No change.")
        else:
            intensity_modifier = ""
            print("  -> Trend: HIGH FOCUS. System operating in optimal focus band.")
            
    else:
        print("  -> Trend: Establishing baseline baseline...")
        intensity_modifier = ""

    # construct the dynamic text prompt that gets handed to the ai thread
    final_prompt = base_relaxation_prompt + intensity_modifier
    print(f"  -> Generated Target Prompt: '{final_prompt}'")
    
    time.sleep(0.5)

print("\n--- feedback tracking test complete ---")