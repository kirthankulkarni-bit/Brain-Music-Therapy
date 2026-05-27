import pickle
import numpy as np
import mne
import collections
import time

# set up the rolling window to smooth out the ratios over time
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

# load participant 1 data
with open('s01.dat', 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

# grab trial 0, channel 0 (frontal lobe / f3)
# deap data is sampled at 128hz, meaning 128 data points per second
trial_data = raw_data['data'][0, 0, :] 
sfreq = 128
points_per_chunk = sfreq * 5 # 5 seconds of data per chunk (640 points)

# find how many 5-second chunks we can slice out of the total trial length
total_points = len(trial_data)
num_chunks = total_points // points_per_chunk

print(f"--- analyzing {num_chunks} consecutive chunks of real brainwaves ---")

for chunk_idx in range(num_chunks):
    # slice the data array to get just the current 5-second window
    start_pt = chunk_idx * points_per_chunk
    end_pt = start_pt + points_per_chunk
    chunk_data = trial_data[start_pt:end_pt]
    
    # pass the chunk into mne to compute frequencies
    info = mne.create_info(ch_names=['F3'], sfreq=sfreq, ch_types=['eeg'])
    raw_chunk = mne.io.RawArray(chunk_data.reshape(1, -1), info, verbose=False)
    
    spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # pull out alpha and beta values
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    beta_mask = (freqs >= 13) & (freqs <= 30)
    
    alpha_power = psds[0][alpha_mask].mean()
    beta_power = psds[0][beta_mask].mean()
    raw_ratio = beta_power / alpha_power
    
    # add the new raw ratio to the rolling window and average it
    ratio_window.append(raw_ratio)
    smoothed_ratio = np.mean(ratio_window)
    
    print(f"Chunk {chunk_idx + 1}/{num_chunks}")
    print(f"  -> Raw Ratio:      {raw_ratio:.4f}")
    print(f"  -> Smoothed Ratio:  {smoothed_ratio:.4f}")
    
    if smoothed_ratio > 1.0:
        print("  DECISION: target focus music")
    else:
        print("  DECISION: target calm music")
    print("-" * 40)
    
    # small delay just so we can read the terminal updates easily
    time.sleep(0.5)