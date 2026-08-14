"""
RETIRED - was src/buffer_stream.py. Kept for provenance; nothing imports it.

Earlier draft of stream_processing_sim.py: same time.sleep(3) generation stub,
binary prompt switching, no trend term. Superseded twice over, first by
stream_processing.py and then by src/music_engine.py.

Run from the testing/ directory: it expects ../s01.dat.
"""

import pickle
import numpy as np
import mne
import collections
import threading
import time

# shared memory space for our concurrent lanes
# the main lane reads the buffer, the background lane writes to it
audio_buffer = "initial_ambient_loop.wav"
next_audio_prompt = "calm ambient music"
system_running = True

# rolling window setup from step 1
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

# load deap data from the root folder
with open('../s01.dat', 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

# use trial 0, channel 0 (frontal lobe f3)
trial_data = raw_data['data'][0, 0, :] 
sfreq = 128
points_per_chunk = sfreq * 5 # 5-second steps (640 points)

total_points = len(trial_data)
num_chunks = total_points // points_per_chunk

def background_music_generator():
    """this lane watches the prompt whiteboard and creates new music paths"""
    global audio_buffer, next_audio_prompt, system_running
    last_processed_prompt = None
    
    while system_running:
        current_job_prompt = next_audio_prompt
        
        # only simulate a new generation if the prompt has actually changed
        if current_job_prompt != last_processed_prompt:
            print(f"\n[BG THREAD] detected state change. generating: '{current_job_prompt}'...")
            
            # simulate the 3 seconds it takes musicgen to render an audio loop
            time.sleep(3) 
            
            # update the buffer with the new path name
            formatted_name = current_job_prompt.replace(' ', '_') + ".wav"
            audio_buffer = f"generated_loops/{formatted_name}"
            print(f"[BG THREAD] track ready! buffer updated to: {audio_buffer}")
            
            last_processed_prompt = current_job_prompt
            
        # brief pause to keep the while loop from overheating a core
        time.sleep(0.2)

# spin up our background lane before the main loop starts
bg_worker = threading.Thread(target=background_music_generator, daemon=True)
bg_worker.start()

print("--- starting concurrent data stream and playback loop ---")

try:
    for chunk_idx in range(num_chunks):
        # 1. slice out the current 5-second raw block
        start_pt = chunk_idx * points_per_chunk
        end_pt = start_pt + points_per_chunk
        chunk_data = trial_data[start_pt:end_pt]
        
        # 2. run the fourier transform via mne to get frequencies
        info = mne.create_info(ch_names=['F3'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(chunk_data.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # 3. separate alpha and beta power bands
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        raw_ratio = beta_power / alpha_power
        
        # 4. update our rolling window to get the smooth index trend
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # 5. change the target prompt text based on the smoothed trend
        if smoothed_ratio > 1.0:
            next_audio_prompt = "high energy synthwave focus loop"
        else:
            next_audio_prompt = "calm ambient music relaxation loop"
            
        print(f"\n[MAIN THREAD] processing chunk {chunk_idx + 1}/{num_chunks}")
        print(f"  -> smooth ratio: {smoothed_ratio:.4f}")
        print(f"  -> now playing:  {audio_buffer}")
        
        # simulate playing back the 5-second loop audio track
        time.sleep(5)

except KeyboardInterrupt:
    print("\nshutting down streaming system...")

system_running = False
print("--- system stopped ---")