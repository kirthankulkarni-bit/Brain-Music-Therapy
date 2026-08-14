"""
RETIRED - was src/stream_processing.py. Kept for provenance; nothing imports it.

This is the DEAP-driven simulation of the closed loop. The generative stage is a
time.sleep(3) stub, not a model: no audio was ever produced by this file. It is
retained because it documents the design progression (trend-sensitive prompt
construction first appeared here) and because the roadmap's claim that "the
generative loop existed only as a stub" should be verifiable in the repo.

Superseded by src/music_engine.py, which does continuation-based MusicGen
generation with a real bounded queue and gapless callback playback.

Run from the testing/ directory: it expects ../s01.dat.
"""

import pickle
import numpy as np
import mne
import collections
import threading
import time

# global state for communication between threads
audio_buffer = "initial_ambient_loop.wav"
next_audio_prompt = "calm ambient music"
system_running = True

# step 1 parameters: rolling window configuration
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

# load the raw data matrix 
with open('../s01.dat', 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

# extract data points for target channel (f3)
trial_data = raw_data['data'][0, 0, :] 
sfreq = 128
points_per_chunk = sfreq * 5  # 5-second sampling block (640 points)

total_points = len(trial_data)
num_chunks = total_points // points_per_chunk

def background_music_generator():
    """background worker thread handling local ai generation pipeline"""
    global audio_buffer, next_audio_prompt, system_running
    last_processed_prompt = None
    
    while system_running:
        current_job_prompt = next_audio_prompt
        
        # only trigger a generation cycle if the prompt has shifted
        if current_job_prompt != last_processed_prompt:
            print(f"\n[BG THREAD] compiling audio loop for: '{current_job_prompt}'...")
            
            # simulate processing delay of the musicgen model on the gpu
            time.sleep(3) 
            
            # format filename and update the shared audio storage link
            formatted_name = current_job_prompt.replace(' ', '_').replace(',', '') + ".wav"
            audio_buffer = f"generated_loops/{formatted_name}"
            print(f"[BG THREAD] generation complete. buffer updated to: {audio_buffer}")
            
            last_processed_prompt = current_job_prompt
            
        time.sleep(0.2)

# spin up step 2 background audio lane
bg_worker = threading.Thread(target=background_music_generator, daemon=True)
bg_worker.start()

print("--- launching integrated closed-loop brain-music architecture ---")

previous_smoothed_ratio = None

try:
    for chunk_idx in range(num_chunks):
        # 1. extract the current signal slice
        start_pt = chunk_idx * points_per_chunk
        end_pt = start_pt + points_per_chunk
        chunk_data = trial_data[start_pt:end_pt]
        
        # 2. apply fourier transform to isolate signal power frequencies
        info = mne.create_info(ch_names=['F3'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(chunk_data.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # 3. separate band windows and calculate current ratio
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        raw_ratio = beta_power / alpha_power
        
        # 4. push data to rolling window for low-pass smoothing
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # 5. step 3 feedback logic: evaluate slope delta to alter text composition
        intensity_modifier = ""
        if previous_smoothed_ratio is not None:
            trend = smoothed_ratio - previous_smoothed_ratio
            
            if smoothed_ratio < 1.0:
                base_prompt = "calm ambient music relaxation loop"
                if trend < 0:
                    intensity_modifier = ", deep low frequency binaural drone"
            else:
                base_prompt = "high energy synthwave focus loop"
                if trend > 0:
                    intensity_modifier = ", aggressive driving percussion"
        else:
            # default configuration for the initial setup block
            base_prompt = "calm ambient music relaxation loop"
            
        # update target instructions for the generator thread
        next_audio_prompt = base_prompt + intensity_modifier
        
        print(f"\n[MAIN THREAD] processing chunk {chunk_idx + 1}/{num_chunks}")
        print(f"  -> smooth ratio:  {smoothed_ratio:.4f}")
        print(f"  -> active prompt: '{next_audio_prompt}'")
        print(f"  -> audio route:   {audio_buffer}")
        
        # cache current index state for the next calculation loop
        previous_smoothed_ratio = smoothed_ratio
        
        # simulate gapless loop execution block
        time.sleep(5)

except KeyboardInterrupt:
    print("\nterminating system execution...")

system_running = False
print("--- system deployment ended ---")