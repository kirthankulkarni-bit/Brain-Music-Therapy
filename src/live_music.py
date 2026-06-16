import collections
import threading
import time
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import os

# 1. INITIALIZE AUDIO ENGINE
import pygame
pygame.mixer.init()

# global state for communication between threads
current_state = "ambient"  # can be "ambient" or "focus"
system_running = True

# configuration parameters
sfreq = 128  # standard muse 2 sampling rate
seconds_per_chunk = 5
target_buffer_points = sfreq * seconds_per_chunk  # 640 data points

# path variables for your assets
AUDIO_DIR = "generated_loops"
AMBIENT_PATH = os.path.join(AUDIO_DIR, "calm_ambient.wav")
FOCUS_PATH = os.path.join(AUDIO_DIR, "focus_synthwave.wav")

# verify files exist before starting
if not os.path.exists(AMBIENT_PATH) or not os.path.exists(FOCUS_PATH):
    raise FileNotFoundError(f"Missing audio assets in '{AUDIO_DIR}/'. Ensure calm_ambient.wav and focus_synthwave.wav are present.")

# ring buffers to store raw historical data across 4 channels (tp9, af7, af8, tp10)
live_data_buffer = [collections.deque(maxlen=target_buffer_points) for _ in range(4)]

# control loop smoothing parameters
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

def background_music_generator():
    """background worker thread handling real-time audio hot-swapping"""
    global current_state, system_running
    
    print("[BG THREAD] Audio mixer engine initialized and standby.")
    
    # start by playing the baseline ambient loop infinitely (-1 means loop forever)
    pygame.mixer.music.load(AMBIENT_PATH)
    pygame.mixer.music.play(-1)
    active_playing_state = "ambient"
    
    while system_running:
        # poll the current target state dictated by the main EEG thread
        target_state = current_state
        
        # if the brain state changed, swap the audio track smoothly
        if target_state != active_playing_state:
            print(f"\n[BG THREAD] Cognitive state shift detected! Fading out old loop...")
            
            # fade out current track over 1.5 seconds
            pygame.mixer.music.fadeout(1500)
            time.sleep(1.5)  # wait for fadeout to complete
            
            # load and play the new track based on state
            if target_state == "focus":
                print("[BG THREAD] Loading: focus_synthwave.wav (Drums/Percussion Active)")
                pygame.mixer.music.load(FOCUS_PATH)
            else:
                print("[BG THREAD] Loading: calm_ambient.wav (Ambient Pads/Drone Active)")
                pygame.mixer.music.load(AMBIENT_PATH)
                
            pygame.mixer.music.play(-1)  # play new track on infinite loop
            active_playing_state = target_state
            
        time.sleep(0.2)

# connect to the network stream
print("--- searching for live eeg network stream ---")
streams = resolve_byprop('type', 'EEG', timeout=3.0)
if len(streams) == 0:
    raise RuntimeError("No active EEG stream found! Run your bridge application first.")

inlet = StreamInlet(streams[0])
print(f"📡 Hooked into live stream: {streams[0].name()}")

# spin up background audio lane
bg_worker = threading.Thread(target=background_music_generator, daemon=True)
bg_worker.start()

print("\n--- filling initial 5-second buffer window... ---")

try:
    while system_running:
        # pull available new samples from lsl
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
        
        if chunk:
            for sample in chunk:
                for i in range(4):
                    live_data_buffer[i].append(sample[i])
                    
        # only compute if buffer is completely full
        if len(live_data_buffer[0]) < target_buffer_points:
            time.sleep(0.1)
            continue
            
        # isolate and average frontal lobe sensors (af7 and af8)
        af7_data = np.array(live_data_buffer[1])
        af8_data = np.array(live_data_buffer[2])
        frontal_signal = (af7_data + af8_data) / 2.0
        
        # wrap into mne structure for fourier analysis
        info = mne.create_info(ch_names=['Frontal'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(frontal_signal.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # calculate beta/alpha ratio
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        raw_ratio = beta_power / alpha_power
        
        # low-pass filter smoothing
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # update target audio state based on ratio threshold
        # threshold of 1.0 is a starting baseline; adjust this based on your personal readings
        if smoothed_ratio > 1.0:
            current_state = "focus"
        else:
            current_state = "ambient"
        
        print(f"[MAIN EEG THREAD] Smooth Ratio: {smoothed_ratio:.4f} | Target Audio State: {current_state.upper()}")
        
        # evaluate the rolling 5-sec buffer every 2 seconds
        time.sleep(2.0)

except KeyboardInterrupt:
    print("\nterminating live engine loop...")

system_running = False
pygame.mixer.music.stop()
print("--- system deployment ended ---")