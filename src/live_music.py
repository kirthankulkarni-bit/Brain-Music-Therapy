import collections
import threading
import time
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import os
import csv
from datetime import datetime
import pygame

# 1. INITIALIZE AUDIO ENGINE
pygame.mixer.init()

# global state for communication between threads
current_state = "ambient"  
system_running = True

# configuration parameters
sfreq = 128  
seconds_per_chunk = 5
target_buffer_points = sfreq * seconds_per_chunk  

# path variables for your assets
AUDIO_DIR = "generated_loops"
AMBIENT_PATH = os.path.join(AUDIO_DIR, "calm_ambient.wav")
FOCUS_PATH = os.path.join(AUDIO_DIR, "focus_synthwave.wav")

if not os.path.exists(AMBIENT_PATH) or not os.path.exists(FOCUS_PATH):
    raise FileNotFoundError(f"Missing audio assets in '{AUDIO_DIR}/'. Ensure calm_ambient.wav and focus_synthwave.wav are present.")

# 2. INITIALIZE LOGGING ENGINE
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# create a unique filename based on the exact time the script starts
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(LOG_DIR, f"brain_data_{session_time}.csv")

# write the header row to the new csv file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Alpha_Power", "Beta_Power", "Raw_Ratio", "Smoothed_Ratio", "Audio_State"])
print(f"📄 Logging engine initialized. Saving data to: {csv_filename}")

# ring buffers to store raw historical data 
live_data_buffer = [collections.deque(maxlen=target_buffer_points) for _ in range(4)]
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

def background_music_generator():
    """background worker thread handling real-time audio hot-swapping"""
    global current_state, system_running
    
    print("[BG THREAD] Audio mixer engine initialized and standby.")
    pygame.mixer.music.load(AMBIENT_PATH)
    pygame.mixer.music.play(-1)
    active_playing_state = "ambient"
    
    while system_running:
        target_state = current_state
        
        if target_state != active_playing_state:
            print(f"\n[BG THREAD] Cognitive state shift detected! Fading out old loop...")
            pygame.mixer.music.fadeout(1500)
            time.sleep(1.5)  
            
            if target_state == "focus":
                pygame.mixer.music.load(FOCUS_PATH)
            else:
                pygame.mixer.music.load(AMBIENT_PATH)
                
            pygame.mixer.music.play(-1)  
            active_playing_state = target_state
            
        time.sleep(0.2)

# connect to the network stream
print("--- searching for live eeg network stream ---")
streams = resolve_byprop('type', 'EEG', timeout=3.0)
if len(streams) == 0:
    raise RuntimeError("No active EEG stream found! Run your bridge application first.")

inlet = StreamInlet(streams[0])
print(f"📡 Hooked into live stream: {streams[0].name()}")

bg_worker = threading.Thread(target=background_music_generator, daemon=True)
bg_worker.start()

print("\n--- filling initial 5-second buffer window... ---")

try:
    while system_running:
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
        
        if chunk:
            for sample in chunk:
                for i in range(4):
                    live_data_buffer[i].append(sample[i])
                    
        if len(live_data_buffer[0]) < target_buffer_points:
            time.sleep(0.1)
            continue
            
        # isolate and average frontal lobe sensors (af7 and af8)
        af7_data = np.array(live_data_buffer[1])
        af8_data = np.array(live_data_buffer[2])
        frontal_signal = (af7_data + af8_data) / 2.0
        
        info = mne.create_info(ch_names=['Frontal'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(frontal_signal.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        raw_ratio = beta_power / alpha_power
        
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # customized biological threshold
        if smoothed_ratio > 0.20:
            current_state = "focus"
        else:
            current_state = "ambient"
        
        # 3. WRITE TO CSV
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, f"{alpha_power:.4f}", f"{beta_power:.4f}", f"{raw_ratio:.4f}", f"{smoothed_ratio:.4f}", current_state])
            
        print(f"[MAIN EEG THREAD] Smooth Ratio: {smoothed_ratio:.4f} | Target Audio State: {current_state.upper()} | Logged to CSV")
        
        time.sleep(2.0)

except KeyboardInterrupt:
    print("\nterminating live engine loop...")

system_running = False
pygame.mixer.music.stop()
print("--- system deployment ended ---")