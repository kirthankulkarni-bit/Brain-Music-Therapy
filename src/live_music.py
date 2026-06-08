import collections
import threading
import time
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop

# global state for communication between threads
audio_buffer = "initial_ambient_loop.wav"
next_audio_prompt = "calm ambient music"
system_running = True

# configuration parameters
sfreq = 128  # standard muse 2 sampling rate
seconds_per_chunk = 5
target_buffer_points = sfreq * seconds_per_chunk  # 640 data points

# ring buffers to store raw historical data across 4 channels (tp9, af7, af8, tp10)
# this holds the raw microvolts so we always have the most recent 5 seconds
live_data_buffer = [collections.deque(maxlen=target_buffer_points) for _ in range(4)]

# control loop smoothing parameters
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

def background_music_generator():
    """background worker thread handling local ai generation pipeline"""
    global audio_buffer, next_audio_prompt, system_running
    last_processed_prompt = None
    
    while system_running:
        current_job_prompt = next_audio_prompt
        
        if current_job_prompt != last_processed_prompt:
            print(f"\n[BG THREAD] compiling audio loop for: '{current_job_prompt}'...")
            
            # simulate processing delay of the musicgen model on the gpu
            time.sleep(3) 
            
            formatted_name = current_job_prompt.replace(' ', '_').replace(',', '') + ".wav"
            audio_buffer = f"generated_loops/{formatted_name}"
            print(f"[BG THREAD] generation complete. buffer updated to: {audio_buffer}")
            
            last_processed_prompt = current_job_prompt
            
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
previous_smoothed_ratio = None

try:
    while system_running:
        # 1. pull all available new samples from the lsl network queue
        # chunk is a list of lists: [[ch1, ch2, ch3, ch4], [ch1, ch2, ch3, ch4], ...]
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
        
        if chunk:
            for sample in chunk:
                # populate ring buffers for each channel
                for i in range(4):
                    live_data_buffer[i].append(sample[i])
                    
        # 2. only compute if our rolling history buffer is completely full
        if len(live_data_buffer[0]) < target_buffer_points:
            time.sleep(0.1)
            continue
            
        # 3. isolate frontal lobe sensors (af7 = index 1, af8 = index 2)
        # we average them to get a clean aggregate frontal metric
        af7_data = np.array(live_data_buffer[1])
        af8_data = np.array(live_data_buffer[2])
        frontal_signal = (af7_data + af8_data) / 2.0
        
        # 4. wrap array into mne structure for fourier analysis
        info = mne.create_info(ch_names=['Frontal'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(frontal_signal.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # 5. calculate band metrics and compute ratio
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        raw_ratio = beta_power / alpha_power
        
        # 6. low-pass filter smoothing
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # 7. closed-loop trend analytics
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
            base_prompt = "calm ambient music relaxation loop"
            
        next_audio_prompt = base_prompt + intensity_modifier
        
        print(f"\n[MAIN THREAD] Live Analysis Metric")
        print(f"  -> smooth ratio:  {smoothed_ratio:.4f}")
        print(f"  -> active prompt: '{next_audio_prompt}'")
        print(f"  -> audio route:   {audio_buffer}")
        
        previous_smoothed_ratio = smoothed_ratio
        
        # evaluation frequency step (analyze the rolling 5-sec buffer every 2 seconds)
        time.sleep(2.0)

except KeyboardInterrupt:
    print("\nterminating live engine loop...")

system_running = False
print("--- system deployment ended ---")