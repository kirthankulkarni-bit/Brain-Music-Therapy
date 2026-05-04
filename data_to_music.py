import pickle
import numpy as np
import mne
from audiocraft.models import MusicGen
import scipy.io.wavfile as wav

# 1. LOAD THE REAL BRAIN DATA
print("--- Loading DEAP Data ---")
with open('s01.dat', 'rb') as f:
    raw_data = pickle.load(f, encoding='latin1')

# Grab Trial 0, Channel 0 (Frontal Lobe)
# Data shape is (trial, channel, data)
# Channel 0 is F3 (frontal lobe), channel 13 is FC2 (frontal-center), channel 16 is P3 (parietal lobe), channel 31 is O2 (occipital lobe)
trial_data = raw_data['data'][0, 16, :] 
sfreq = 128 

# 2. CALCULATE THE RATIO (The 'Brain' Logic)
# Convert to MNE to use their powerful spectral tools
info = mne.create_info(ch_names=['F3'], sfreq=sfreq, ch_types=['eeg'])
raw = mne.io.RawArray(trial_data.reshape(1, -1), info)

spectrum = raw.compute_psd(fmin=1, fmax=40, verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)

alpha_mask = (freqs >= 8) & (freqs <= 13)
beta_mask = (freqs >= 13) & (freqs <= 30)

alpha_power = psds[0][alpha_mask].mean()
beta_power = psds[0][beta_mask].mean()
focus_ratio = beta_power / alpha_power

print(f"Calculated Focus Ratio: {focus_ratio:.4f}")

# 3. GENERATE THE MUSIC (The 'AI' Logic)
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=5)

if focus_ratio > 1.0:
    prompt = "high-energy electronic synthwave for deep concentration"
    print("Detected: HIGH FOCUS -> Generating Synthwave")
else:
    prompt = "calm ambient piano with soft strings for relaxation"
    print("Detected: LOW FOCUS -> Generating Ambient")

descriptions = [prompt]
wav_output = model.generate(descriptions)

# 4. SAVE TO DISK
sampling_rate = 32000
audio_data = wav_output[0, 0].cpu().numpy()
wav.write('data_output.wav', rate=sampling_rate, data=audio_data)

print("--- FINISHED: Listen to 'data_output.wav' ---")