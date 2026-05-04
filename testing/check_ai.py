import os
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- 1. FFmpeg Manual Link ---
# This points directly to the folder you just extracted on your C: drive
ffmpeg_bin_path = r'C:\ffmpeg-2026-04-06-git-7fd2be97b9-essentials_build\bin'

if os.path.exists(os.path.join(ffmpeg_bin_path, 'ffmpeg.exe')):
    os.environ["PATH"] = ffmpeg_bin_path + os.pathsep + os.environ["PATH"]
    print(f"✅ FFmpeg Linked: {ffmpeg_bin_path}")
else:
    print("❌ ERROR: Still can't find ffmpeg.exe. Check the folder name on C:\\")

# --- 2. Simulation Parameters ---
# Later, this will be: ratio = (beta / alpha) from your DEAP dataset or Muse 2
simulated_ratio = 0.85 

print("--- Starting Brain-Music AI Simulation ---")

# --- 3. Load Model ---
# This uses the 'small' model to save VRAM on your Nitro 5
model = MusicGen.get_pretrained('facebook/musicgen-small')

# --- 4. Brain-to-Music Mapping ---
if simulated_ratio > 0.7:
    prompt = "Fast electronic synthwave, 130bpm, high energy, focus"
    state_label = "FOCUS"
else:
    prompt = "Lo-fi ambient piano, 70bpm, calm, relaxation"
    state_label = "RELAX"

print(f"Current Brain State: {state_label}")
print(f"Generating Prompt: {prompt}...")

# --- 5. Generation ---
# Generating 5 seconds of audio based on the prompt
wav = model.generate([prompt], progress=True)
print("✅ Audio tensor generated in GPU memory!")

# --- 6. Export to Disk ---
# This is the line that requires FFmpeg to be linked correctly
try:
    for idx, one_wav in enumerate(wav):
        audio_write(
            f'sample_{state_label.lower()}', 
            one_wav.cpu(), 
            model.sample_rate, 
            strategy="loudness"
        )
    print(f"🎉 SUCCESS! Created: sample_{state_label.lower()}.wav")
except Exception as e:
    print(f"❌ Export Failed: {e}")