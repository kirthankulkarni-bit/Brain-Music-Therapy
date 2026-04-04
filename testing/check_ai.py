import torch
from audiocraft.models import MusicGen

# Simulated "Focus" ratio (0.0 to 1.0)
# Later, this will be: ratio = (alpha / beta) from your Muse 2
simulated_ratio = 0.85 

print("--- Starting AI Simulation ---")
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Brain-to-Music Mapping Logic
if simulated_ratio > 0.7:
    prompt = "Fast electronic synthwave, 130bpm, high energy, focus"
else:
    prompt = "Lo-fi ambient piano, 70bpm, calm, relaxation"

print(f"Brain State: {'FOCUS' if simulated_ratio > 0.7 else 'RELAX'}")
print(f"Generating: {prompt}...")

# Generate 5 seconds of audio
wav = model.generate([prompt], progress=True)
print("✅ Audio generated successfully!")