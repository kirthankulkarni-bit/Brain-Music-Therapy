import os
import time
import torch
from audiocraft.models import MusicGen
import scipy.io.wavfile

print("--- Initializing MusicGen-Small ---")
start_load = time.time()

# 1. Load the Model (Audiocraft automatically handles GPU placement here)
model = MusicGen.get_pretrained('facebook/musicgen-small')
print(f"Model loaded in {time.time() - start_load:.2f} seconds")

# 2. Configure Parameters
chunk_duration = 5
model.set_generation_params(duration=chunk_duration)

# 3. Define the Prompt
prompt = ["retro synthwave focus music, driving bassline, 120 bpm"]

# 4. Define Device for Autocast
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n--- Generating {chunk_duration}s Audio Chunk with FP16 ---")
start_gen = time.time()

# 5. Generate with FP16 Autocast Optimization
with torch.inference_mode():
    with torch.amp.autocast(device_type=device, dtype=torch.float16):
        wav = model.generate(prompt, progress=True)
print("--- Generation Complete ---")

# 6. Calculate Results
elapsed_time = time.time() - start_gen
speed_factor = elapsed_time / chunk_duration

print(f"Total Generation Time: {elapsed_time:.2f} seconds")
print(f"Hardware Speed Factor: {speed_factor:.2f}x (Compute time per second of audio)")

# 7. Save Output safely
output_dir = "test_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "musicgen_test_fp16.wav")

audio_data = wav[0, 0].cpu().float().numpy() 
scipy.io.wavfile.write(output_path, model.sample_rate, audio_data)

print(f"\nSaved sample to {output_path}")