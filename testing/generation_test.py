import os
import time
import torch
from audiocraft.models import MusicGen
import scipy.io.wavfile

print("--- Initializing MusicGen-Small ---")
start_load = time.time()

# This downloads the ~500MB weights on the first run
model = MusicGen.get_pretrained('facebook/musicgen-small')
print(f"Model loaded in {time.time() - start_load:.2f} seconds")

# Configure for a short, testable real-time chunk
chunk_duration = 5
model.set_generation_params(duration=chunk_duration)

# Define a clean prompt text
prompt = ["retro synthwave focus music, driving bassline, 120 bpm"]

print(f"\n--- Generating {chunk_duration}s Audio Chunk ---")
print("Targeting hardware acceleration...")

start_gen = time.time()
with torch.inference_mode():
    # Generate the audio tensor
    wav = model.generate(prompt, progress=True)
print("--- Generation Complete ---")

elapsed_time = time.time() - start_gen
speed_factor = elapsed_time / chunk_duration

print(f"Total Generation Time: {elapsed_time:.2f} seconds")
print(f"Hardware Speed Factor: {speed_factor:.2f}x (Compute time per second of audio)")

if speed_factor < 1.0:
    print("Your hardware runs faster than real-time. Stable streaming is possible!")
else:
    print("Your hardware runs slower than real-time. You will need a heavy pre-buffer queue.")

# Save the file to verify the audio quality
output_dir = "test_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "musicgen_test.wav")
# Extract raw audio data from GPU tensor to CPU numpy array
audio_data = wav[0, 0].cpu().numpy()
scipy.io.wavfile.write(output_path, model.sample_rate, audio_data)
print(f"\nSaved sample to {output_path}. Play it to check the sound profile.")