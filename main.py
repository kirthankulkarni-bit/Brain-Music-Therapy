from src.stream_utils import get_inlet
from src.analyze import get_band_powers
import numpy as np
import matplotlib.pyplot as plt
from src import analyze

# 1. Start the connection
inlet = get_inlet()

# 2. Define the buffer variable BEFORE the loop starts
buffer = [] 
BUFFER_SIZE = 256

if inlet:
    print("Stream connected. Starting data collection...")
    while True:
        # 3. Pull one sample
        sample, timestamp = inlet.pull_sample()
        
        # 4. Add the sample to the buffer
        buffer.append(sample[0])
        
        # 5. When the buffer is full, process it
        if len(buffer) >= BUFFER_SIZE:
            print("processing buffer...")
            # Clean the data of spikes (+/- 250)
            clean_data = [x for x in buffer if -250 < x < 250]
            plt.plot(clean_data) 
            plt.title("Live EEG Buffer")
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
            
            # Analyze only if enough data is clean
            if len(clean_data) > 100:
                alpha, beta = get_band_powers(np.array(clean_data))
                # Calculate the focus score
                focus_score = analyze.calculate_focus_index(alpha, beta)
                music_intensity = analyze.get_musical_parameters(focus_score)

                # Print a visual 'Focus Meter' in your terminal
                bar = "█" * int(music_intensity * 20)
                print(f"Intensity: [{bar:<20}] {round(music_intensity, 2)}")
            else:
                print(f"Data too noisy: {len(clean_data)} clean samples.")
            # Reset buffer to start a new batch
            buffer = []