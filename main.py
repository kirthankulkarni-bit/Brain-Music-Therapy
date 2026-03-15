from src.stream_utils import get_inlet
from src.analyze import get_band_powers
import numpy as np
import matplotlib.pyplot as plt

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
            # Clean the data of spikes (+/- 99)
            clean_data = [x for x in buffer if -99 < x < 99]
            plt.plot(clean_data)
            plt.title("Live EEG Buffer")
            plt.show(block=False)
            plt.pause(0.1)
            plt.clf()
            
            # Analyze only if enough data is clean
            if len(clean_data) > 200:
                alpha, beta = get_band_powers(np.array(clean_data))
                print(f"Alpha: {alpha:.2f} | Beta: {beta:.2f}")
            
            # Reset buffer to start a new batch
            buffer = []