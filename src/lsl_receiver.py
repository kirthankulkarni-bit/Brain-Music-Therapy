from pylsl import StreamInlet, resolve_byprop
import time

print("--- searching for an active live eeg stream on the network... ---")

# 1. use resolve_byprop instead of resolve_stream
# the 3.0 specifies a timeout of 3 seconds so the script doesn't hang forever
streams = resolve_byprop('type', 'EEG', timeout=3.0)

if len(streams) == 0:
    raise RuntimeError("No active EEG stream found, Make sure your bridge app is running and streaming.")

print(f"📡 Found stream: {streams[0].name()} (Source ID: {streams[0].source_id()})")

# 2. initialize the network inlet channel to ingest the data packets
inlet = StreamInlet(streams[0])

print("\n--- starting live data collection loop ---")
print("Press Ctrl+C to terminate.")

try:
    while True:
        # 3. pull a single sample slice from the network ring buffer
        # 'sample' is an array representing the current voltage values of your electrodes
        sample, timestamp = inlet.pull_sample()
        
        # muse 2 transmits 4 primary eeg channels: TP9, AF7, AF8, TP10
        # let's look at the first 4 elements of the incoming list
        clean_voltages = [round(val, 2) for val in sample[:4]]
        
        print(f"Timestamp: {timestamp:.4f} | Volts (µV): {clean_voltages}")
        
        # throttle the terminal print slightly so it's readable
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nDetaching from live stream inlet.")