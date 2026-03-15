from pylsl import StreamInlet, resolve_streams

print("Looking for an LSL stream...")
# This will wait until it finds *any* stream on the network
streams = resolve_streams()

# Create a new inlet to read from the stream
inlet = StreamInlet(streams[0])
print(f"Connected to stream: {streams[0].name()}")

while True:
    # Get a new sample (this is the raw data)
    sample, timestamp = inlet.pull_sample()
    print(f"Data: {sample}")