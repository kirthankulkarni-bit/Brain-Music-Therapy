from pylsl import resolve_streams, StreamInlet

def get_inlet():
    print("Looking for Muse stream...")
    # 1. Get ALL available streams
    all_streams = resolve_streams()
    
    # 2. Filter for the one you want
    target_stream = None
    for stream in all_streams:
        # Check if 'Muse' is in the stream name
        if "Muse" in stream.name():
            target_stream = stream
            break
            
    if target_stream:
        inlet = StreamInlet(target_stream)
        print(f"Successfully connected to: {target_stream.name()}")
        return inlet
    else:
        print("No Muse stream found. Is BlueMuse running?")
        return None