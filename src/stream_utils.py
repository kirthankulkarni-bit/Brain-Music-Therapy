from pylsl import resolve_streams, StreamInlet


def get_inlet(timeout=5.0):
    """
    Connect to the Muse LSL stream.

    Returns (inlet, sampling_rate). The sampling rate comes from the stream's own
    metadata and is never a literal anywhere in the live path - that is the
    structural fix for the 128-vs-256 Hz defect. As long as any file can construct
    a rate from a hardcoded number, the bug can come back.

    Callers should always unpack both values:
        inlet, sampling_rate = get_inlet()
    """
    print("Looking for Muse stream...")
    all_streams = resolve_streams(wait_time=timeout)

    target_stream = None
    for stream in all_streams:
        if "Muse" in stream.name():
            target_stream = stream
            break

    if not target_stream:
        print("No Muse stream found. Is BlueMuse running?")
        return None, None

    inlet = StreamInlet(target_stream)
    sampling_rate = inlet.info().nominal_srate()
    print(f"Successfully connected to: {target_stream.name()} at {sampling_rate:g} Hz")

    if sampling_rate <= 0:
        print("Stream reports an irregular sampling rate. Run scripts/verify_sample_rate.py.")
    elif abs(sampling_rate - 256.0) > 1.0:
        print(
            f"WARNING: expected 256 Hz from a Muse 2, got {sampling_rate:g} Hz. "
            "Confirm with scripts/verify_sample_rate.py before collecting data."
        )

    return inlet, sampling_rate
