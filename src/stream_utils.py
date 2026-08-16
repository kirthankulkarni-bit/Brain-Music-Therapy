from pylsl import resolve_streams, StreamInlet

# BlueMuse publishes SEVERAL streams at once from a single headset:
#
#   Muse-XXXX EEG            256 Hz, 4-5 channels   <- the one we want
#   Muse-XXXX PPG             64 Hz, 3 channels
#   Muse-XXXX Accelerometer   52 Hz, 3 channels
#   Muse-XXXX Gyroscope       52 Hz, 3 channels
#   Muse-XXXX Telemetry      ~0.1 Hz, battery etc.
#
# Matching on "Muse" in the stream name alone therefore selects an ARBITRARY one
# of these, whichever resolves first. That is how a session ended up reading the
# 52 Hz accelerometer and reporting it as the EEG sampling rate.
#
# Selection here is by stream TYPE first, with the name only used to disambiguate
# between multiple EEG streams. The channel count is checked too, because a 3-channel
# stream cannot be Muse EEG no matter what it calls itself.

MIN_EEG_CHANNELS = 4
EXPECTED_MUSE_RATE = 256.0


def list_streams(timeout=5.0):
    """Return every LSL stream on the network as a list of info objects."""
    return resolve_streams(wait_time=timeout)


def describe_streams(streams):
    """Human-readable table of what is on the network. Used for diagnostics."""
    if not streams:
        return "  (no LSL streams found on the network)"
    lines = [f"  {'name':<28} {'type':<14} {'ch':>3} {'rate':>9}"]
    lines.append("  " + "-" * 58)
    for s in streams:
        lines.append(
            f"  {s.name()[:27]:<28} {s.type()[:13]:<14} "
            f"{s.channel_count():>3} {s.nominal_srate():>8.1f} Hz"
        )
    return "\n".join(lines)


def get_inlet(timeout=5.0, verbose=True):
    """
    Connect to the Muse EEG stream.

    Returns (inlet, sampling_rate). The sampling rate comes from the stream's own
    metadata and is never a literal anywhere in the live path.

    Selection is by stream type 'EEG' with at least MIN_EEG_CHANNELS channels, so
    the accelerometer, gyroscope, PPG and telemetry streams cannot be picked up by
    accident. Returns (None, None) if no suitable stream exists, after printing
    everything it did find.

    Callers should always unpack both values:
        inlet, sampling_rate = get_inlet()
    """
    print("Looking for Muse EEG stream...")
    all_streams = list_streams(timeout)

    if not all_streams:
        print("No LSL streams found at all. Is BlueMuse running and streaming?")
        return None, None

    eeg_streams = [
        s for s in all_streams
        if s.type().upper() == "EEG" and s.channel_count() >= MIN_EEG_CHANNELS
    ]

    if not eeg_streams:
        print("\nNo EEG stream with >= {} channels found. What IS on the network:\n".format(MIN_EEG_CHANNELS))
        print(describe_streams(all_streams))
        print("\nIf you see Accelerometer/Gyroscope/PPG but no EEG, enable EEG in")
        print("BlueMuse settings, then stop and restart streaming.")
        return None, None

    # Prefer a stream whose name mentions Muse, but any qualifying EEG stream beats none.
    target = next((s for s in eeg_streams if "muse" in s.name().lower()), eeg_streams[0])

    if verbose and len(all_streams) > 1:
        print("\nStreams on the network:")
        print(describe_streams(all_streams))
        print()

    inlet = StreamInlet(target)
    sampling_rate = inlet.info().nominal_srate()
    print(f"Connected to: {target.name()}  [type={target.type()}, "
          f"{target.channel_count()} ch, {sampling_rate:g} Hz]")

    if sampling_rate <= 0:
        print("Stream reports an irregular sampling rate. Run scripts/verify_sample_rate.py.")
    elif abs(sampling_rate - EXPECTED_MUSE_RATE) > 1.0:
        print(
            f"\nWARNING: expected {EXPECTED_MUSE_RATE:g} Hz from a Muse 2, got {sampling_rate:g} Hz."
        )
        print("Confirm with scripts/verify_sample_rate.py before collecting any data.")

    return inlet, sampling_rate


if __name__ == "__main__":
    # python src/stream_utils.py  -> show everything on the network, connect to nothing
    streams = list_streams(5.0)
    print("\nLSL streams visible right now:\n")
    print(describe_streams(streams))
    print()
