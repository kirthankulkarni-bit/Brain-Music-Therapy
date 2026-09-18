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

# A backward timestamp step larger than this is a clock reset, not a duplicate, and is
# accepted rather than filtered. Duplicated packets step back by one chunk (~43 ms);
# filtering a genuine reset instead would silently discard every sample until the new
# clock caught up with the old one.
_CLOCK_RESET_S = 1.0


# How long a sample is held before release, so a packet that arrives after a later one
# can be slotted back into order. Measured on the live stream, 2026-09-17: 92 of 5148
# unique samples arrived late, median 6.0 ms, p99 13.7 ms, max 13.9 ms. 50 ms covers that
# with 3.5x margin and adds 50 ms to a 6.5 s end-to-end budget.
_REORDER_HOLD_S = 0.05

# Seen timestamps older than this behind the newest are forgotten, bounding memory.
_SEEN_HORIZON_S = 5.0


class DedupInlet:
    """
    An LSL inlet that removes duplicated packets and restores sample order.

    WHY THIS EXISTS. On 2026-09-17 BlueMuse delivered every 12-sample packet more than
    once - four copies at first, two after a full BlueMuse restart AND a Windows Bluetooth
    reset. Of 8292 samples received in 8 s, 2052 were unique: exactly 256 Hz. Each packet
    arrived, the timestamps stepped back 43 ms, and the same packet arrived again.

    Nothing downstream checked. Every script counts samples, not time, so a duplicated
    stream reads as the signal leaping backwards every 12 samples - a discontinuity large
    enough that contact_check called AF7 and AF8 "RAILING" on a headset that was probably
    fine. live_music would have recorded and analysed it as real data.

    THE FIRST VERSION OF THIS WAS WRONG, and the way it was wrong is the design. It
    dropped any sample whose timestamp did not advance. On a 10 s live capture that kept
    2457 of 2568 genuine samples - it discarded 111 real ones, 4.3%, because the copies
    interleave: a genuine packet sometimes arrives AFTER a copy of a later packet, and
    "does not advance" cannot tell that from a duplicate. The truly unique stream was a
    perfect 256.0 Hz; the loss was entirely the filter's.

    So duplicates are identified by timestamp IDENTITY (a set of timestamps already seen),
    not by order, and samples are held for _REORDER_HOLD_S and released sorted, so a late
    genuine packet slots back into place. A sample later than the hold - never observed -
    is dropped and counted separately as late, rather than being emitted out of order.

    A backward jump beyond _CLOCK_RESET_S is a clock reset: the buffer is flushed and
    state restarts, rather than discarding real data until the new clock catches up.

    Counts are kept so each session records that filtering happened and how much - the
    raw file on disk is the filtered stream, and a reader needs to know.
    """

    def __init__(self, inlet):
        self._inlet = inlet
        self._seen = set()
        self._pending = []          # (ts, sample) awaiting release
        self._newest = None         # highest timestamp received
        self._last_out = None       # highest timestamp released
        self.received = 0
        self.dropped = 0            # duplicates
        self.late = 0               # genuine but later than the hold
        self._warned = False

    def _release(self, flush=False):
        if not self._pending:
            return [], []
        if flush:
            ready = self._pending
            self._pending = []
        else:
            cutoff = self._newest - _REORDER_HOLD_S
            ready = [p for p in self._pending if p[0] <= cutoff]
            self._pending = [p for p in self._pending if p[0] > cutoff]
        ready.sort(key=lambda p: p[0])
        out_s, out_t = [], []
        for ts, sample in ready:
            if self._last_out is not None and ts <= self._last_out:
                self.late += 1
                continue
            out_s.append(sample)
            out_t.append(ts)
            self._last_out = ts
        return out_s, out_t

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        samples, timestamps = self._inlet.pull_chunk(timeout=timeout, max_samples=max_samples)
        pre_s, pre_t = [], []
        for sample, ts in zip(samples or [], timestamps or []):
            self.received += 1
            if self._newest is not None and ts < self._newest - _CLOCK_RESET_S:
                # Clock reset: release everything held, then start over on the new clock.
                fs, ft = self._release(flush=True)
                pre_s += fs
                pre_t += ft
                self._seen.clear()
                self._newest = None
                self._last_out = None
            if ts in self._seen:
                self.dropped += 1
                continue
            self._seen.add(ts)
            self._pending.append((ts, sample))
            if self._newest is None or ts > self._newest:
                self._newest = ts
        if self._newest is not None and len(self._seen) > 4096:
            horizon = self._newest - _SEEN_HORIZON_S
            self._seen = {t for t in self._seen if t >= horizon}
        if self.dropped and not self._warned:
            self._warned = True
            print(chr(10) + "[stream] WARNING: duplicated packets on the EEG stream - removing them. "
                  "The data kept is unaffected; see stream_utils.DedupInlet.")
        out_s, out_t = self._release()
        return pre_s + out_s, pre_t + out_t

    def summary(self):
        """Counts for the session log."""
        frac = self.dropped / self.received if self.received else 0.0
        return {"stream_samples_received": self.received,
                "stream_duplicates_dropped": self.dropped,
                "stream_duplicate_fraction": round(frac, 4),
                "stream_late_samples_dropped": self.late,
                "stream_reorder_hold_s": _REORDER_HOLD_S}

    def __getattr__(self, name):
        return getattr(self._inlet, name)


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

    inlet = DedupInlet(StreamInlet(target))
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
