"""
session_logger.py - structured session recording.

Replaces the inline CSV writer in the old live_music.py.

Why JSONL instead of CSV: the old CSV had one fixed schema (six columns) and no
way to record anything that was not a window - not the baseline statistics, not
the audio segments, not the rejected windows, not the true sampling rate. And
analyze_session.py had to reconstruct elapsed time by multiplying the row count
by the 2 s loop period, which quietly becomes wrong the moment a window is
rejected or the loop stalls.

Each session writes three files into sessions/<session_id>/:

  events.jsonl   one JSON object per line: manifest, window, audio_segment, note
  raw_eeg.f32    raw float32 samples, (n_samples, 1 + n_channels): [lsl_ts, ch...]
  manifest.json  a standalone copy of the manifest, for quick inspection

The raw file exists so the whole session can be re-analyzed offline with different
windows, filters, or band definitions without recollecting data. Given the
sampling-rate defect, being able to reprocess is not optional.
"""

from __future__ import annotations

import glob
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

SESSIONS_ROOT = "sessions"
SCHEMA_VERSION = 2


class SessionLogger:
    """Append-only session writer. Use as a context manager."""

    def __init__(
        self,
        participant_id: str = "self",
        condition: str = "adaptive",
        root: str = SESSIONS_ROOT,
        n_channels: int = 4,
        session_id: Optional[str] = None,
        synthetic: bool = False,
    ):
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.participant_id = participant_id
        self.condition = condition
        self.n_channels = n_channels
        self.synthetic = synthetic

        # A synthetic run is named so it CANNOT be mistaken for data, and .gitignore
        # excludes the pattern so it cannot be committed by a `git add -A`.
        #
        # Defence in depth, because the manifest marker alone was not enough. It already
        # existed and nothing read it, and on 9/5 a thirty-second alpha_test --demo
        # recomputed six manuscript claims from a signal generator purely by sorting
        # after the real recording. real_sessions() reads the marker now; this makes the
        # mistake visible in `ls` and unpushable as well.
        prefix = "DEMO_" if synthetic else ""
        self.dir = os.path.join(root, f"{prefix}{participant_id}_{self.session_id}")
        os.makedirs(self.dir, exist_ok=True)

        self.events_path = os.path.join(self.dir, "events.jsonl")
        self.raw_path = os.path.join(self.dir, "raw_eeg.f32")
        self.manifest_path = os.path.join(self.dir, "manifest.json")

        self._events = open(self.events_path, "a", encoding="utf-8", newline="\n")
        self._raw = open(self.raw_path, "ab")
        self._t0 = time.time()
        self.n_raw_samples = 0
        self.raw_time_origin: Optional[float] = None
        self.counts: Dict[str, int] = {}

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            self.note(f"session ended with {exc_type.__name__}: {exc}", level="error")
        self.close()

    def close(self) -> None:
        for handle in (self._events, self._raw):
            try:
                handle.flush()
                handle.close()
            except (OSError, ValueError):
                pass

    # ---------------------------------------------------------------- writes

    def _write(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "type": event_type,
            "wall_time": time.time(),
            "elapsed_s": round(time.time() - self._t0, 4),
        }
        record.update(payload)
        self._events.write(json.dumps(record, default=_json_default) + "\n")
        self._events.flush()  # crash-safe: a killed session keeps everything up to the crash
        self.counts[event_type] = self.counts.get(event_type, 0) + 1

    def write_manifest(self, **fields: Any) -> Dict[str, Any]:
        """
        Call once, before data collection. Must include the empirically verified
        sampling rate - that is the field whose absence caused the original defect.
        """
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "condition": self.condition,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "n_channels": self.n_channels,
            "raw_dtype": "float32",
            "raw_layout": "[seconds since first sample, ch0..chN]; absolute LSL origin in the raw time origin note",
        }
        manifest.update(fields)
        if "sampling_rate" not in manifest:
            raise ValueError("manifest must record the sampling_rate actually used")
        self._write("manifest", manifest)
        with open(self.manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=_json_default)
        return manifest

    def log_raw(self, timestamps: Iterable[float], samples: Iterable[Iterable[float]]) -> None:
        """
        Append a pulled LSL chunk to the raw binary file.

        TIMESTAMPS ARE STORED RELATIVE TO THE FIRST SAMPLE. They used to be written as raw
        LSL time cast straight to float32. LSL time is seconds since the machine booted,
        and float32 carries 24 bits of mantissa, so the stored precision depends on UPTIME:
        about 8 ms after a fresh boot (PILOT01, clock ~70,000 s), but 0.25 s once the laptop
        had been up 27 days (the 2026-09-17 alpha test and PILOT02, clock ~2,339,000 s).
        Every sample in a quarter-second shared one timestamp, and anything aligning by time
        - the estimator sweep, the replay, the eye-closure check - was working from a
        quarter-second staircase.

        The subtraction happens in float64, before the cast. Relative to the first sample
        a twenty-minute session needs values up to ~1300 s, where float32 resolves ~0.1 ms,
        and even three hours stays under 1 ms. The absolute origin is logged once, at full
        precision, as a note - so nothing is lost, and every existing reader, which only
        ever uses differences, is unaffected.
        """
        ts64 = np.asarray(list(timestamps), dtype=np.float64).reshape(-1, 1)
        if ts64.size == 0:
            return
        if self.raw_time_origin is None:
            self.raw_time_origin = float(ts64[0, 0])
            self.note("raw time origin", level="info", lsl_time_origin=self.raw_time_origin,
                      raw_timestamps="seconds relative to lsl_time_origin")
        ts = (ts64 - self.raw_time_origin).astype(np.float32)
        data = np.asarray([list(s)[: self.n_channels] for s in samples], dtype=np.float32)
        if data.shape[0] != ts.shape[0]:
            return
        np.hstack([ts, data]).astype(np.float32).tofile(self._raw)
        self.n_raw_samples += ts.shape[0]

    def log_window(self, phase: str, features: Any, **extra: Any) -> None:
        """
        features: a BandFeatures instance (or any object with .to_dict()).
        Rejected windows MUST be logged too - the rejection rate is a reportable
        signal-quality metric, and silently dropping them biases every other number.
        """
        payload: Dict[str, Any] = {"phase": phase}
        payload.update(features.to_dict() if hasattr(features, "to_dict") else dict(features))
        payload.update(extra)
        self._write("window", payload)

    def log_audio_segment(
        self,
        prompt: str,
        duration_s: float,
        generation_s: float,
        queue_depth: int,
        segment_index: int,
        **extra: Any,
    ) -> None:
        payload = {
            "prompt": prompt,
            "duration_s": duration_s,
            "generation_s": generation_s,
            "realtime_factor": (generation_s / duration_s) if duration_s else float("nan"),
            "queue_depth": queue_depth,
            "segment_index": segment_index,
        }
        payload.update(extra)
        self._write("audio_segment", payload)

    def log_baseline(self, **fields: Any) -> None:
        self._write("baseline", fields)

    def note(self, message: str, level: str = "info", **extra: Any) -> None:
        payload = {"message": message, "level": level}
        payload.update(extra)
        self._write("note", payload)


# --------------------------------------------------------------------- reading


def load_session(session_dir: str) -> Dict[str, Any]:
    """
    Load a session directory into plain lists. Used by analyze_session.py and by
    any offline reanalysis.

    Returns {"manifest": dict, "windows": [...], "audio": [...], "baseline": [...],
             "notes": [...], "dir": str}
    """
    events_path = os.path.join(session_dir, "events.jsonl")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"No events.jsonl in {session_dir}")

    out: Dict[str, Any] = {
        "dir": session_dir,
        "manifest": {},
        "windows": [],
        "audio": [],
        "baseline": [],
        "notes": [],
    }
    bucket = {
        "manifest": None,
        "window": "windows",
        "audio_segment": "audio",
        "baseline": "baseline",
        "note": "notes",
    }
    with open(events_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A session killed mid-write can leave one truncated final line.
                out["notes"].append({"message": f"unparseable line {line_no}", "level": "warning"})
                continue
            kind = event.get("type")
            if kind == "manifest":
                out["manifest"] = event
            elif kind in bucket:
                out[bucket[kind]].append(event)
    return out


def load_raw(session_dir: str, n_channels: int = 4) -> np.ndarray:
    """Return the raw recording as (n_samples, 1 + n_channels): [lsl_ts, ch...]."""
    raw_path = os.path.join(session_dir, "raw_eeg.f32")
    flat = np.fromfile(raw_path, dtype=np.float32)
    width = 1 + n_channels
    usable = (flat.size // width) * width
    return flat[:usable].reshape(-1, width)


def sample_clock(timestamps: np.ndarray, sampling_rate: float,
                 min_gap_s: float = 0.5) -> np.ndarray:
    """
    Per-sample times in seconds from the first sample, corrected for stream dropouts.

    Samples are counted at the nominal rate - the most precise clock available, because
    the headset samples at exactly 256 Hz - and each dropout's duration is added where
    the timestamps show one. Neither source alone is right:

      sample_index / fs   exact between dropouts, but after one it runs behind real time
                          by the dropout's length, permanently. The 2026-09-17 alpha test
                          lost 11.75 s and then 20.25 s, so by the end every window was
                          32 s early and the estimator sweep's reproduction fell to
                          r = 0.77 and refused to report.

      raw timestamps      locate dropouts, but on sessions written before 2026-09-17
                          they can be quantised to 0.25 s (see SessionLogger.log_raw), so
                          they cannot be used sample by sample.

    A step of more than min_gap_s between consecutive stored timestamps is a dropout;
    normal steps are one sample, or at worst the 0.25 s quantisation. Each dropout's
    estimated length carries that quantisation as error (up to 0.25 s), against gaps
    of tens of seconds. With no dropouts this returns exactly arange(n) / fs, so
    recordings without gaps are unchanged.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    n = ts.size
    t = np.arange(n, dtype=np.float64) / float(sampling_rate)
    if n < 2:
        return t
    step = np.diff(ts)
    gaps = np.where(step > min_gap_s)[0]
    if gaps.size:
        extra = np.zeros(n, dtype=np.float64)
        extra[gaps + 1] = step[gaps] - 1.0 / float(sampling_rate)
        t = t + np.cumsum(extra)
    return t


# Manifest values of sampling_rate_source that mean the recording is SYNTHETIC. The
# marker already existed - alpha_test writes "demo" and live_music writes "mock" - but
# nothing read it, and every selector below took the newest matching directory.
#
# That is a real failure mode and it fired on 2026-09-05: running alpha_test --demo to
# smoke-test the hardware gate wrote sessions/alphatest_<today>, which sorted after the
# real recording, and six manuscript claims silently recomputed against synthetic data.
# verify_claims caught it - the numbers moved - but only because the numbers were locked
# a few hours earlier. Before that it would have rewritten the alpha validation, the
# channel-mismatch figure and the whole estimator sweep from a signal generator.
_SYNTHETIC_SOURCES = frozenset({"demo", "mock"})


def is_synthetic(session_dir: str) -> bool:
    """
    True if this session was generated rather than recorded from a headset.

    Checks the directory name first. A session whose manifest is truncated - a demo
    interrupted with Ctrl+C before the manifest was flushed - would otherwise read as
    real, which is the wrong way round to fail.
    """
    if os.path.basename(session_dir.rstrip("/\\")).startswith("DEMO_"):
        return True
    try:
        with open(os.path.join(session_dir, "events.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("type") == "manifest":
                    return event.get("sampling_rate_source") in _SYNTHETIC_SOURCES
                break
    except (OSError, ValueError):
        return False
    return False


def real_sessions(pattern: str) -> List[str]:
    """
    Session directories matching a glob, synthetic ones removed, oldest first.

    Use this instead of sorted(glob(...)) anywhere a number that reaches the manuscript
    is derived. A demo run must not be able to change a published figure by existing.
    """
    return [d for d in sorted(glob.glob(pattern))
            if os.path.isdir(d) and not is_synthetic(d)]


def newest_real_session(pattern: str) -> Optional[str]:
    """The most recent non-synthetic session matching a glob, or None."""
    found = real_sessions(pattern)
    return found[-1] if found else None


def latest_session(root: str = SESSIONS_ROOT) -> Optional[str]:
    """Most recently modified session directory, or None."""
    if not os.path.isdir(root):
        return None
    candidates: List[str] = [
        os.path.join(root, name)
        for name in os.listdir(root)
        if os.path.isfile(os.path.join(root, name, "events.jsonl"))
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
