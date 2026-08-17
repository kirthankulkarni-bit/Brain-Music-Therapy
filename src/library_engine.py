"""
library_engine.py - realtime playback from the precomputed segment library.

Drop-in replacement for StreamingMusicEngine: same start/stop/set_target_prompt/
stats surface, same on_segment callback, so live_music.py can swap engines without
knowing which one it holds.

WHY THIS IS FASTER THAN STREAMING, AND IT IS NOT JUST "NO GENERATION WAIT"

The obvious win is that selecting a segment costs microseconds where generating one
cost 9-25 s. That alone only fixes queue starvation; it does not by itself improve
closed-loop latency, because StreamingMusicEngine's real audio-side bottleneck was
never generation time. It was COMMITMENT: once a segment entered the queue it had to
play to completion, so a prompt change could take up to queue_depth x segment_seconds
(8 s at the defaults) to become audible, no matter how fast the GPU was.

A precomputed library removes the commitment, not merely the wait. Because every
segment is already in memory, playback can abandon the current one MID-SEGMENT and
crossfade into a segment of the new prompt from any position. Worst-case latency
drops from one whole segment to one crossfade:

    streaming : queue_depth x segment_seconds        = 8.0 s
    library   : crossfade_seconds                    = 1.0 s

That 8x is the actual contribution of this file. It is also why crossfade_seconds is
the latency knob here, exactly as queue_depth was there, and why it is reported by
worst_case_audio_latency_s in both engines - the number means the same thing and can
be compared directly.

CROSSFADE SHAPE

Equal power (sin/cos), not linear. Library segments are independent renders, so at a
seam the two signals are uncorrelated; linear fades sum to a perceptible dip in the
middle of the transition, equal-power fades hold constant loudness. Segments are
already RMS-matched at build time, so the fade only has to handle the shape.

The same crossfade machinery serves two different events, which is why there is only
one code path for both:

  prompt change     - abandon the current segment immediately, fade into the new one
  segment exhausted - fade into another variant of the SAME prompt before the current
                      one runs out, so playback is continuous with no gap

WHAT THIS GIVES UP RELATIVE TO STREAMING

StreamingMusicEngine used generate_continuation, conditioning each segment on the
tail of the last, so the music developed as one piece. Here, transitions are blends
between independent renders. Within a segment the audio is unchanged; across a seam
it is a crossfade rather than a musical development. That is the honest cost of the
8x latency improvement, and it is the thing to listen for when judging whether the
tradeoff was worth it.
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Optional

import numpy as np


@dataclass
class LibraryConfig:
    library_dir: str = "library"
    crossfade_seconds: float = 1.0   # THE latency knob - see module docstring
    output_gain: float = 0.8
    envelope_rate_hz: float = 20.0   # matches MusicConfig, for the coupling analysis
    random_start: bool = True        # enter new segments at a random offset, for variety
    seed: Optional[int] = None
    blocksize: int = 1024

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Segment:
    audio: np.ndarray
    file: str
    prompt: str
    rung: int
    variant: str


@dataclass
class _Playhead:
    segment: Optional[_Segment] = None
    pos: int = 0


class LibraryMusicEngine:
    """
    Realtime playback from a prebuilt library.

    engine = LibraryMusicEngine(LibraryConfig(library_dir="library"))
    engine.start()
    engine.set_target_prompt(build_prompt(z, target_z=-1.0))
    engine.stop()
    """

    def __init__(
        self,
        config: Optional[LibraryConfig] = None,
        initial_prompt: Optional[str] = None,
        on_segment: Optional[Callable[[dict], None]] = None,
    ):
        self.cfg = config or LibraryConfig()
        self.on_segment = on_segment
        self._rng = random.Random(self.cfg.seed)

        self.sample_rate, self._by_prompt, self.manifest = self._load_library()
        self._prompts = list(self._by_prompt)
        if not self._prompts:
            raise RuntimeError(
                f"No segments in {self.cfg.library_dir}. "
                "Run: python scripts/build_library.py"
            )

        self._prompt_lock = threading.Lock()
        self._target_prompt = initial_prompt or self._prompts[0]
        self._prompt_changed_at: Optional[float] = None

        self._mix_lock = threading.Lock()
        self._cur = _Playhead()
        self._nxt = _Playhead()
        self._xfade_i = 0
        self._xfade_n = max(1, int(self.cfg.crossfade_seconds * self.sample_rate))
        self._fade_out, self._fade_in = self._equal_power_ramps(self._xfade_n)

        self._running = threading.Event()
        self._stream = None

        # Metrics
        self.segments_played = 0
        self.switches = 0
        self.underruns = 0
        self.clipped_samples = 0
        self.selection_times_ms: list[float] = []
        self.switch_latencies_s: list[float] = []
        self.missing_prompts: list[str] = []
        self.started_at: Optional[float] = None
        self._envelope_accum: list[float] = []

    # ---------------------------------------------------------------- loading

    def _load_library(self) -> tuple[int, dict, dict]:
        import soundfile as sf

        manifest_path = os.path.join(self.cfg.library_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"{manifest_path} not found. Run: python scripts/build_library.py"
            )
        with open(manifest_path, encoding="utf-8") as fh:
            manifest = json.load(fh)

        sample_rate = int(manifest["sample_rate"])
        by_prompt: dict[str, list[_Segment]] = {}

        # Everything is loaded up front and held in RAM. The whole library is about
        # 80 segments x 8 s x 32 kHz float32 = ~80 MB, and keeping it resident is
        # what lets the audio callback switch segments without touching the disk.
        for entry in manifest.get("prompts", []):
            loaded = []
            for seg in entry.get("segments", []):
                path = os.path.join(self.cfg.library_dir, seg["file"])
                if not os.path.exists(path):
                    continue
                audio, sr = sf.read(path, dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != sample_rate:
                    raise ValueError(f"{path}: {sr} Hz, manifest says {sample_rate} Hz")
                loaded.append(_Segment(
                    audio=np.ascontiguousarray(audio, dtype=np.float32),
                    file=seg["file"], prompt=entry["prompt"],
                    rung=entry.get("rung", -1), variant=entry.get("variant", "?"),
                ))
            if loaded:
                by_prompt[entry["prompt"]] = loaded

        return sample_rate, by_prompt, manifest

    @staticmethod
    def _equal_power_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return np.cos(t * math.pi / 2).astype(np.float32), np.sin(t * math.pi / 2).astype(np.float32)

    # ---------------------------------------------------------------- control

    @property
    def worst_case_audio_latency_s(self) -> float:
        """
        Worst-case delay from a prompt change to it being audible.

        One crossfade. Directly comparable to StreamingMusicEngine's
        queue_depth x segment_seconds, and about 8x smaller at the defaults.
        """
        return self.cfg.crossfade_seconds

    def set_target_prompt(self, prompt: str) -> bool:
        """Returns True if the prompt actually changed. Cheap, safe to call every hop."""
        with self._prompt_lock:
            if prompt == self._target_prompt:
                return False
            self._target_prompt = prompt
            self._prompt_changed_at = time.time()
            return True

    def get_target_prompt(self) -> str:
        with self._prompt_lock:
            return self._target_prompt

    def resolve(self, prompt: str) -> list[_Segment]:
        """
        Segments for a prompt, falling back rather than failing.

        A prompt with no segments means the library is stale relative to
        music_engine.py - a rung was added, or the wording changed. Killing the
        session over that would lose the participant's data, so the nearest rung is
        substituted and the miss is recorded in stats() for the run to be judged on.
        """
        segments = self._by_prompt.get(prompt)
        if segments:
            return segments

        if prompt not in self.missing_prompts:
            self.missing_prompts.append(prompt)

        want_rung = -1
        for candidates in self._by_prompt.values():
            base = candidates[0]
            if prompt.startswith(base.prompt.split(",")[0]):
                want_rung = base.rung
                break

        best, best_dist = None, 10**6
        for candidates in self._by_prompt.values():
            dist = abs(candidates[0].rung - want_rung) if want_rung >= 0 else candidates[0].rung
            if dist < best_dist:
                best, best_dist = candidates, dist
        return best or next(iter(self._by_prompt.values()))

    def _select(self, prompt: str, avoid: Optional[str]) -> _Segment:
        """Pick a variant, avoiding an immediate repeat of the same file."""
        t0 = time.perf_counter()
        candidates = self.resolve(prompt)
        pool = [s for s in candidates if s.file != avoid] or candidates
        chosen = self._rng.choice(pool)
        self.selection_times_ms.append((time.perf_counter() - t0) * 1000.0)
        return chosen

    # ----------------------------------------------------------------- mixing

    def _begin_switch(self, reason: str) -> None:
        """Start a crossfade into a segment of the current target prompt."""
        prompt = self.get_target_prompt()
        avoid = self._cur.segment.file if self._cur.segment else None
        segment = self._select(prompt, avoid)

        start = 0
        if self.cfg.random_start and segment.audio.size > 2 * self._xfade_n:
            # Entering at a random offset stops every transition into a given prompt
            # from sounding like the same edit, which is the most obvious tell that
            # the audio is precomputed.
            start = self._rng.randrange(0, segment.audio.size - 2 * self._xfade_n)

        self._nxt = _Playhead(segment=segment, pos=start)
        self._xfade_i = 0
        self.switches += 1

        with self._prompt_lock:
            changed_at = self._prompt_changed_at
            self._prompt_changed_at = None
        if changed_at is not None:
            self.switch_latencies_s.append(time.time() - changed_at)

        if self.on_segment is not None:
            try:
                self.on_segment({
                    "prompt": prompt,
                    "file": segment.file,
                    "rung": segment.rung,
                    "variant": segment.variant,
                    "reason": reason,
                    "duration_s": segment.audio.size / self.sample_rate,
                    "generation_s": 0.0,     # precomputed; kept for schema parity
                    "segment_index": self.segments_played,
                    "start_offset_s": start / self.sample_rate,
                    "envelope_rate_hz": self.cfg.envelope_rate_hz,
                })
            except Exception:  # noqa: BLE001 - logging must never stop the audio
                pass

    def _pull(self, n: int) -> np.ndarray:
        """
        Produce n samples of the mixed stream.

        Pure with respect to the audio device, so it can be driven offline by
        render_offline() and tested with no sound card present.
        """
        out = np.zeros(n, dtype=np.float32)
        filled = 0

        while filled < n:
            if self._cur.segment is None:
                self._cur = _Playhead(segment=self._select(self.get_target_prompt(), None), pos=0)
                self.segments_played += 1

            # A prompt change abandons the current segment immediately. This is the
            # whole latency argument: no commitment to playing it out.
            if self._nxt.segment is None:
                prompt = self.get_target_prompt()
                if self._cur.segment.prompt != prompt:
                    self._begin_switch("prompt-change")
                elif self._cur.pos + self._xfade_n >= self._cur.segment.audio.size:
                    # Start the fade before the segment runs out, so playback is
                    # continuous rather than gapped.
                    self._begin_switch("segment-exhausted")

            take = min(n - filled, self._xfade_n - self._xfade_i if self._nxt.segment else n - filled)
            take = max(1, take)

            if self._nxt.segment is None:
                chunk = self._read(self._cur, take)
                out[filled:filled + chunk.size] += chunk
                filled += chunk.size
                if chunk.size < take:      # segment ended without a queued successor
                    self._cur = _Playhead()
                continue

            # Crossfading: sum the outgoing and incoming playheads under equal-power
            # ramps. Either may run short at the end of its buffer; zero-padding is
            # correct there because the ramp is already near zero for the outgoing
            # one and the incoming one has just started.
            a = self._read(self._cur, take)
            b = self._read(self._nxt, take)
            m = max(a.size, b.size)
            if m == 0:
                self._cur, self._nxt, self._xfade_i = self._nxt, _Playhead(), 0
                continue
            a = np.pad(a, (0, m - a.size))
            b = np.pad(b, (0, m - b.size))

            i = self._xfade_i
            out[filled:filled + m] += a * self._fade_out[i:i + m] + b * self._fade_in[i:i + m]
            filled += m
            self._xfade_i += m

            if self._xfade_i >= self._xfade_n:
                self._cur, self._nxt, self._xfade_i = self._nxt, _Playhead(), 0
                self.segments_played += 1

        out *= self.cfg.output_gain

        # Equal-power crossfade of two UNCORRELATED signals can peak at sqrt(2) times
        # either input - the ramps hold power constant, not peak amplitude. Segments
        # are RMS-matched, so a high-crest-factor pair can transiently exceed 1.0 even
        # though neither segment does alone. Clip rather than let the device wrap, and
        # count it: a nonzero clipped_samples means output_gain is too high for this
        # library, which is a build-time fix, not something to discover by ear.
        clipped = int(np.count_nonzero(np.abs(out) > 1.0))
        if clipped:
            self.clipped_samples += clipped
            np.clip(out, -1.0, 1.0, out=out)
        return out

    def _read(self, head: _Playhead, n: int) -> np.ndarray:
        if head.segment is None:
            return np.zeros(0, dtype=np.float32)
        chunk = head.segment.audio[head.pos:head.pos + n]
        head.pos += chunk.size
        return chunk

    # --------------------------------------------------------------- playback

    def start(self) -> None:
        if self._running.is_set():
            return
        import sounddevice as sd  # noqa: PLC0415 - offline rendering needs no device

        self._running.set()
        self.started_at = time.time()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self.cfg.blocksize, callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        try:
            with self._mix_lock:
                outdata[:, 0] = self._pull(frames)
        except Exception:  # noqa: BLE001 - never raise out of the audio thread
            outdata[:] = 0
            self.underruns += 1

    def stop(self, timeout: float = 5.0) -> None:  # noqa: ARG002 - parity with the other engine
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def render_offline(self, seconds: float,
                       control: Optional[Callable[[float], Optional[str]]] = None,
                       block: int = 4096) -> np.ndarray:
        """
        Render without an audio device.

        control(t) is called once per block with elapsed seconds and may return a new
        prompt, which makes the whole steering path testable on a machine with no
        sound card - and lets a demo WAV be produced from a scripted arousal
        trajectory for anyone who wants to hear the transitions.
        """
        total = int(seconds * self.sample_rate)
        out = np.zeros(total, dtype=np.float32)
        pos = 0
        with self._mix_lock:
            while pos < total:
                if control is not None:
                    prompt = control(pos / self.sample_rate)
                    if prompt:
                        self.set_target_prompt(prompt)
                n = min(block, total - pos)
                out[pos:pos + n] = self._pull(n)
                pos += n
        return out

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        sel = np.asarray(self.selection_times_ms, dtype=float)
        lat = np.asarray(self.switch_latencies_s, dtype=float)
        return {
            "engine": "library",
            "segments_played": int(self.segments_played),
            "switches": int(self.switches),
            "underruns": int(self.underruns),
            "clipped_samples": int(self.clipped_samples),
            "median_selection_ms": float(np.median(sel)) if sel.size else float("nan"),
            "p95_selection_ms": float(np.percentile(sel, 95)) if sel.size else float("nan"),
            "median_switch_latency_s": float(np.median(lat)) if lat.size else float("nan"),
            "p95_switch_latency_s": float(np.percentile(lat, 95)) if lat.size else float("nan"),
            "worst_case_audio_latency_s": self.worst_case_audio_latency_s,
            "library_prompts": len(self._by_prompt),
            "library_segments": sum(len(v) for v in self._by_prompt.values()),
            "missing_prompts": list(self.missing_prompts),
        }


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from music_engine import build_prompt  # noqa: E402

    parser = argparse.ArgumentParser(description="Library engine demo")
    parser.add_argument("--library", default="library")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--wav", default=None, help="render offline to this WAV instead of playing")
    args = parser.parse_args()

    engine = LibraryMusicEngine(
        LibraryConfig(library_dir=args.library, seed=0),
        on_segment=lambda i: print(f"  [{i['segment_index']:>3}] {i['reason']:<17} "
                                   f"rung {i['rung']} {i['variant']:<9} {i['file']}"),
    )
    print(f"library: {len(engine._by_prompt)} prompts, "
          f"{sum(len(v) for v in engine._by_prompt.values())} segments @ {engine.sample_rate} Hz")

    # A participant relaxing from +2 SD down to the target, the same trajectory the
    # music_engine smoke test uses, so the two engines can be compared by ear.
    def control(t: float) -> Optional[str]:
        z = 2.0 - 0.05 * t
        return build_prompt(z, target_z=-1.0, trend=-0.05)

    if args.wav:
        import soundfile as sf
        audio = engine.render_offline(args.seconds, control=control)
        sf.write(args.wav, audio, engine.sample_rate, subtype="PCM_16")
        print(f"wrote {args.wav}")
    else:
        engine.start()
        try:
            deadline = time.time() + args.seconds
            while time.time() < deadline:
                time.sleep(1.0)
                engine.set_target_prompt(control(time.time() - engine.started_at))
        except KeyboardInterrupt:
            pass
        finally:
            engine.stop()

    print(json.dumps(engine.stats(), indent=2))
