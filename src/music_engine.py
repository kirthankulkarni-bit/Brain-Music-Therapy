"""
music_engine.py - continuous, gapless, prompt-steerable MusicGen playback.

Replaces two things: the crossfade-between-two-pre-rendered-wavs logic in the old
live_music.py, and the time.sleep(3) generation stub in stream_processing.py. The
repo previously had no real generative loop anywhere, which is the first thing a
reviewer notices when they open it.

ARCHITECTURE

    control thread            generator thread              audio callback
    (live_music.py)           (this file)                   (sounddevice)
         |                          |                              |
    set_target_prompt() ---> reads latest prompt                   |
                             generate_continuation()               |
                             from the previous tail                |
                                    |                              |
                             queue (maxsize=1) -----------> pulls float32 frames
                                                            writes to device

Two design decisions carry the closed-loop latency argument:

1. CONTINUATION, NOT INDEPENDENT CLIPS. Each segment is conditioned on the tail of
   the previous one, so the music evolves continuously instead of hard-cutting
   between two loops. Prompt changes then steer an ongoing piece rather than
   restarting it, which is what makes the adaptive/sham contrast meaningful - both
   arms hear continuous music, only the steering differs.

2. QUEUE DEPTH IS 1 BY DEFAULT, AND THAT IS A LATENCY DECISION, NOT AN OVERSIGHT.
   This is bottleneck #2. Once a segment is queued it must play to completion, so
   every extra unit of queue depth adds one full segment_seconds to the worst-case
   time between a brain-state change and the audio responding to it. Depth 1 gives
   worst-case (1 x segment) of commitment; depth 2 gives (2 x segment). Raise it
   only if you measure underruns. `worst_case_audio_latency_s` reports the number.

MEASURED CONSTRAINT, 2026-08-14, GTX 1650 Ti (4 GB, no tensor cores):

    fp32  4 s segment -> 25.1 s median   (6.3x realtime)
    fp32  8 s segment -> 42.0 s median   (5.2x realtime)
    fp16  4 s segment -> 18.0 s median   (4.5x realtime)
    fp16  8 s segment -> 48.0 s median   (6.0x realtime)

Generation is roughly 5x SLOWER than realtime on this hardware, not faster. The
architecture below assumes generation hides behind playback; on this GPU it does
not, and the queue starves permanently - about 8 s of audio per 48 s of wall time,
a 17% duty cycle with silence in between.

Note also that fp16 is not reliably faster here. The GTX 16-series has no tensor
cores, so autocast pays casting overhead without the matmul speedup an RTX card
would give it. This is consistent with the June half_precision_test result
(17.81 s -> 13.74 s, only 1.3x).

Live generation therefore needs either a much faster model/GPU, or a precomputed
segment library selected at runtime. Re-measure on any new machine with
benchmarks/latency_probe.py before assuming the streaming path is viable.

Set mock=True to run the whole pipeline with procedurally synthesized pads and no
GPU, model download, or audiocraft install. Use it to debug the control loop.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Callable, Deque, Optional

import numpy as np

# ------------------------------------------------------------------ prompting

# Five graded energy levels replacing the old binary ambient/focus switch. Binary
# states needed hysteresis to stop chattering; graded levels do not, because a
# small change in z produces a small change in the prompt instead of a hard flip.
#
# Index 0 is the sparsest, index 4 the most energetic. One rung corresponds to
# roughly one SD of the participant's own baseline arousal, so rung 2 is where
# they sit at rest.
_ENERGY_LADDER = [
    "sparse ambient drone, single sustained pad, very slow, no percussion, 50 bpm",
    "calm ambient piano with soft strings, gentle and spacious, no drums, 60 bpm",
    "warm downtempo ambient with a soft pulse, light texture, 70 bpm",
    "flowing melodic electronica, steady gentle rhythm, 85 bpm",
    "bright rhythmic electronic music, clear driving pulse, 100 bpm",
]

_LADDER_CENTRE = 2  # rung corresponding to z = 0, the participant's own baseline
_DEADBAND_Z = 0.35  # within this much of target, stop steering


def state_rung(z: float) -> int:
    """Where the participant currently sits on the energy ladder. 1 rung ~ 1 SD."""
    return int(np.clip(round(_LADDER_CENTRE + z), 0, len(_ENERGY_LADDER) - 1))


def build_prompt(z: float, target_z: float = -1.0, trend: Optional[float] = None) -> str:
    """
    Map the normalized arousal index onto a graded musical prompt.

    z         : current log(beta/alpha), z-scored against this participant's own baseline
    target_z  : desired z (-1.0 = one SD below their resting baseline, i.e. relaxation)
    trend     : z(t) - z(t-1), used to decide whether to hold or push

    THE ISO-PRINCIPLE. The prompt does not jump straight to the target state. It
    meets the participant roughly where they are and moves ONE rung toward the
    target. Dropping a highly aroused listener straight into a sparse drone is the
    classic failure mode - it reads as a jarring mismatch and arousal goes up, not
    down. Matching first and leading gradually is standard music-therapy practice
    and is what makes the graded ladder worth having over a binary switch.

    The target is a STATE, not a direction: a participant who overshoots to z = -3
    with a target of -1 gets moved back UP the ladder. That is deliberate, and it
    is what makes "time in band" a meaningful outcome rather than a one-way ratchet.

    Because the mapping is symmetric in z, the same function serves a relaxation
    arm (target -1.0) and a focus arm (target +1.0) with no branching.
    """
    if not math.isfinite(z):
        return _ENERGY_LADDER[_LADDER_CENTRE]

    here = state_rung(z)
    goal = state_rung(target_z)
    error = z - target_z

    if abs(error) <= _DEADBAND_Z or here == goal:
        level = goal
    else:
        level = here + (1 if goal > here else -1)  # lead by exactly one rung
    level = int(np.clip(level, 0, len(_ENERGY_LADDER) - 1))

    base = _ENERGY_LADDER[level]

    if trend is None or not math.isfinite(trend) or abs(trend) < 0.05:
        return base

    # trend is dz/step; moving toward the target means error is shrinking.
    moving_toward_target = (trend < 0) if error > 0 else (trend > 0)
    if moving_toward_target:
        return base + ", holding steady, minimal variation"
    return base + (", softer and slower, receding" if error > 0 else ", gradually more present")


# -------------------------------------------------------------------- config


@dataclass
class MusicConfig:
    segment_seconds: float = 8.0
    continuation_seconds: float = 2.0   # tail fed back as the continuation prompt
    crossfade_seconds: float = 0.05     # anti-click seam only; continuation handles musicality
    queue_depth: int = 1                # see module docstring - this is a latency knob
    model_name: str = "facebook/musicgen-small"
    use_fp16: bool = True
    top_k: int = 250
    cfg_coef: float = 3.0
    output_gain: float = 0.8
    mock: bool = False
    mock_sample_rate: int = 32000
    mock_generation_seconds: float = 1.0  # pretend-compute, to exercise the queue timing

    # Amplitude envelope logged per segment at this rate, for the lagged
    # audio-neural coupling index in analyze_session.py. Storing the envelope
    # instead of the audio keeps sessions small (20 Hz x 8 s = 160 floats per
    # segment) while preserving everything the coupling analysis needs.
    envelope_rate_hz: float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)


class StreamingMusicEngine:
    """
    Start it, call set_target_prompt() whenever the control loop has a new prompt,
    stop() at the end. Everything else is internal.

    on_segment(info: dict) fires once per generated segment, on the generator
    thread - wire it to SessionLogger.log_audio_segment.
    """

    def __init__(
        self,
        config: Optional[MusicConfig] = None,
        initial_prompt: str = _ENERGY_LADDER[_LADDER_CENTRE],
        on_segment: Optional[Callable[[dict], None]] = None,
    ):
        self.cfg = config or MusicConfig()
        self.on_segment = on_segment

        self._prompt_lock = threading.Lock()
        self._target_prompt = initial_prompt
        self._running = threading.Event()
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max(1, self.cfg.queue_depth))

        self._gen_thread: Optional[threading.Thread] = None
        self._stream = None
        self._model = None
        self.sample_rate: int = self.cfg.mock_sample_rate

        # Audio callback state
        self._play_buffer: Deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._tail: Optional[np.ndarray] = None  # continuation prompt for the next segment

        # Metrics for the results section
        self.segment_index = 0
        self.generation_times: list[float] = []
        self.underruns = 0
        self.started_at: Optional[float] = None
        self._playback_started = False

    # --------------------------------------------------------------- control

    @property
    def worst_case_audio_latency_s(self) -> float:
        """Worst-case delay from a prompt change to it being audible."""
        return self.cfg.segment_seconds * self.cfg.queue_depth

    def set_target_prompt(self, prompt: str) -> bool:
        """Returns True if the prompt actually changed. Cheap, safe to call every hop."""
        with self._prompt_lock:
            if prompt == self._target_prompt:
                return False
            self._target_prompt = prompt
            return True

    def get_target_prompt(self) -> str:
        with self._prompt_lock:
            return self._target_prompt

    def start(self) -> None:
        if self._running.is_set():
            return
        self._load_model()
        self._open_output_stream()
        self._running.set()
        self.started_at = time.time()
        self._gen_thread = threading.Thread(target=self._generator_loop, name="musicgen", daemon=True)
        self._gen_thread.start()

    def stop(self, timeout: float = 20.0) -> None:
        self._running.clear()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=timeout)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001 - device teardown must never mask a real error
                pass
            self._stream = None

    def stats(self) -> dict:
        times = np.asarray(self.generation_times, dtype=float)
        return {
            "segments_generated": int(self.segment_index),
            "underruns": int(self.underruns),
            "median_generation_s": float(np.median(times)) if times.size else float("nan"),
            "p95_generation_s": float(np.percentile(times, 95)) if times.size else float("nan"),
            "median_realtime_factor": float(np.median(times) / self.cfg.segment_seconds)
            if times.size
            else float("nan"),
            "worst_case_audio_latency_s": self.worst_case_audio_latency_s,
        }

    # ------------------------------------------------------------ generation

    def _load_model(self) -> None:
        if self.cfg.mock:
            self.sample_rate = self.cfg.mock_sample_rate
            print(f"[music] MOCK mode - synthesized pads at {self.sample_rate} Hz, no model loaded")
            return

        import torch  # noqa: PLC0415 - deliberately lazy so mock mode needs no torch
        from audiocraft.models import MusicGen  # noqa: PLC0415

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._device == "cpu":
            print("[music] WARNING: no CUDA device. musicgen-small on CPU is far slower than realtime.")

        t0 = time.time()
        self._model = MusicGen.get_pretrained(self.cfg.model_name, device=self._device)
        self.sample_rate = int(self._model.sample_rate)
        print(f"[music] loaded {self.cfg.model_name} on {self._device} in {time.time() - t0:.1f} s")

    def _generate_segment(self, prompt: str) -> tuple[np.ndarray, float]:
        """Returns (mono float32 audio of segment_seconds, generation wall time)."""
        t0 = time.time()

        if self.cfg.mock:
            audio = self._synthesize_mock(prompt)
            elapsed_target = self.cfg.mock_generation_seconds
            time.sleep(max(0.0, elapsed_target - (time.time() - t0)))
            return audio, time.time() - t0

        torch = self._torch
        total = self.cfg.segment_seconds
        tail = self._tail

        self._model.set_generation_params(
            duration=(total + self.cfg.continuation_seconds) if tail is not None else total,
            top_k=self.cfg.top_k,
            cfg_coef=self.cfg.cfg_coef,
        )

        autocast = (
            torch.amp.autocast(device_type=self._device, dtype=torch.float16)
            if (self.cfg.use_fp16 and self._device == "cuda")
            else _NullContext()
        )

        with torch.inference_mode():
            with autocast:
                if tail is None:
                    wav = self._model.generate([prompt], progress=False)
                else:
                    prompt_wav = torch.from_numpy(tail).to(self._device).reshape(1, 1, -1)
                    wav = self._model.generate_continuation(
                        prompt_wav,
                        prompt_sample_rate=self.sample_rate,
                        descriptions=[prompt],
                        progress=False,
                    )

        audio = wav[0, 0].detach().cpu().float().numpy()
        if tail is not None:
            # generate_continuation returns the prompt followed by the new audio;
            # keep only the new part so segments concatenate without repeating.
            audio = audio[len(tail):]

        want = int(self.cfg.segment_seconds * self.sample_rate)
        audio = audio[:want] if audio.size >= want else np.pad(audio, (0, want - audio.size))
        return audio.astype(np.float32), time.time() - t0

    def _generator_loop(self) -> None:
        while self._running.is_set():
            prompt = self.get_target_prompt()
            try:
                audio, gen_s = self._generate_segment(prompt)
            except Exception as exc:  # noqa: BLE001 - a failed segment must not kill the session
                print(f"[music] generation failed ({exc}); emitting silence for this segment")
                audio = np.zeros(int(self.cfg.segment_seconds * self.sample_rate), dtype=np.float32)
                gen_s = float("nan")

            audio = self._apply_seam_fades(audio) * self.cfg.output_gain
            n_tail = int(self.cfg.continuation_seconds * self.sample_rate)
            self._tail = audio[-n_tail:].copy() if audio.size >= n_tail else audio.copy()

            self.segment_index += 1
            if math.isfinite(gen_s):
                self.generation_times.append(gen_s)

            if self.on_segment is not None:
                try:
                    self.on_segment(
                        {
                            "prompt": prompt,
                            "duration_s": self.cfg.segment_seconds,
                            "generation_s": gen_s,
                            "queue_depth": self._queue.qsize(),
                            "segment_index": self.segment_index,
                            "underruns": self.underruns,
                            "envelope": self._envelope(audio),
                            "envelope_rate_hz": self.cfg.envelope_rate_hz,
                        }
                    )
                except Exception:  # noqa: BLE001 - logging must never stop the audio
                    pass

            # Blocking put with a timeout is the back-pressure: when the queue is
            # full, generation waits instead of running ahead and committing the
            # session to stale prompts.
            while self._running.is_set():
                try:
                    self._queue.put(audio, timeout=0.5)
                    break
                except queue.Full:
                    continue

    # -------------------------------------------------------------- playback

    def _open_output_stream(self) -> None:
        import sounddevice as sd  # noqa: PLC0415

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )
        self._stream.start()

    def _audio_callback(self, outdata, frames, time_info, status) -> None:  # noqa: ARG002
        """Real-time thread. No allocation-heavy work, no locks held long, no I/O."""
        needed = frames
        out = np.zeros(frames, dtype=np.float32)
        filled = 0

        with self._buffer_lock:
            while filled < needed and self._play_buffer:
                block = self._play_buffer[0]
                take = min(needed - filled, block.size)
                out[filled: filled + take] = block[:take]
                filled += take
                if take == block.size:
                    self._play_buffer.popleft()
                else:
                    self._play_buffer[0] = block[take:]

        if filled < needed:
            # Buffer ran dry: pull the next segment without blocking the callback.
            try:
                segment = self._queue.get_nowait()
            except queue.Empty:
                segment = None
            if segment is not None:
                take = min(needed - filled, segment.size)
                out[filled: filled + take] = segment[:take]
                if take < segment.size:
                    with self._buffer_lock:
                        self._play_buffer.append(segment[take:])
                filled += take
                self._playback_started = True
            elif self._playback_started:
                # Only count as an underrun once audio has actually begun. The
                # silence before the first segment finishes generating is startup
                # latency, not a dropout, and conflating the two makes the
                # underrun count useless as a buffer-health metric.
                self.underruns += 1

        outdata[:, 0] = out

    def _apply_seam_fades(self, audio: np.ndarray) -> np.ndarray:
        n = int(self.cfg.crossfade_seconds * self.sample_rate)
        if n <= 1 or audio.size < 2 * n:
            return audio
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio = audio.copy()
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
        return audio

    def _envelope(self, audio: np.ndarray) -> list[float]:
        """
        Amplitude envelope decimated to envelope_rate_hz.

        Rectify-and-box-average rather than Hilbert: over a whole segment the two
        agree closely at 20 Hz output, and this costs nothing on the generator
        thread. The 0.5-4 Hz band-pass that the coupling analysis needs is applied
        later, in analyze_session.py, on the stitched envelope - filtering per
        segment would corrupt the seams.
        """
        step = max(1, int(self.sample_rate / self.cfg.envelope_rate_hz))
        n = (audio.size // step) * step
        if n == 0:
            return []
        return np.abs(audio[:n]).reshape(-1, step).mean(axis=1).astype(float).tolist()

    def _synthesize_mock(self, prompt: str) -> np.ndarray:
        """Deterministic pad whose brightness tracks the prompt's ladder position."""
        sr = self.sample_rate
        n = int(self.cfg.segment_seconds * sr)
        t = np.arange(n, dtype=np.float32) / sr
        level = next((i for i, text in enumerate(_ENERGY_LADDER) if text in prompt), 2)
        root = 110.0 * (2 ** (level / 12.0))
        audio = np.zeros(n, dtype=np.float32)
        for harmonic, weight in ((1, 0.5), (2, 0.25), (3, 0.12 * level / 4.0)):
            audio += weight * np.sin(2 * np.pi * root * harmonic * t).astype(np.float32)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * (0.5 + 0.3 * level) * t)).astype(np.float32)
        return (audio * (0.6 + 0.4 * envelope)).astype(np.float32)


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    # Smoke test: python src/music_engine.py  (mock mode, no GPU needed)
    import argparse

    parser = argparse.ArgumentParser(description="Music engine smoke test")
    parser.add_argument("--real", action="store_true", help="use MusicGen instead of mock pads")
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    engine = StreamingMusicEngine(
        MusicConfig(mock=not args.real),
        on_segment=lambda info: print(
            f"[seg {info['segment_index']}] gen={info['generation_s']:.2f}s "
            f"queue={info['queue_depth']} :: {info['prompt'][:60]}"
        ),
    )
    engine.start()
    try:
        deadline = time.time() + args.seconds
        z = 1.5
        while time.time() < deadline:
            time.sleep(2.0)
            z -= 0.25  # simulate a participant relaxing toward target
            engine.set_target_prompt(build_prompt(z, target_z=-1.0, trend=-0.25))
            print(f"  z={z:+.2f} -> {engine.get_target_prompt()[:60]}")
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print(engine.stats())
