"""
live_music.py - closed-loop session orchestrator.

This file used to do five jobs: LSL ingest, DSP, a binary state machine, audio
playback, and the Qt dashboard. Four of those moved out:

    DSP + smoothing + normalization  ->  eeg_features.py
    generation + playback            ->  music_engine.py
    logging                          ->  session_logger.py
    sampling rate discovery          ->  stream_utils.get_inlet()

What is left here is the control loop and the dashboard. Three behavioural changes
are deliberate and worth stating plainly:

1. THE BASELINE PHASE IS MANDATORY. 120 s of eyes-open rest before any audio plays.
   Without it there are no z-scores, absolute beta/alpha values are not comparable
   across people, and participants cannot be pooled. A failed baseline aborts the
   session on purpose - it means bad electrode contact, and catching that before
   the intervention is the whole point.

2. HYSTERESIS IS GONE. The 0.35/0.55 thresholds existed to debounce a binary
   ambient/focus switch, and they were tuned on a signal that was mostly 60 Hz
   mains hum. Graded prompts need no debouncing, and the exponential smoother
   already suppresses blink-driven jumps. Any deadband that remains is defined in
   z units, so it transfers across participants.

3. INVALID WINDOWS DO NOT UPDATE THE SMOOTHER. The old loop had no concept of
   validity, so one blink propagated through five subsequent chunks of the moving
   average. Rejected windows are logged (rejection rate is a reportable signal
   quality metric) and then skipped.

Usage:
    python src/live_music.py --participant P01
    python src/live_music.py --participant P01 --duration 10 --target -1.0
    python src/live_music.py --mock --baseline-seconds 20 --duration 2   # no headset needed
    python src/live_music.py --participant P02 --condition sham --yoke-from sessions/P01_2026...
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eeg_features import (  # noqa: E402
    BaselineNormalizer,
    ExponentialSmoother,
    FeatureConfig,
    FeatureExtractor,
    latency_budget,
)
from library_engine import LibraryConfig, LibraryMusicEngine  # noqa: E402
from music_engine import MusicConfig, StreamingMusicEngine, build_prompt  # noqa: E402
from session_logger import SessionLogger, load_session  # noqa: E402
from stream_utils import get_inlet  # noqa: E402

PLOT_POINTS = 120  # at a 1 s hop this is a 2-minute rolling view

# Hops of z used for the least-squares trend estimate. 20 s at the default 1 s hop.
_TREND_WINDOW_HOPS = 20


class SessionState:
    """Shared between the EEG worker and the Qt dashboard. Writes are single-producer."""

    def __init__(self, target_z: float):
        self.running = True
        self.phase = "connecting"
        self.target_z = target_z
        self.z = float("nan")
        self.smoothed = float("nan")
        self.prompt = ""
        self.windows_total = 0
        self.windows_rejected = 0
        self.last_reject_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self.baseline_progress = 0.0
        self.z_history = collections.deque([float("nan")] * PLOT_POINTS, maxlen=PLOT_POINTS)
        self.reject_history = collections.deque([0.0] * PLOT_POINTS, maxlen=PLOT_POINTS)

    @property
    def rejection_rate(self) -> float:
        return self.windows_rejected / self.windows_total if self.windows_total else 0.0


# --------------------------------------------------------------------- worker


class MockInlet:
    """Synthetic 256 Hz Muse-like stream so the control loop can be tested offline."""

    def __init__(self, sampling_rate: float = 256.0):
        self.sampling_rate = sampling_rate
        self._t = 0.0
        self._last = time.time()
        self._rng = np.random.default_rng(7)

    def pull_chunk(self, timeout: float = 0.0, max_samples: int = 256):  # noqa: ARG002
        now = time.time()
        n = int((now - self._last) * self.sampling_rate)
        if n <= 0:
            time.sleep(0.01)
            return [], []
        n = min(n, max_samples)
        self._last += n / self.sampling_rate
        t = self._t + np.arange(n) / self.sampling_rate
        self._t = t[-1] + 1.0 / self.sampling_rate
        # Alpha slowly rises over the session, so log(beta/alpha) drifts downward
        # toward a relaxation target - enough motion to exercise the prompt ladder.
        alpha_gain = 10.0 + 8.0 * min(1.0, self._t / 300.0)
        samples = []
        for tt in t:
            alpha = alpha_gain * np.sin(2 * np.pi * 10.0 * tt)
            beta = 5.0 * np.sin(2 * np.pi * 20.0 * tt)
            noise = self._rng.normal(0, 6, 4)
            samples.append(list(alpha + beta + noise))
        return samples, list(t + 1.0e9)


def eeg_worker(args, state: SessionState, logger: SessionLogger) -> None:
    """Runs the whole session: connect, baseline, intervention, teardown."""
    if args.mock:
        inlet, sampling_rate = MockInlet(), 256.0
        print("[eeg] MOCK inlet - synthetic 256 Hz signal, no headset required")
    else:
        inlet, sampling_rate = get_inlet()
        if inlet is None:
            state.phase = "no stream"
            state.running = False
            logger.note("no LSL stream found", level="error")
            return

    pair = tuple(c.strip().upper() for c in args.channels.split(","))
    cfg = FeatureConfig(
        sampling_rate=sampling_rate,
        window_seconds=args.window,
        hop_seconds=args.hop,
        frontal_channels=pair,
        reject_peak_to_peak_uv=args.reject_p2p,
    )
    extractor = FeatureExtractor(cfg)
    smoother = ExponentialSmoother(hop_seconds=cfg.hop_seconds, tau_seconds=args.tau)
    normalizer = BaselineNormalizer(min_windows=max(10, int(args.baseline_seconds * 0.25 / cfg.hop_seconds)))
    budget = latency_budget(cfg, args.tau)

    music_cfg = MusicConfig(mock=args.mock_audio or args.mock, segment_seconds=args.segment)

    # Resolve the engine once, here, so the manifest records what actually ran rather
    # than what was requested. The mock flags describe the EEG source AND the audio
    # source, but the library needs no GPU, so an explicit --engine library stays
    # honoured under --mock - otherwise the only realtime audio path would be
    # untestable without a headset.
    engine_kind = args.engine or ("streaming" if (args.mock or args.mock_audio) else "library")
    logger.write_manifest(
        sampling_rate=sampling_rate,
        sampling_rate_source="mock" if args.mock else "lsl_nominal_srate",
        feature_config=cfg.to_dict(),
        music_config=music_cfg.to_dict(),
        # Which engine produced the audio changes the closed-loop latency by 8x, so
        # it belongs in the manifest beside the analysis budget. Without it, two
        # sessions with identical feature configs are not actually comparable.
        engine=engine_kind,
        engine_config=(
            LibraryConfig(library_dir=args.library, crossfade_seconds=args.crossfade).to_dict()
            if engine_kind == "library" else music_cfg.to_dict()
        ),
        smoother_tau_s=args.tau,
        target_z=args.target,
        baseline_seconds=args.baseline_seconds,
        intervention_seconds=args.duration * 60.0,
        latency_budget=budget,
        yoked_from=args.yoke_from,
        code_version="v2-sample-rate-corrected",
        index_channels=list(pair),
    )

    print(f"\n[eeg] {sampling_rate:g} Hz, {cfg.window_seconds:g} s window, {cfg.hop_seconds:g} s hop")
    print(f"[eeg] analysis-path latency budget: {budget['total_analysis_latency_s']:.2f} s")

    buffers = [collections.deque(maxlen=cfg.window_samples) for _ in range(len(cfg.channels))]

    def pump() -> None:
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=512)
        if not chunk:
            return
        for sample in chunk:
            for i in range(len(buffers)):
                buffers[i].append(sample[i])
        logger.log_raw(timestamps, chunk)

    def next_window() -> Optional[np.ndarray]:
        if len(buffers[0]) < cfg.window_samples:
            return None
        return np.asarray([list(b) for b in buffers], dtype=np.float64)

    # ------------------------------------------------------- PHASE 1: baseline

    state.phase = "baseline"
    print(f"\n[eeg] BASELINE: {args.baseline_seconds:.0f} s, eyes open, sit still, no audio.")
    t_start = time.time()
    next_hop = t_start + cfg.window_seconds

    while state.running and (time.time() - t_start) < args.baseline_seconds:
        pump()
        now = time.time()
        if now < next_hop:
            time.sleep(0.01)
            continue
        next_hop += cfg.hop_seconds

        window = next_window()
        if window is None:
            continue

        feats = extractor.extract(window, timestamp=now)
        state.windows_total += 1
        state.baseline_progress = min(1.0, (now - t_start) / args.baseline_seconds)

        if feats.valid:
            normalizer.add(feats.log_beta_alpha)
        else:
            state.windows_rejected += 1
            state.last_reject_reason = feats.reject_reason

        logger.log_window("baseline", feats, baseline_n=normalizer.n)
        print(
            f"  baseline {now - t_start:5.1f}/{args.baseline_seconds:.0f}s  "
            f"valid={normalizer.n:3d}  rejected={state.windows_rejected:3d}  "
            f"{'' if feats.valid else '<- ' + str(feats.reject_reason)}",
            end="\r" if feats.valid else "\n",
        )

    if not state.running:
        return

    try:
        mean, sd = normalizer.finalize()
    except RuntimeError as exc:
        print(f"\n\n[eeg] BASELINE FAILED: {exc}")
        logger.note(f"baseline failed: {exc}", level="error")
        state.phase = "baseline failed"
        state.running = False
        return

    logger.log_baseline(**normalizer.to_dict(), rejection_rate=state.rejection_rate)
    print(f"\n\n[eeg] baseline OK: mean={mean:+.4f}  sd={sd:.4f}  "
          f"n={normalizer.n}  rejection rate={state.rejection_rate:.1%}")

    # --------------------------------------------------- PHASE 2: intervention

    yoked_prompts = _load_yoked_prompts(args.yoke_from) if args.yoke_from else None
    if yoked_prompts:
        print(f"[eeg] SHAM (yoked): replaying {len(yoked_prompts)} prompts from {args.yoke_from}")

    initial_prompt = build_prompt(0.0, args.target, None)
    if engine_kind == "library":
        # Explicit failure, never a silent fallback to streaming. A session that
        # quietly ran the 8 s-commitment engine when the operator asked for the
        # 1.0 s one is unusable data that looks fine in the log.
        manifest_path = os.path.join(args.library, "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"[eeg] FATAL: no library at {manifest_path}")
            print("[eeg]   build it:  python scripts/build_library.py")
            print("[eeg]   or force generation:  --engine streaming")
            return 2
        engine = LibraryMusicEngine(
            LibraryConfig(library_dir=args.library, crossfade_seconds=args.crossfade),
            initial_prompt=initial_prompt,
            on_segment=lambda info: logger.log_audio_segment(**info),
        )
    else:
        engine = StreamingMusicEngine(
            music_cfg,
            initial_prompt=initial_prompt,
            on_segment=lambda info: logger.log_audio_segment(**info),
        )
    engine.start()

    state.phase = "intervention"
    print(f"\n[eeg] INTERVENTION: {args.duration:.0f} min, target z = {args.target:+.1f}")
    print(f"[eeg] engine: {engine_kind}")
    print(f"[eeg] worst-case audio commitment: {engine.worst_case_audio_latency_s:.1f} s\n")

    t_start = time.time()
    next_hop = t_start
    previous_z: Optional[float] = None
    trend_history: collections.deque = collections.deque(maxlen=_TREND_WINDOW_HOPS)
    smoother.reset()

    try:
        while state.running and (time.time() - t_start) < args.duration * 60.0:
            pump()
            now = time.time()
            if now < next_hop:
                time.sleep(0.01)
                continue
            next_hop += cfg.hop_seconds

            window = next_window()
            if window is None:
                continue

            feats = extractor.extract(window, timestamp=now)
            state.windows_total += 1

            if not feats.valid:
                # Do NOT touch the smoother. An artifact is missing information,
                # not a measurement of zero.
                state.windows_rejected += 1
                state.last_reject_reason = feats.reject_reason
                state.reject_history.append(1.0)
                state.z_history.append(state.z)
                logger.log_window("intervention", feats, prompt=engine.get_target_prompt(), applied=False)
                continue

            smoothed = smoother.update(feats.log_beta_alpha)
            z = normalizer.normalize(smoothed)

            # Trend as a least-squares slope over a window, not a one-hop difference.
            # PILOT01 measured the one-hop estimator's sd at 0.275 z-units per hop
            # against a genuine drift of 0.00088 - it was almost entirely noise, and
            # it drove 491 prompt changes in 20 minutes. A 20-hop regression cuts the
            # estimator noise to 0.068. That is still well above the drift, which is
            # why build_prompt also applies hysteresis rather than trusting this.
            trend_history.append(z)
            trend = None
            if len(trend_history) >= _TREND_WINDOW_HOPS:
                window = np.asarray(trend_history, dtype=float)
                trend = float(np.polyfit(np.arange(window.size, dtype=float), window, 1)[0])
            previous_z = z

            if yoked_prompts is not None:
                elapsed = now - t_start
                prompt = _yoked_prompt_at(yoked_prompts, elapsed)
            else:
                # Previous prompt threaded through so build_prompt can apply
                # hysteresis while staying a pure function.
                prompt = build_prompt(z, target_z=args.target, trend=trend,
                                      previous_prompt=engine.get_target_prompt())

            changed = engine.set_target_prompt(prompt)

            state.smoothed = smoothed
            state.z = z
            state.prompt = prompt
            state.z_history.append(z)
            state.reject_history.append(0.0)

            logger.log_window(
                "intervention",
                feats,
                smoothed_log_beta_alpha=smoothed,
                z=z,
                trend=trend,
                target_z=args.target,
                prompt=prompt,
                prompt_changed=changed,
                applied=True,
                in_target_band=bool(abs(z - args.target) <= 0.5),
            )

            print(
                f"  {now - t_start:6.1f}s  z={z:+6.2f}  (target {args.target:+.1f})  "
                f"rej={state.rejection_rate:5.1%}  {'* ' if changed else '  '}{prompt[:52]}",
                end="\r",
            )
    except BaseException as exc:  # noqa: BLE001 - re-raised below, see comment
        # A crash in this worker used to be invisible. The thread died, the finally
        # block still wrote "session complete", and the result was a session
        # directory that looked successful while containing no intervention data at
        # all. A NameError introduced on 8/16 did exactly that: baseline logged
        # normally, the loop died on its first intervention window, and the manifest
        # and completion note were written as if the run had finished.
        #
        # With a participant in the chair that is the worst possible failure mode -
        # you would not find out until analysis, by which point the session cannot
        # be repeated. Record the failure in the session itself, and re-raise so it
        # reaches the console instead of being swallowed by the thread.
        failure = f"{type(exc).__name__}: {exc}"
        logger.note("session FAILED", level="error", error=failure,
                    phase=state.phase, windows_total=state.windows_total)
        state.last_error = failure
        print(f"\n\n[eeg] SESSION FAILED during {state.phase}: {failure}")
        raise
    finally:
        stats = engine.stats()
        engine.stop()
        if state.last_error is None:
            logger.note("session complete", level="info", engine_stats=stats,
                        rejection_rate=state.rejection_rate, windows_total=state.windows_total)
            print("\n\n[eeg] engine stats:", stats)
        print(f"[eeg] session written to {logger.dir}")
        state.phase = "done" if state.last_error is None else "failed"
        state.running = False


def _load_yoked_prompts(session_dir: str) -> list[tuple[float, str]]:
    """
    Yoked sham: replay a prior adaptive session's prompt schedule on its original
    timeline, ignoring the current participant's EEG. Every acoustic property is
    matched; only the contingency between brain and music is broken. That contrast
    is what rules out regression to the mean, which a single-arm design cannot.

    KNOWN FIDELITY LIMIT, measured rather than assumed. Offsets are normalized
    against the FIRST AUDIO EVENT, but an audio event is logged when a segment is
    produced, not when the prompt that chose it was set. Those differ by roughly one
    hop, so a replay reproduces the source's prompt SEQUENCE exactly while running
    about 1 s early throughout, and the source's very first prompt is skipped when it
    was superseded before the first segment was logged.

    Verified against sessions/ARTIFACT_20260816_144127 over its first 86 s: 8 source
    changes against 7 replayed, identical order, no source prompt ever missing from
    the replay, every replayed change 0.4-1.0 s early.

    That is acceptable for the acoustic-matching claim - the same prompts play in the
    same order for the same durations - and it is small against the 8 s segment
    tenure and 5.5 s analysis latency. It is recorded here because "matched for every
    acoustic property" is a claim a reviewer may probe, and the honest answer is
    "matched in sequence and duration, with a systematic sub-hop lead", not "identical".
    """
    data = load_session(session_dir)
    t0 = None
    schedule: list[tuple[float, str]] = []
    for event in data["audio"]:
        if t0 is None:
            t0 = event["elapsed_s"]
        schedule.append((event["elapsed_s"] - t0, event["prompt"]))
    return schedule


def _yoked_prompt_at(schedule: list[tuple[float, str]], elapsed: float) -> str:
    prompt = schedule[0][1]
    for offset, text in schedule:
        if offset <= elapsed:
            prompt = text
        else:
            break
    return prompt


# ------------------------------------------------------------------ dashboard


def run_dashboard(state: SessionState) -> int:
    from PyQt5 import QtCore, QtWidgets  # noqa: PLC0415
    import pyqtgraph as pg  # noqa: PLC0415

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Closed-Loop EEG Music Therapy")
            self.resize(980, 520)

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QVBoxLayout(central)

            self.status = QtWidgets.QLabel("connecting...")
            self.status.setStyleSheet("color:#dddddd; font-size:14px; padding:4px;")
            layout.addWidget(self.status)

            self.graph = pg.PlotWidget()
            self.graph.setBackground("#121212")
            layout.addWidget(self.graph)

            # Plotting z, not the raw ratio. Raw beta/alpha is not comparable across
            # people or even across sessions on the same person; z against their own
            # baseline is.
            self.graph.setLabel("left", "Arousal index, z vs own baseline", color="#cccccc", size="14pt")
            self.graph.setLabel("bottom", "Time (rolling 2 min)", color="#cccccc", size="14pt")
            self.graph.setYRange(-3, 3)
            self.graph.showGrid(x=True, y=True, alpha=0.2)

            x = list(range(PLOT_POINTS))
            self.line = self.graph.plot(x, [0.0] * PLOT_POINTS, pen=pg.mkPen("#00d2ff", width=3))
            self.rejects = self.graph.plot(
                [], [], pen=None, symbol="x", symbolSize=8, symbolBrush="#ff3333"
            )

            target = pg.InfiniteLine(
                pos=state.target_z, angle=0,
                pen=pg.mkPen(color="#00ff9d", width=2, style=QtCore.Qt.DashLine),
                label=f"target z={state.target_z:+.1f}",
            )
            self.graph.addItem(target)
            self.graph.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#666666", width=1)))

            self.prompt_label = QtWidgets.QLabel("")
            self.prompt_label.setWordWrap(True)
            self.prompt_label.setStyleSheet("color:#00d2ff; font-size:13px; padding:4px;")
            layout.addWidget(self.prompt_label)

            self.timer = QtCore.QTimer()
            self.timer.setInterval(200)
            self.timer.timeout.connect(self.refresh)
            self.timer.start()

        def refresh(self):
            y = list(state.z_history)
            x = list(range(len(y)))
            self.line.setData(x, [0.0 if not np.isfinite(v) else v for v in y])

            flags = list(state.reject_history)
            rx = [i for i, flag in enumerate(flags) if flag > 0]
            self.rejects.setData(rx, [2.8] * len(rx))

            if state.phase == "baseline":
                head = f"BASELINE  {state.baseline_progress:.0%}  (no audio yet)"
            elif state.phase == "intervention":
                head = f"INTERVENTION  z={state.z:+.2f}  target={state.target_z:+.1f}"
            else:
                head = state.phase.upper()
            self.status.setText(
                f"{head}   |   windows {state.windows_total}   "
                f"rejected {state.windows_rejected} ({state.rejection_rate:.1%})"
                + (f"   last: {state.last_reject_reason}" if state.last_reject_reason else "")
            )
            self.prompt_label.setText(state.prompt)
            if not state.running and state.phase in ("done", "baseline failed", "no stream"):
                self.timer.stop()

        def closeEvent(self, event):  # noqa: N802 - Qt naming
            state.running = False
            event.accept()

    app = QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    return app.exec_()


# ----------------------------------------------------------------------- main


def parse_args():
    p = argparse.ArgumentParser(description="Closed-loop EEG-driven music therapy session")
    p.add_argument("--participant", default="self")
    p.add_argument("--condition", default="adaptive", choices=["adaptive", "sham", "pilot"])
    p.add_argument("--target", type=float, default=-1.0,
                   help="target z of log(beta/alpha); -1.0 = one SD below own baseline (relaxation)")
    p.add_argument("--baseline-seconds", type=float, default=120.0)
    p.add_argument("--duration", type=float, default=10.0, help="intervention length, minutes")
    p.add_argument("--channels", default="AF7,AF8",
                   help="electrode pair the arousal index is computed from, e.g. TP9,TP10. "
                        "Frontal (AF7,AF8) is the default and the pre-registered choice; "
                        "temporal (TP9,TP10) is the fallback when frontal contact will not hold.")
    p.add_argument("--reject-p2p", type=float, default=350.0,
                             help="artifact rejection threshold, uV peak-to-peak per window. "
                             "Derived from measured blink amplitudes; see FeatureConfig.")
    p.add_argument("--window", type=float, default=4.0, help="analysis window, seconds")
    p.add_argument("--hop", type=float, default=1.0, help="hop between windows, seconds")
    p.add_argument("--tau", type=float, default=3.0, help="smoother time constant, seconds")
    p.add_argument("--segment", type=float, default=8.0, help="audio segment length, seconds")
    p.add_argument("--engine", default=None, choices=["library", "streaming"],
                   help="library selects precomputed segments (1.0 s worst-case latency); "
                        "streaming generates live (8.0 s commitment, and measured slower "
                        "than realtime on every GPU tested - see docs/results_latency.md). "
                        "Default: streaming under --mock/--mock-audio, library otherwise. "
                        "Passing it explicitly always wins, so --engine library --mock "
                        "exercises the real audio path against synthetic EEG.")
    p.add_argument("--library", default="library", help="library directory for --engine library")
    p.add_argument("--crossfade", type=float, default=1.0,
                   help="crossfade seconds; this is THE latency knob for --engine library, "
                        "exactly as queue depth is for streaming")
    p.add_argument("--yoke-from", default=None, help="session dir to replay prompts from (yoked sham)")
    p.add_argument("--mock", action="store_true", help="synthetic EEG and synthetic audio, no hardware")
    p.add_argument("--mock-audio", action="store_true", help="real EEG, synthesized pads instead of MusicGen")
    p.add_argument("--headless", action="store_true", help="no Qt dashboard")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.condition == "sham" and not args.yoke_from:
        print("--condition sham requires --yoke-from <session_dir> so the sham is properly yoked.")
        return 2

    state = SessionState(target_z=args.target)
    logger = SessionLogger(participant_id=args.participant, condition=args.condition)
    print(f"[session] logging to {logger.dir}")

    worker = threading.Thread(target=eeg_worker, args=(args, state, logger), name="eeg", daemon=True)
    worker.start()

    try:
        if args.headless:
            while state.running:
                time.sleep(0.25)
        else:
            run_dashboard(state)
            state.running = False
    except KeyboardInterrupt:
        print("\n[session] interrupted, shutting down cleanly...")
        state.running = False
    finally:
        worker.join(timeout=30.0)
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
