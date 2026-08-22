"""
analyze_session.py - session metrics from the JSONL log.

The previous version read the CSV and multiplied the row count by 2 to get elapsed
time. That assumption breaks the moment a window is rejected or the loop stalls,
and it silently under- or over-reports every duration derived from it. Everything
here uses real timestamps.

Reported metrics, which double as the study's dependent variables:

  time in band          fraction of intervention windows with z inside the target band
  time to target        seconds until z first crosses target and holds for 30 s
  rejection rate        fraction of windows dropped for artifacts (signal quality;
                        reviewers will ask, and it must be reported for both arms)
  generation latency    median and p95 MusicGen wall time per segment
  ACI                   lagged audio-neural coupling index (see below)

THE COUPLING INDEX

Cross-correlate the generated audio's amplitude envelope against the alpha power
envelope across a range of lags, and report the peak correlation and the lag where
it occurs. In the adaptive arm, coupling should peak at a positive lag with music
leading brain. In a yoked sham matched for every acoustic property, it should
collapse - and that contrast cannot be produced by regression to the mean, which a
single-arm design cannot rule out. The lag is also a latency measurement taken on
the brain rather than on the clock.

Two honest caveats are enforced in the code rather than buried in a footnote:

  1. Both series are sampled on the analysis hop grid (1 Hz by default). The alpha
     power envelope simply does not exist at a finer resolution, so the 0.5-4 Hz
     band the roadmap describes is not recoverable here; what is computed is the
     slow (<0.5 Hz) envelope coupling that a 1 s hop can actually support. Drop
     --hop to 0.25 s if you want to push the usable band up.
  2. Playback times are reconstructed as: first segment starts when it finished
     generating, and each later segment starts when the previous one ends (queue
     depth 1 makes this exact, barring underruns, which are counted).

Usage:
    python src/analyze_session.py                          # latest session
    python src/analyze_session.py sessions/P01_20260814_...
    python src/analyze_session.py --compare sessions/A sessions/B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session_logger import latest_session, load_session  # noqa: E402

TARGET_BAND_HALF_WIDTH = 0.5   # z units; "in band" means |z - target| <= this
HOLD_SECONDS = 30.0            # a crossing only counts if it is held this long
MAX_LAG_SECONDS = 20.0
MIN_LAG_SECONDS = -5.0


# ------------------------------------------------------------------- metrics


def basic_metrics(session: Dict) -> Dict:
    manifest = session["manifest"]
    windows = session["windows"]
    if not windows:
        return {"error": "no windows logged"}

    hop = manifest.get("feature_config", {}).get("hop_seconds", 1.0)
    target_z = manifest.get("target_z", -1.0)

    baseline = [w for w in windows if w.get("phase") == "baseline"]
    intervention = [w for w in windows if w.get("phase") == "intervention"]

    def duration(rows: List[Dict]) -> float:
        return (rows[-1]["elapsed_s"] - rows[0]["elapsed_s"]) if len(rows) > 1 else 0.0

    applied = [w for w in intervention if w.get("applied") and np.isfinite(w.get("z", float("nan")))]
    z = np.asarray([w["z"] for w in applied], dtype=float)
    t = np.asarray([w["elapsed_s"] for w in applied], dtype=float)

    in_band = np.abs(z - target_z) <= TARGET_BAND_HALF_WIDTH if z.size else np.array([], dtype=bool)

    metrics = {
        "session_dir": session["dir"],
        "participant": manifest.get("participant_id"),
        "condition": manifest.get("condition"),
        "sampling_rate": manifest.get("sampling_rate"),
        "target_z": target_z,
        "hop_seconds": hop,
        "baseline_duration_s": duration(baseline),
        "intervention_duration_s": duration(intervention),
        "windows_baseline": len(baseline),
        "windows_intervention": len(intervention),
        "rejection_rate_baseline": _rejection_rate(baseline),
        "rejection_rate_intervention": _rejection_rate(intervention),
        "rejection_reasons": _reject_reasons(windows),
        "z_mean": float(z.mean()) if z.size else float("nan"),
        "z_median": float(np.median(z)) if z.size else float("nan"),
        "z_sd": float(z.std(ddof=1)) if z.size > 1 else float("nan"),
        "z_min": float(z.min()) if z.size else float("nan"),
        "z_max": float(z.max()) if z.size else float("nan"),
        "time_in_band_fraction": float(in_band.mean()) if in_band.size else float("nan"),
        "time_in_band_s": float(in_band.sum() * hop) if in_band.size else float("nan"),
        "time_to_target_s": time_to_target(t, z, target_z, hop),
        "first_half_z_mean": float(z[: z.size // 2].mean()) if z.size > 3 else float("nan"),
        "second_half_z_mean": float(z[z.size // 2:].mean()) if z.size > 3 else float("nan"),
    }
    metrics["z_drift"] = metrics["second_half_z_mean"] - metrics["first_half_z_mean"]
    metrics.update(audio_metrics(session))
    return metrics


def _rejection_rate(rows: List[Dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if not r.get("valid", False)) / len(rows)


def _reject_reasons(rows: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if not row.get("valid", False):
            reason = str(row.get("reject_reason", "unknown"))
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def time_to_target(t: np.ndarray, z: np.ndarray, target_z: float, hop: float) -> float:
    """
    Seconds from intervention start until z first reaches the target band AND stays
    in it (allowing brief excursions) for HOLD_SECONDS. Returns nan if never reached
    - which is a legitimate result and must not be silently coded as the session
    length.
    """
    if z.size == 0:
        return float("nan")
    in_band = np.abs(z - target_z) <= TARGET_BAND_HALF_WIDTH
    hold_n = max(1, int(HOLD_SECONDS / hop))
    for i in range(in_band.size - hold_n + 1):
        if in_band[i] and in_band[i: i + hold_n].mean() >= 0.8:
            return float(t[i] - t[0])
    return float("nan")


def audio_metrics(session: Dict) -> Dict:
    segments = session["audio"]
    if not segments:
        return {"segments": 0}
    gen = np.asarray([s.get("generation_s", np.nan) for s in segments], dtype=float)
    gen = gen[np.isfinite(gen)]
    duration = float(segments[0].get("duration_s", np.nan))
    return {
        "segments": len(segments),
        "median_generation_s": float(np.median(gen)) if gen.size else float("nan"),
        "p95_generation_s": float(np.percentile(gen, 95)) if gen.size else float("nan"),
        "median_realtime_factor": float(np.median(gen) / duration) if gen.size and duration else float("nan"),
        "generation_slower_than_realtime_pct": float((gen > duration).mean() * 100) if gen.size else float("nan"),
        "underruns": int(max((s.get("underruns", 0) for s in segments), default=0)),
        "unique_prompts": len({s.get("prompt") for s in segments}),
    }


# -------------------------------------------------------- coupling index (ACI)


def coupling_index(session: Dict, n_permutations: int = 200) -> Dict:
    """
    Lagged audio-neural coupling index.

    Positive lag = audio leads brain, which is the direction the causal claim
    predicts. The circular-shift null gives a p-value that respects the strong
    autocorrelation in both series; a naive Pearson p-value would be wildly
    anticonservative here.
    """
    manifest = session["manifest"]
    hop = manifest.get("feature_config", {}).get("hop_seconds", 1.0)

    windows = [
        w for w in session["windows"]
        if w.get("phase") == "intervention" and w.get("valid") and np.isfinite(w.get("alpha", np.nan))
    ]
    if len(windows) < 40 or not session["audio"]:
        return {"aci_error": "not enough valid intervention windows or no audio segments"}

    t_brain = np.asarray([w["elapsed_s"] for w in windows], dtype=float)
    alpha = np.log10(np.asarray([w["alpha"] for w in windows], dtype=float))

    audio_t, audio_env = _stitch_audio_envelope(session)
    if audio_t.size < 20:
        return {"aci_error": "audio envelope too short (was envelope logging enabled?)"}

    # Resample the audio envelope onto the brain grid. The brain series cannot be
    # made denser, so the audio series comes down to meet it.
    audio_on_grid = np.interp(t_brain, audio_t, audio_env, left=np.nan, right=np.nan)
    ok = np.isfinite(audio_on_grid) & np.isfinite(alpha)
    if ok.sum() < 40:
        return {"aci_error": "insufficient overlap between audio and brain timelines"}

    x = _prep(audio_on_grid[ok])   # audio envelope
    y = _prep(alpha[ok])           # alpha power envelope

    lags = np.arange(int(MIN_LAG_SECONDS / hop), int(MAX_LAG_SECONDS / hop) + 1)
    r = np.asarray([_lagged_corr(x, y, int(lag)) for lag in lags])
    finite = np.isfinite(r)
    if not finite.any():
        return {"aci_error": "correlation undefined at all lags"}

    best = int(np.nanargmax(np.abs(r)))
    peak_r = float(r[best])
    peak_lag_s = float(lags[best] * hop)

    rng = np.random.default_rng(0)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shifted = np.roll(x, int(rng.integers(len(x) // 10, len(x) - len(x) // 10)))
        null[i] = np.nanmax(np.abs([_lagged_corr(shifted, y, int(lag)) for lag in lags]))
    p_value = float((null >= abs(peak_r)).mean())

    return {
        "aci_peak_r": peak_r,
        "aci_peak_lag_s": peak_lag_s,
        "aci_p_circular_shift": p_value,
        "aci_r_at_zero_lag": float(r[lags == 0][0]) if (lags == 0).any() else float("nan"),
        "aci_null_mean": float(null.mean()),
        "aci_n_points": int(ok.sum()),
        "aci_lag_range_s": [float(lags[0] * hop), float(lags[-1] * hop)],
        "aci_note": (
            "Both series on the analysis hop grid; the 0.5-4 Hz band is not "
            "recoverable at this rate, so this is slow-envelope coupling."
        ),
    }


def _stitch_audio_envelope(session: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct a continuous playback-time envelope from the per-segment logs."""
    times: List[float] = []
    values: List[float] = []
    cursor: Optional[float] = None
    for seg in session["audio"]:
        env = seg.get("envelope") or []
        rate = seg.get("envelope_rate_hz", 20.0)
        if not env:
            continue
        step = 1.0 / rate

        if seg.get("envelope_retrospective"):
            # library_engine: the envelope is audio ALREADY HEARD, drained at the
            # moment of logging, so it ends at elapsed_s rather than starting there.
            # Anchoring it forward like a streaming segment would shift the entire
            # audio timeline later by one segment tenure and bias every lag estimate.
            start = float(seg["elapsed_s"]) - len(env) * step
            times.extend(start + i * step for i in range(len(env)))
            values.extend(float(v) for v in env)
            cursor = float(seg["elapsed_s"])
            continue

        if cursor is None:
            cursor = float(seg["elapsed_s"])  # first segment plays as soon as it exists
        times.extend(cursor + i * step for i in range(len(env)))
        values.extend(float(v) for v in env)
        cursor += len(env) * step

    # Retrospective drains can overlap slightly if a switch lands mid-bin, and the
    # interpolation downstream needs a monotonic grid.
    order = np.argsort(np.asarray(times, dtype=float), kind="stable")
    t = np.asarray(times, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], v[keep]


def _prep(v: np.ndarray) -> np.ndarray:
    """Linear-detrend and z-score, so slow drift in either series cannot fake coupling."""
    v = np.asarray(v, dtype=float)
    idx = np.arange(v.size, dtype=float)
    slope, intercept = np.polyfit(idx, v, 1)
    v = v - (slope * idx + intercept)
    sd = v.std()
    return v / sd if sd > 0 else v


def _lagged_corr(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """Pearson r between x(t) and y(t+lag). Positive lag: x leads y."""
    if lag > 0:
        a, b = x[:-lag], y[lag:]
    elif lag < 0:
        a, b = x[-lag:], y[:lag]
    else:
        a, b = x, y
    if a.size < 20:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# -------------------------------------------------------------------- report


def report(session_dir: str, skip_aci: bool = False) -> Dict:
    session = load_session(session_dir)
    metrics = basic_metrics(session)
    if not skip_aci:
        metrics.update(coupling_index(session))
        metrics.update(event_locked_response(session))

    print("=" * 74)
    print(f"SESSION: {session_dir}")
    print("=" * 74)
    print(f"  participant / condition : {metrics.get('participant')} / {metrics.get('condition')}")
    print(f"  sampling rate (recorded): {metrics.get('sampling_rate')} Hz")
    print(f"  baseline                : {metrics.get('baseline_duration_s', 0):.0f} s "
          f"({metrics.get('windows_baseline')} windows, "
          f"{metrics.get('rejection_rate_baseline', float('nan')):.1%} rejected)")
    print(f"  intervention            : {metrics.get('intervention_duration_s', 0) / 60:.1f} min "
          f"({metrics.get('windows_intervention')} windows, "
          f"{metrics.get('rejection_rate_intervention', float('nan')):.1%} rejected)")

    print("\n  PRIMARY OUTCOMES")
    print(f"    target z              : {metrics.get('target_z'):+.2f} "
          f"(band +/- {TARGET_BAND_HALF_WIDTH})")
    print(f"    time in band          : {metrics.get('time_in_band_fraction', float('nan')):.1%} "
          f"({metrics.get('time_in_band_s', float('nan')):.0f} s)")
    ttt = metrics.get("time_to_target_s", float("nan"))
    print(f"    time to target        : {'never reached' if not np.isfinite(ttt) else f'{ttt:.0f} s'}")
    print(f"    z mean / sd           : {metrics.get('z_mean', float('nan')):+.3f} / "
          f"{metrics.get('z_sd', float('nan')):.3f}")
    print(f"    z drift (2nd - 1st)   : {metrics.get('z_drift', float('nan')):+.3f}")

    if metrics.get("segments"):
        print("\n  AUDIO PIPELINE")
        print(f"    segments generated    : {metrics['segments']} "
              f"({metrics.get('unique_prompts')} unique prompts)")
        print(f"    generation median/p95 : {metrics.get('median_generation_s', float('nan')):.2f} s / "
              f"{metrics.get('p95_generation_s', float('nan')):.2f} s")
        print(f"    realtime factor       : {metrics.get('median_realtime_factor', float('nan')):.2f}x "
              f"({metrics.get('generation_slower_than_realtime_pct', float('nan')):.0f}% of segments slower than RT)")
        print(f"    buffer underruns      : {metrics.get('underruns')}")

    if "aci_peak_r" in metrics:
        print("\n  LAGGED AUDIO-NEURAL COUPLING")
        print(f"    peak r                : {metrics['aci_peak_r']:+.3f}")
        print(f"    at lag                : {metrics['aci_peak_lag_s']:+.1f} s "
              f"({'audio leads brain' if metrics['aci_peak_lag_s'] > 0 else 'brain leads audio'})")
        print(f"    r at zero lag         : {metrics['aci_r_at_zero_lag']:+.3f}")
        print(f"    p (circular shift)    : {metrics['aci_p_circular_shift']:.3f} "
              f"(null peak mean {metrics['aci_null_mean']:.3f})")
    elif "aci_error" in metrics:
        print(f"\n  coupling index unavailable: {metrics['aci_error']}")

    if "elr_effect_z" in metrics:
        print("\n  EVENT-LOCKED RESPONSE (rung changes)")
        print(f"    events / epochs       : {metrics['elr_n_events']} / {metrics['elr_n_epochs']}")
        print(f"    effect                : {metrics['elr_effect_z']:+.3f} z "
              f"(positive = z moved the way the music asked)")
        print(f"    p (shuffled onsets)   : {metrics['elr_p_permutation']:.3f} "
              f"(null sd {metrics['elr_null_sd']:.3f})")
        print("    NOT causal on its own : rung changes are TRIGGERED by z moving, so a")
        print("                            positive effect is what a loop with no effect")
        print("                            also produces. Only adaptive minus yoked sham")
        print("                            is interpretable. See the docstring.")
    elif "elr_error" in metrics:
        print(f"\n  event-locked response unavailable: {metrics['elr_error']}")

    reasons = metrics.get("rejection_reasons") or {}
    if reasons:
        print("\n  ARTIFACT REJECTIONS")
        for reason, count in reasons.items():
            print(f"    {count:5d}  {reason}")

    print("=" * 74)

    out_path = os.path.join(session_dir, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    print(f"Wrote {out_path}\n")
    return metrics


def compare(dirs: List[str]) -> None:
    rows = [report(d) for d in dirs]
    print("=" * 74)
    print("ARM COMPARISON")
    print("=" * 74)
    header = f"  {'metric':<26}" + "".join(f"{os.path.basename(d)[:16]:>18}" for d in dirs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ("condition", "time_in_band_fraction", "z_mean", "z_drift",
                "time_to_target_s", "aci_peak_r", "aci_peak_lag_s", "aci_p_circular_shift",
                "rejection_rate_intervention"):
        cells = ""
        for row in rows:
            value = row.get(key, float("nan"))
            cells += f"{value:>18.3f}" if isinstance(value, (int, float)) else f"{str(value):>18}"
        print(f"  {key:<26}{cells}")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a closed-loop session log")
    parser.add_argument("session", nargs="?", default=None, help="session dir (default: most recent)")
    parser.add_argument("--compare", nargs="+", default=None, help="compare two or more session dirs")
    parser.add_argument("--skip-aci", action="store_true", help="skip the coupling index (faster)")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    session_dir = args.session or latest_session()
    if not session_dir:
        print("No sessions found. Run: python src/live_music.py --mock --baseline-seconds 20 --duration 2")
        return 1
    report(session_dir, skip_aci=args.skip_aci)
    return 0


# ------------------------------------------------- event-locked coupling


EVENT_PRE_S = 10.0    # baseline window before a rung change
EVENT_POST_S = 30.0   # response window after it
EVENT_MIN_SEPARATION_S = 15.0


def event_locked_response(session: Dict, n_permutations: int = 500) -> Dict:
    """
    Brain response time-locked to rung changes, as a companion to coupling_index.

    WHY THIS EXISTS, AND WHY THE CONTINUOUS INDEX IS NOT ENOUGH

    coupling_index cross-correlates the whole session. That has a perverse property
    which PILOT01 made concrete: a participant who reaches target and stays there
    keeps the controller on a single rung, so the audio stops varying in the way the
    controller drives, and there is almost nothing left for a continuous correlation
    to find. The pilot returned r = -0.054, p = 0.795 while sitting on rung 1 for
    95.9% of the session. The estimator was not failing - it is validated to recover
    known lags at r ~ 0.85 - there was simply no controller-driven variation in the
    window it was given.

    So the continuous index is weakest exactly when the intervention is working
    best. That is a bad property for a primary outcome, and it is not fixable by
    improving the estimator.

    This conditions on the events instead. A rung change is a discrete, timestamped
    moment when the music demonstrably changed; the question is whether the brain
    moved afterward. Standard event-related logic, and it stays powered when the
    session is mostly stable because it only ever looks at the moments that carry
    information.

    DIRECTION IS FOLDED IN, NOT AVERAGED OUT. Changes up and down the ladder predict
    opposite responses, so averaging them raw would cancel the effect. Each epoch is
    signed by its direction, and the reported effect is "did z move the way the
    music asked", positive meaning it did.

    The null shuffles event times within the intervention rather than shuffling the
    data, which preserves the autocorrelation of z. A naive t-test against zero
    would be badly anticonservative on a signal this smooth.

    THIS NUMBER IS NOT INTERPRETABLE ON ITS OWN. READ THIS BEFORE REPORTING IT.

    A rung change happens BECAUSE z moved. z then keeps moving, because z is
    autocorrelated and whatever was driving it did not stop. So a positive effect is
    exactly what a closed loop with no therapeutic effect whatsoever would produce:
    the music follows the brain, the brain continues on its existing trajectory, and
    time-locking to the follow makes it look like a lead. Baseline-correcting on the
    pre-window does not remove this, because z was already moving during that window
    - that movement is what triggered the event.

    On PILOT01 this returns +0.41 z at p = 0.104 where the continuous index found
    nothing. That is a demonstration that the method has power, NOT evidence of an
    effect, and the two must not be confused in writing.

    The yoked sham is what makes it causal. In the sham arm the same rung changes
    occur at the same times, driven by a DIFFERENT participant's brain, so the sham's
    event-locked effect is an estimate of this confound with the contingency removed.
    Adaptive minus sham is the causal quantity. A single-arm number is uninterpretable
    and reporting one would be a serious error.
    """
    hop = session["manifest"].get("feature_config", {}).get("hop_seconds", 1.0)

    windows = [w for w in session["windows"]
               if w.get("phase") == "intervention" and w.get("valid")
               and isinstance(w.get("z"), (int, float)) and np.isfinite(w["z"])]
    if len(windows) < 60 or not session["audio"]:
        return {"elr_error": "not enough valid intervention windows or no audio events"}

    t_brain = np.asarray([w["elapsed_s"] for w in windows], dtype=float)
    z = np.asarray([w["z"] for w in windows], dtype=float)

    events = _rung_change_events(session)
    if len(events) < 4:
        return {"elr_error": f"only {len(events)} rung changes; need at least 4"}

    grid = np.arange(-EVENT_PRE_S, EVENT_POST_S + hop, hop)
    epochs, directions = [], []
    for onset, direction in events:
        seg = np.interp(onset + grid, t_brain, z, left=np.nan, right=np.nan)
        if np.isnan(seg).mean() > 0.25:
            continue
        pre = seg[grid < 0]
        if not np.isfinite(pre).any():
            continue
        # Baseline-correct so the epoch measures CHANGE, not the level the
        # participant happened to be at when the music switched.
        epochs.append(seg - np.nanmean(pre))
        directions.append(direction)

    if len(epochs) < 4:
        return {"elr_error": f"only {len(epochs)} usable epochs after rejection"}

    stack = np.asarray(epochs, dtype=float)
    sign = np.asarray(directions, dtype=float).reshape(-1, 1)

    # A change UP the ladder asks for more arousal, so agreement means z rises.
    # Multiplying by the direction makes "agrees with the music" positive for both.
    aligned = stack * sign
    mean_curve = np.nanmean(aligned, axis=0)
    post = mean_curve[grid > 0]
    effect = float(np.nanmean(post))

    rng = np.random.default_rng(0)
    span = (t_brain[0] + EVENT_PRE_S, t_brain[-1] - EVENT_POST_S)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        fake = rng.uniform(span[0], span[1], size=len(epochs))
        vals = []
        for onset, direction in zip(fake, directions):
            seg = np.interp(onset + grid, t_brain, z, left=np.nan, right=np.nan)
            pre = seg[grid < 0]
            if not np.isfinite(pre).any():
                continue
            vals.append((seg - np.nanmean(pre)) * direction)
        null[i] = np.nanmean(np.asarray(vals)[:, grid > 0]) if vals else np.nan

    finite_null = null[np.isfinite(null)]
    p_value = (float((np.abs(finite_null) >= abs(effect)).mean())
               if finite_null.size else float("nan"))

    return {
        "elr_n_events": int(len(events)),
        "elr_n_epochs": int(len(epochs)),
        "elr_effect_z": effect,
        "elr_p_permutation": p_value,
        "elr_null_sd": float(np.std(finite_null)) if finite_null.size else float("nan"),
        "elr_peak_z": float(np.nanmax(np.abs(post))) if post.size else float("nan"),
        "elr_window_s": [-EVENT_PRE_S, EVENT_POST_S],
        "elr_curve": [round(float(v), 4) for v in mean_curve],
        "elr_note": (
            "Positive effect means z moved the way the music asked. Companion to "
            "the continuous ACI, which loses power when the session is stable."
        ),
    }


def _rung_change_events(session: Dict) -> List[Tuple[float, int]]:
    """
    (onset_seconds, direction) for each rung change, direction +1 up / -1 down.

    Keys on the RUNG, not the prompt string. Before the hysteresis fix the trend
    suffix rewrote the prompt hundreds of times per session without the music's
    energy level changing at all, and treating those as events would swamp the real
    ones with noise. Closely spaced changes are also dropped: overlapping epochs are
    not independent, and the pilot has runs of rung changes 1-3 s apart.
    """
    events: List[Tuple[float, int]] = []
    previous_rung: Optional[int] = None
    for seg in session["audio"]:
        rung = seg.get("rung")
        if rung is None:
            rung = _rung_from_prompt(seg.get("prompt", ""))
        if rung is None:
            continue
        if previous_rung is not None and rung != previous_rung:
            direction = 1 if rung > previous_rung else -1
            onset = float(seg["elapsed_s"])
            if not events or onset - events[-1][0] >= EVENT_MIN_SEPARATION_S:
                events.append((onset, direction))
        previous_rung = rung
    return events


def _rung_from_prompt(prompt: str) -> Optional[int]:
    """Recover the ladder position from a prompt string, for streaming-engine logs."""
    if not prompt:
        return None
    from music_engine import _ENERGY_LADDER  # noqa: PLC0415 - avoids a cycle at import
    for i, base in enumerate(_ENERGY_LADDER):
        if prompt.startswith(base):
            return i
    return None


if __name__ == "__main__":
    raise SystemExit(main())
