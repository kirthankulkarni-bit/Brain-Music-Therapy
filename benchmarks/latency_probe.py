"""
latency_probe.py - measure the closed-loop latency budget end to end.

Generalizes testing/half_precision_test.py (which measured only MusicGen, at one
duration) into the full budget. The output table is the results section for the
latency characterization: the claim is not "we made the GPU faster" but "here is
the irreducible latency floor for consumer-EEG closed-loop audio, and here is
which term dominates".

Three components are measured separately:

  A. ANALYSIS PATH (structural, no GPU involved)
     window centroid delay + hop quantization + smoother group delay.
     This was ~10 s under the old 5 s window / 2 s hop / 5-sample boxcar, and no
     amount of model optimization touches a single term of it.

  B. DSP COMPUTE (per-window wall time for filter + Welch + band integration)
     Expect sub-millisecond. Included to show it is not the problem.

  C. GENERATION + PLAYBACK COMMITMENT
     MusicGen wall time per segment at fp32/fp16 across durations, plus the queue
     commitment term (queue_depth x segment_seconds), which is the real audio-side
     bottleneck once generation runs faster than realtime.

RUN THIS MORE THAN ONCE. On a thermally limited laptop GPU, between-run variance
dominates within-run variance by roughly an order of magnitude. Measured on a
GTX 1650 Ti: within a single run p95/median is 1.01-1.11, but the same
configuration across three separate runs varied by up to 1.96x. A single run is
not a reproducible measurement, and `--trials` does not fix it because all trials
in a run share the same thermal state.

Use --label to give each run a distinct output file, run several, and report the
range. Conclusions that survive the full spread are the ones worth publishing.

Usage:
    python benchmarks/latency_probe.py --skip-musicgen     # A and B, seconds
    python benchmarks/latency_probe.py                     # everything
    python benchmarks/latency_probe.py --durations 4 8 12 --trials 3
    python benchmarks/latency_probe.py --label run1 --out benchmarks/run1.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from eeg_features import FeatureConfig, FeatureExtractor, ExponentialSmoother, latency_budget  # noqa: E402
from music_engine import MusicConfig  # noqa: E402


# GPUs that report compute capability 7.5 but have NO tensor cores. Nvidia stripped
# the tensor and RT cores out of the GTX 16-series (TU116/TU117) while keeping the
# same capability number as the RTX 20-series, so capability alone cannot be used to
# predict whether fp16 autocast will help. This is exactly why fp16 gave no speedup
# on the 1650 Ti and is expected to on a T4.
_NO_TENSOR_CORE_PATTERNS = ("GTX 16", "GTX 10", "GTX 9", "MX", "GT 1")


def hardware_fingerprint() -> dict:
    """Identify the machine precisely enough to compare runs across hardware."""
    info = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor() or "unknown",
        "device": "cpu",
    }
    try:
        import torch
    except ImportError:
        return info

    info["torch"] = torch.__version__
    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    has_tensor_cores = capability[0] >= 7 and not any(p in name for p in _NO_TENSOR_CORE_PATTERNS)

    info.update({
        "device": "cuda",
        "gpu_name": name,
        "gpu_vram_gb": round(props.total_memory / 1e9, 1),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "multiprocessors": props.multi_processor_count,
        "has_tensor_cores": has_tensor_cores,
        "cuda": torch.version.cuda,
    })
    return info


def synth_eeg(sampling_rate: float, seconds: float, n_channels: int = 4, seed: int = 0) -> np.ndarray:
    """Pink-ish noise with an alpha bump and a mains line, so the DSP does real work."""
    rng = np.random.default_rng(seed)
    n = int(sampling_rate * seconds)
    t = np.arange(n) / sampling_rate
    out = np.empty((n_channels, n))
    for ch in range(n_channels):
        noise = rng.normal(0, 8, n)
        alpha = 12.0 * np.sin(2 * np.pi * 10.0 * t + rng.uniform(0, 6.28))
        beta = 4.0 * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 6.28))
        mains = 6.0 * np.sin(2 * np.pi * 60.0 * t)
        out[ch] = noise + alpha + beta + mains
    return out


def bench_analysis(sampling_rate: float, window_s: float, hop_s: float, tau_s: float, trials: int) -> dict:
    cfg = FeatureConfig(sampling_rate=sampling_rate, window_seconds=window_s, hop_seconds=hop_s)
    extractor = FeatureExtractor(cfg)
    smoother = ExponentialSmoother(hop_seconds=hop_s, tau_seconds=tau_s)

    data = synth_eeg(sampling_rate, window_s + 1.0)
    window = data[:, : cfg.window_samples]

    extractor.extract(window)  # warm up scipy's filter design caches
    times = []
    for i in range(trials):
        t0 = time.perf_counter()
        feats = extractor.extract(window, timestamp=float(i))
        smoother.update(feats.log_beta_alpha)
        times.append(time.perf_counter() - t0)

    budget = latency_budget(cfg, tau_s)
    arr = np.asarray(times)
    return {
        "sampling_rate": sampling_rate,
        "window_seconds": window_s,
        "hop_seconds": hop_s,
        "smoother_tau_s": tau_s,
        "dsp_median_ms": float(np.median(arr) * 1000),
        "dsp_p95_ms": float(np.percentile(arr, 95) * 1000),
        "dsp_headroom_x": float(hop_s / np.median(arr)),
        **budget,
    }


def bench_musicgen(durations: list[float], trials: int, model_name: str) -> list[dict]:
    import torch
    from audiocraft.models import MusicGen

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {model_name} on {device}...")
    t0 = time.time()
    model = MusicGen.get_pretrained(model_name, device=device)
    print(f"  loaded in {time.time() - t0:.1f} s")

    prompt = ["calm ambient piano with soft strings, gentle and spacious, no drums, 60 bpm"]
    rows = []

    for use_fp16 in (False, True):
        if use_fp16 and device != "cuda":
            continue
        for duration in durations:
            model.set_generation_params(duration=duration)

            # Untimed warm-up. The first generation after a config change pays for
            # CUDA kernel autotuning and allocator growth, and with a small trial
            # count that one slow run dominates the median: a trials=2 run measured
            # 5-6x realtime where trials=3 with warm-up measures ~3.2x on the same
            # hardware. Discard it explicitly rather than hoping the median absorbs it.
            with torch.inference_mode():
                if use_fp16:
                    with torch.amp.autocast(device_type=device, dtype=torch.float16):
                        model.generate(prompt, progress=False)
                else:
                    model.generate(prompt, progress=False)
            if device == "cuda":
                torch.cuda.synchronize()

            times = []
            for trial in range(trials):
                torch.cuda.synchronize() if device == "cuda" else None
                t0 = time.time()
                with torch.inference_mode():
                    if use_fp16:
                        with torch.amp.autocast(device_type=device, dtype=torch.float16):
                            model.generate(prompt, progress=False)
                    else:
                        model.generate(prompt, progress=False)
                torch.cuda.synchronize() if device == "cuda" else None
                elapsed = time.time() - t0
                times.append(elapsed)
                print(
                    f"  {'fp16' if use_fp16 else 'fp32'} {duration:4.0f}s  "
                    f"trial {trial + 1}/{trials}: {elapsed:6.2f}s  ({elapsed / duration:.2f}x realtime)"
                )
            arr = np.asarray(times)
            rows.append(
                {
                    "precision": "fp16" if use_fp16 else "fp32",
                    "duration_s": duration,
                    "median_generation_s": float(np.median(arr)),
                    "p95_generation_s": float(np.percentile(arr, 95)),
                    "realtime_factor": float(np.median(arr) / duration),
                    "faster_than_realtime": bool(np.median(arr) < duration),
                    "device": device,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop latency budget probe")
    parser.add_argument("--sampling-rate", type=float, default=256.0)
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--hop", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=3.0, help="smoother time constant")
    parser.add_argument("--durations", type=float, nargs="+", default=[4.0, 8.0, 12.0])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--skip-musicgen", action="store_true")
    parser.add_argument("--label", default=None,
                        help="short name for this machine, e.g. nitro5-1650ti or colab-t4")
    parser.add_argument("--out", default="benchmarks/latency_results.json")
    args = parser.parse_args()

    hardware = hardware_fingerprint()
    results: dict = {
        "timestamp": datetime.now().isoformat(),
        "label": args.label or hardware.get("gpu_name", hardware["platform"]),
        "hardware": hardware,
    }

    print("=" * 74)
    print("HARDWARE")
    print("=" * 74)
    for key, value in hardware.items():
        print(f"  {key:<22}: {value}")
    if hardware.get("device") == "cuda" and not hardware.get("has_tensor_cores", True):
        print()
        print(f"  NOTE: this GPU reports compute capability "
              f"{hardware.get('compute_capability')} but has no tensor cores,")
        print("  so fp16 autocast pays casting overhead without the matmul speedup.")
    print()

    print("=" * 74)
    print("A + B. ANALYSIS PATH")
    print("=" * 74)

    configs = [
        ("old (5 s window, 2 s hop, 5-tap boxcar)", 5.0, 2.0, 4.0),
        (f"new ({args.window:g} s window, {args.hop:g} s hop, tau={args.tau:g} s)", args.window, args.hop, args.tau),
    ]
    analysis_rows = []
    for label, window_s, hop_s, tau_s in configs:
        row = bench_analysis(args.sampling_rate, window_s, hop_s, tau_s, max(args.trials * 20, 50))
        row["label"] = label
        analysis_rows.append(row)
        print(f"\n  {label}")
        print(f"    window centroid delay : {row['window_centroid_delay_s']:6.2f} s")
        print(f"    hop quantization      : {row['hop_quantization_s']:6.2f} s")
        print(f"    smoother group delay  : {row['smoother_group_delay_s']:6.2f} s")
        print(f"    TOTAL ANALYSIS LAG    : {row['total_analysis_latency_s']:6.2f} s")
        print(f"    DSP compute per window: {row['dsp_median_ms']:6.2f} ms median, "
              f"{row['dsp_p95_ms']:.2f} ms p95  ({row['dsp_headroom_x']:.0f}x headroom vs hop)")

    results["analysis"] = analysis_rows
    saved = analysis_rows[0]["total_analysis_latency_s"] - analysis_rows[1]["total_analysis_latency_s"]
    print(f"\n  Analysis-path latency reduced by {saved:.2f} s, with zero GPU work involved.")

    print("\n" + "=" * 74)
    print("C. GENERATION AND PLAYBACK COMMITMENT")
    print("=" * 74)

    if args.skip_musicgen:
        print("  (skipped: --skip-musicgen)")
        results["musicgen"] = None
    else:
        try:
            results["musicgen"] = bench_musicgen(args.durations, args.trials, args.model)
            print(f"\n  {'precision':>9} | {'duration':>8} | {'median':>8} | {'p95':>7} | {'RT factor':>9} | faster than RT")
            print("  " + "-" * 68)
            for row in results["musicgen"]:
                print(
                    f"  {row['precision']:>9} | {row['duration_s']:>7.0f}s | {row['median_generation_s']:>7.2f}s | "
                    f"{row['p95_generation_s']:>6.2f}s | {row['realtime_factor']:>8.2f}x | "
                    f"{'YES' if row['faster_than_realtime'] else 'NO'}"
                )
        except ImportError as exc:
            print(f"  audiocraft/torch not importable ({exc}); rerun with --skip-musicgen or fix the install")
            results["musicgen"] = None

    cfg = MusicConfig()
    commitment = cfg.segment_seconds * cfg.queue_depth
    print(f"\n  Queue commitment (depth {cfg.queue_depth} x {cfg.segment_seconds:g} s segment): {commitment:.1f} s worst case")
    print("  Every extra unit of queue depth adds one full segment to closed-loop latency.")
    results["audio_commitment_s"] = commitment

    total = analysis_rows[1]["total_analysis_latency_s"] + commitment
    print("\n" + "=" * 74)
    print(f"  END-TO-END WORST CASE: {total:.1f} s "
          f"({analysis_rows[1]['total_analysis_latency_s']:.1f} s analysis + {commitment:.1f} s audio commitment)")
    print("=" * 74)
    results["end_to_end_worst_case_s"] = total

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
