"""
verify_library.py - prove the library covers the controller and the engine plays it.

Two things can silently break the library path, and neither shows up as an exception
at runtime:

  COVERAGE DRIFT. build_prompt() is edited - a rung is added, a suffix reworded - and
  the library no longer contains a prompt the controller can emit. The engine falls
  back to the nearest rung rather than crashing (losing a participant's session to a
  KeyError would be worse), so the failure is silent and the study runs with the
  wrong music. This checks every reachable prompt resolves EXACTLY.

  MIXING FAULTS. Gaps, clipping, or dropouts at the crossfade seams. These are
  audible but easy to miss when you are listening for musical quality rather than
  for a 20 ms hole, and they do not raise.

--synthetic builds a stand-in library of cheap synthesized tones with the same
manifest schema, so all of this is testable with no GPU and no rendered library. That
matters: it decouples engine correctness from a 20-minute build, and it means the
engine stays under test even on a machine that can never run MusicGen.

Usage:
    python scripts/verify_library.py                    # against library/
    python scripts/verify_library.py --synthetic        # self-contained, no GPU
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from music_engine import build_prompt  # noqa: E402
from library_engine import LibraryConfig, LibraryMusicEngine  # noqa: E402
from build_library import enumerate_prompts, normalize  # noqa: E402


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        else:
            self.failed.append(name)
            print(f"  FAIL  {name}" + (f"  ({detail})" if detail else ""))
        return ok


def build_synthetic(root: str, variants: int = 3, seconds: float = 8.0,
                    sample_rate: int = 32000) -> None:
    """A stand-in library with the real schema and trivially cheap audio."""
    import soundfile as sf

    prompts = enumerate_prompts()
    os.makedirs(os.path.join(root, "segments"), exist_ok=True)
    manifest = {
        "created": "synthetic", "model": "synthetic", "backend": "synthetic",
        "device": "cpu", "sample_rate": sample_rate, "segment_seconds": seconds,
        "variants_per_prompt": variants, "target_rms": 0.1, "prompts": [],
    }

    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    for entry in prompts:
        segments = []
        for v in range(variants):
            root_hz = 110.0 * (2 ** ((entry["rung"] * 2 + v) / 12.0))
            audio = normalize(
                np.sin(2 * np.pi * root_hz * t).astype(np.float32)
                + 0.3 * np.sin(2 * np.pi * root_hz * 2 * t).astype(np.float32)
            )
            rel = f"segments/rung{entry['rung']}_{entry['variant']}_v{v}.wav"
            sf.write(os.path.join(root, rel), audio, sample_rate, subtype="PCM_16")
            segments.append({
                "file": rel, "seed": v, "duration_s": seconds, "generation_s": 0.0,
                "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
                "peak": round(float(np.max(np.abs(audio))), 4),
            })
        manifest["prompts"].append({**entry, "segments": segments})

    manifest["segment_count"] = sum(len(e["segments"]) for e in manifest["prompts"])
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the segment library and engine")
    parser.add_argument("--library", default="library")
    parser.add_argument("--synthetic", action="store_true",
                        help="build and test a throwaway synthetic library (no GPU needed)")
    args = parser.parse_args()

    tmp = None
    library = args.library
    if args.synthetic:
        tmp = tempfile.mkdtemp(prefix="synthlib_")
        library = tmp
        print(f"Building synthetic library in {library} ...")
        build_synthetic(library)

    checks = Checks()

    print("\n" + "=" * 74)
    print("1. COVERAGE - every prompt the controller can emit must resolve exactly")
    print("=" * 74)

    engine = LibraryMusicEngine(LibraryConfig(library_dir=library, seed=0))
    all_prompts = enumerate_prompts()
    reachable = [p for p in all_prompts if p["reachable_default_targets"]]

    exact = [p for p in reachable if p["prompt"] in engine._by_prompt]
    checks.check(
        "all default-reachable prompts present",
        len(exact) == len(reachable),
        f"{len(exact)}/{len(reachable)}",
    )

    exact_all = [p for p in all_prompts if p["prompt"] in engine._by_prompt]
    checks.check(
        "all enumerated prompts present (incl. wider targets)",
        len(exact_all) == len(all_prompts),
        f"{len(exact_all)}/{len(all_prompts)}",
    )

    # The independent check: sweep the real controller, not the enumeration, so a
    # divergence between the two is caught rather than cancelling out.
    emitted = {
        build_prompt(float(z), target_z=tz, trend=tr)
        for tz in (-1.0, 1.0)
        for z in np.arange(-4.0, 4.05, 0.25)
        for tr in (None, -0.3, 0.0, 0.3)
    }
    unresolved = sorted(p for p in emitted if p not in engine._by_prompt)
    checks.check("live sweep of build_prompt resolves exactly", not unresolved,
                 f"{len(emitted)} distinct prompts emitted"
                 + (f", UNRESOLVED: {unresolved[:2]}" if unresolved else ""))

    print("\n" + "=" * 74)
    print("2. SELECTION - must be microseconds, not the 9-25 s generation it replaces")
    print("=" * 74)

    t0 = time.perf_counter()
    for _ in range(2000):
        engine._select(engine.get_target_prompt(), None)
    per_call_ms = (time.perf_counter() - t0) / 2000 * 1000
    checks.check("selection under 1 ms", per_call_ms < 1.0, f"{per_call_ms * 1000:.1f} us per call")

    print("\n" + "=" * 74)
    print("3. MIXING - continuous audio, no gaps, no clipping through the seams")
    print("=" * 74)

    # A trajectory that forces many prompt changes, so seams are exercised hard.
    def control(t: float) -> str:
        return build_prompt(2.0 - 0.15 * t, target_z=-1.0, trend=-0.15)

    audio = engine.render_offline(90.0, control=control)
    sr = engine.sample_rate

    checks.check("rendered the requested duration",
                 abs(audio.size / sr - 90.0) < 0.05, f"{audio.size / sr:.2f} s")

    peak = float(np.max(np.abs(audio)))
    checks.check("no clipping", peak <= 1.0, f"peak {peak:.3f}")

    # Peak alone is not enough: the engine hard-clips, so a clipped render still
    # reports peak == 1.0 rather than exceeding it. The counter is the real check.
    checks.check("nothing was clipped", engine.clipped_samples == 0,
                 f"{engine.clipped_samples} samples clipped")

    # Headroom against the sqrt(2) worst case of summing two uncorrelated segments
    # under equal-power ramps. Passing on quiet synthetic tones proves nothing about
    # real renders, so report it rather than only asserting.
    print(f"        crossfade headroom: peak {peak:.3f}, "
          f"worst-case sqrt(2) bound would be {peak * 1.414:.3f}")

    # A gap is a run of near-silence longer than a few ms. Real music dips; a
    # dropout is flat zero across a whole block.
    silent = np.abs(audio) < 1e-4
    longest = 0
    run = 0
    for s in silent:
        run = run + 1 if s else 0
        longest = max(longest, run)
    checks.check("no dropouts", longest < sr * 0.01, f"longest silence {longest / sr * 1000:.1f} ms")

    rms = float(np.sqrt(np.mean(audio ** 2)))
    checks.check("output level sane", 0.01 < rms < 0.5, f"rms {rms:.3f}")

    # Level continuity across seams: equal-power crossfade should hold loudness
    # roughly constant. Compare per-100ms RMS; a linear fade would show dips.
    frame = int(0.1 * sr)
    frames = audio[:audio.size // frame * frame].reshape(-1, frame)
    frame_rms = np.sqrt((frames ** 2).mean(axis=1))
    active = frame_rms[frame_rms > 1e-3]
    ratio = float(np.percentile(active, 95) / np.percentile(active, 5)) if active.size else 999
    checks.check("no level collapse at seams", ratio < 12.0, f"p95/p5 frame RMS {ratio:.1f}x")

    print("\n" + "=" * 74)
    print("4. LATENCY - a prompt change must be audible within one crossfade")
    print("=" * 74)

    stats = engine.stats()
    budget = engine.worst_case_audio_latency_s
    checks.check("switches actually happened", stats["switches"] > 5, f"{stats['switches']} switches")
    checks.check("no fallbacks used", not stats["missing_prompts"],
                 f"missing: {stats['missing_prompts'][:2]}")
    checks.check("no underruns", stats["underruns"] == 0)

    from music_engine import MusicConfig
    streaming = MusicConfig().segment_seconds * MusicConfig().queue_depth
    checks.check("beats the streaming commitment", budget < streaming,
                 f"{budget:.1f} s vs {streaming:.1f} s streaming = {streaming / budget:.0f}x better")

    print("\n" + "=" * 74)
    print(f"  {checks.passed} passed, {len(checks.failed)} failed")
    if checks.failed:
        for name in checks.failed:
            print(f"    FAILED: {name}")
    print("=" * 74)

    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)

    return 1 if checks.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
