"""
build_library.py - render every prompt the control loop can ever emit, offline.

WHY THIS IS A COMPLETE SOLUTION AND NOT A DEGRADED FALLBACK

The latency work concluded that live generation is infeasible: MusicGen runs
1.14-6.3x slower than realtime across every machine and backend measured, including
a datacenter T4 (benchmarks/, docs/results_latency.md). The obvious reading is that
a precomputed library is a compromise - you give up open-ended generation to get
realtime response.

That reading is wrong here, and the reason is a property of the controller rather
than of the model. build_prompt() in music_engine.py is a pure function of
(z, target_z, trend) whose range is FINITE AND SMALL: five energy rungs, each with
four trend variants, is twenty distinct strings. It cannot emit anything else. So a
library that covers those twenty prompts covers the controller's entire output
space - every musical state the closed loop can ever ask for, exactly.

Nothing is given up except novelty within a state, and that is bought back with
--variants: K independently seeded renders per prompt, chosen at random at runtime.

The enumeration is derived by sweeping build_prompt over a dense grid rather than
hardcoded, so it cannot drift if the ladder or the deadband changes. If someone adds
a rung, this script renders it on the next run and the coverage check in
verify_library.py fails until they do.

WHAT THE LIBRARY GIVES UP, HONESTLY

Streaming used generate_continuation, so each segment was conditioned on the tail of
the previous one and the music evolved as one piece. Library segments are
independent renders, so transitions are equal-power crossfades rather than musical
continuations. Within a segment the audio is as coherent as before; across a seam it
is a blend, not a development. That is the real cost, and it buys a drop in
worst-case response latency from 8 s to one crossfade.

LOUDNESS

Every segment is normalized to a common RMS target before writing. Without this,
crossfading between independent renders produces audible level jumps, because
MusicGen's output level varies substantially with the prompt - sparse drones come
out much quieter than driving rhythmic material. Normalizing at build time keeps the
runtime path free of any level tracking.

No fades are applied to the stored audio. The engine's crossfade shapes the seams,
and pre-faded edges would dip through every transition.

Usage:
    python scripts/build_library.py --dry-run          # list prompts, render nothing
    python scripts/build_library.py                    # build with defaults
    python scripts/build_library.py --variants 8       # add more; existing are kept
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from music_engine import _ENERGY_LADDER, build_prompt  # noqa: E402

MUSICGEN_TOKENS_PER_SECOND = 50

# build_prompt returns a bare ladder rung and nothing else. The trend suffix that used
# to be appended was removed on 9/5 - it gated on a slope whose noise is 1.8x the
# largest genuine drift, so no threshold could make it work (docs/deviations.md).
#
# This map is kept, with one entry, because the manifest schema carries a `variant`
# field and existing libraries on disk have segments labelled with the old values. A
# library built before the removal still loads; its suffixed segments simply stop being
# requested. Rebuild to reclaim the space - they were 60 of 220 segments, 8.0 of 29.3
# minutes, for prompts the controller could no longer produce.
_VARIANT_LABELS = {
    "": "base",
}

# -20 dBFS RMS. Loud enough to sit well above the noise floor of any playback path,
# quiet enough that the crossfade sum of two segments cannot clip.
_TARGET_RMS = 0.1


# The two therapeutic arms the study actually runs. Prompts reachable with only
# these are the ones the current design can ever play; the rest are rendered as
# insurance against a future target_z, and flagged so the distinction stays visible.
_DEFAULT_TARGETS = (-1.0, 1.0)


def enumerate_prompts() -> list[dict]:
    """
    Every string build_prompt can return, found by sweeping its inputs.

    Derived rather than constructed: if the ladder gains a rung or the deadband
    changes, this picks it up automatically.

    A FINDING THIS SWEEP PRODUCED. Under the two default targets, only 3 of the 5
    ladder rungs are reachable - rungs 0 and 4 are dead code. (Before the trend suffix
    was removed this read 12 of 20 combinations; the suffix multiplied every rung by
    four variants, and 15 of those 20 prompts could not be requested at all.)
    build_prompt always leads by exactly one rung toward `goal`, and `goal` is
    state_rung(target_z), which is rung 1 for target -1.0 and rung 3 for target +1.0.
    Reaching rung 0 requires goal == 0, i.e. a target near -2 SD, which neither arm
    uses. So the sparsest drone never plays - not even to a participant sitting at
    that arousal level, because the one-rung lead moves them off it immediately.

    Whether that is a bug depends on intent, which is a therapeutic call rather than
    an engineering one, so nothing here changes build_prompt. The sweep covers wider
    targets so the library is ready either way, and marks reachability so the choice
    is explicit rather than accidental.
    """
    seen: dict[str, dict] = {}
    wide_targets = [round(t, 1) for t in np.arange(-3.0, 3.05, 0.5)]

    for target_z in wide_targets:
        for z in np.arange(-4.0, 4.05, 0.1):
            text = build_prompt(float(z), target_z=float(target_z))
            default_arm = target_z in _DEFAULT_TARGETS

            if text in seen:
                seen[text]["reachable_default_targets"] |= default_arm
                continue

            rung = next((i for i, base in enumerate(_ENERGY_LADDER)
                         if text.startswith(base)), -1)
            suffix = text[len(_ENERGY_LADDER[rung]):] if rung >= 0 else ""
            seen[text] = {
                "prompt": text,
                "rung": rung,
                "variant": _VARIANT_LABELS.get(suffix, "unknown"),
                "reachable_default_targets": default_arm,
            }

    return sorted(seen.values(), key=lambda p: (p["rung"], p["variant"]))


def normalize(audio: np.ndarray, target_rms: float = _TARGET_RMS) -> np.ndarray:
    """Scale to a common RMS, then hard-limit only if that would clip."""
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    audio = audio * (target_rms / rms)
    peak = float(np.max(np.abs(audio)))
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    return audio.astype(np.float32)


def load_generator(model_name: str, backend: str):
    """Returns (generate(prompt, seconds, seed) -> np.ndarray, sample_rate, device)."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if backend == "transformers":
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()
        sample_rate = int(model.config.audio_encoder.sampling_rate)

        def generate(prompt: str, seconds: float, seed: int) -> np.ndarray:
            torch.manual_seed(seed)
            inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                wav = model.generate(
                    **inputs,
                    do_sample=True,
                    guidance_scale=3.0,
                    max_new_tokens=int(seconds * MUSICGEN_TOKENS_PER_SECOND),
                )
            return wav[0, 0].detach().cpu().float().numpy()

    else:
        from audiocraft.models import MusicGen

        model = MusicGen.get_pretrained(model_name, device=device)
        sample_rate = int(model.sample_rate)

        def generate(prompt: str, seconds: float, seed: int) -> np.ndarray:
            torch.manual_seed(seed)
            model.set_generation_params(duration=seconds, top_k=250, cfg_coef=3.0)
            with torch.inference_mode():
                wav = model.generate([prompt], progress=False)
            return wav[0, 0].detach().cpu().float().numpy()

    return generate, sample_rate, device


def variants_for(entry: dict, base_count: int, suffix_count: int) -> int:
    """
    How many renders a prompt gets, allocated by how much it actually plays.

    WHY THIS IS NOT UNIFORM. The library was first built with 4 renders of each of
    the 20 prompts, on the assumption that all 20 were live. After the 2026-08-16
    hysteresis fix the trend suffix is calibrated above the measured noise ceiling
    and cannot fire in a normal session, so the controller only ever emits the five
    SUFFIX-FREE base prompts - and under the default targets, only rungs 1-3 of those.

    Replaying PILOT01 through the fixed controller, a 17-minute session reached 12 of
    80 segments. The prompt covering 96% of it had 4 renders: 32 seconds of unique
    audio looping for sixteen minutes. That is a worse failure than the chatter it
    replaced, for a relaxation intervention.

    So base prompts get the renders and suffixed prompts stay at the low count as
    insurance against a future target_z or a change to the hysteresis thresholds.
    Rendering 32 of all 20 prompts would cost three hours of GPU to produce audio the
    controller cannot select.
    """
    return base_count if entry["variant"] == "base" else suffix_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the precomputed segment library")
    parser.add_argument("--out", default="library", help="library root directory")
    parser.add_argument("--variants-base", type=int, default=32,
                        help="renders for the suffix-free BASE prompts, which are the only "
                             "ones the controller reaches once hysteresis is applied. These "
                             "carry essentially the whole session, so they need the variety.")
    parser.add_argument("--variants", type=int, default=4,
                        help="independent renders per prompt. Raising this and re-running "
                             "adds only the new ones - existing segments are kept.")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="segment length; must match MusicConfig.segment_seconds")
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "audiocraft"])
    parser.add_argument("--dry-run", action="store_true", help="list prompts and exit")
    args = parser.parse_args()

    prompts = enumerate_prompts()

    reachable = [p for p in prompts if p["reachable_default_targets"]]

    print("=" * 74)
    print(f"PROMPT SPACE: {len(prompts)} distinct prompts reachable from build_prompt()")
    print("=" * 74)
    for entry in prompts:
        mark = " " if entry["reachable_default_targets"] else "*"
        print(f" {mark}rung {entry['rung']}  {entry['variant']:<9} {entry['prompt'][:76]}")
    print()
    print(f"  {len(reachable)} of {len(prompts)} are reachable with the default "
          f"targets z={_DEFAULT_TARGETS}")
    print("  * = needs a wider target_z than either arm currently uses; rendered as insurance.")
    if len(reachable) < len(prompts):
        dead = sorted({p["rung"] for p in prompts if not p["reachable_default_targets"]})
        print(f"  ladder rungs {dead} are unreachable under the current design - see "
              f"enumerate_prompts.__doc__")
    print()
    planned = sum(variants_for(p, args.variants_base, args.variants) for p in prompts)
    n_base = sum(1 for p in prompts if p["variant"] == "base")
    print(f"  {n_base} base prompts x {args.variants_base} renders + "
          f"{len(prompts) - n_base} suffixed x {args.variants} = {planned} segments, "
          f"{args.seconds:g} s each")
    print(f"  base prompts carry the session; suffixed ones are insurance (see variants_for)")

    if args.dry_run:
        return 0

    segments_dir = os.path.join(args.out, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")

    # Resume: anything already on disk and recorded in the manifest is kept. This
    # matters because a full build is ~20 minutes of GPU time and an interruption
    # partway through should cost one segment, not the whole run.
    existing: dict[str, dict] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as fh:
            for entry in json.load(fh).get("prompts", []):
                for seg in entry.get("segments", []):
                    if os.path.exists(os.path.join(args.out, seg["file"])):
                        existing[seg["file"]] = seg
        print(f"  resuming: {len(existing)} segments already rendered")

    todo = sum(1 for p in prompts
               for v in range(variants_for(p, args.variants_base, args.variants))
               if f"segments/rung{p['rung']}_{p['variant']}_v{v}.wav" not in existing)
    if todo == 0:
        print("  nothing to render; library is already complete")
    print(f"  to render this run: {todo}")
    print()

    import soundfile as sf

    generate, sample_rate, device = None, 32000, "cpu"
    if todo:
        print(f"Loading {args.model} via {args.backend}...")
        t0 = time.time()
        generate, sample_rate, device = load_generator(args.model, args.backend)
        print(f"  loaded on {device} in {time.time() - t0:.1f} s\n")

    manifest = {
        "created": datetime.now().isoformat(),
        "model": args.model,
        "backend": args.backend,
        "device": device,
        "sample_rate": sample_rate,
        "segment_seconds": args.seconds,
        "variants_per_prompt": args.variants,
        "variants_per_base_prompt": args.variants_base,
        "target_rms": _TARGET_RMS,
        "prompts": [],
    }

    want_samples = int(args.seconds * sample_rate)
    done = 0
    build_started = time.time()

    for entry in prompts:
        segments = []
        for variant_index in range(variants_for(entry, args.variants_base, args.variants)):
            rel = f"segments/rung{entry['rung']}_{entry['variant']}_v{variant_index}.wav"

            if rel in existing:
                segments.append(existing[rel])
                continue

            # Seed is a pure function of position, so a rebuild reproduces the same
            # library and adding variants never disturbs existing ones.
            seed = 1000 * (entry["rung"] + 1) + 17 * variant_index + len(entry["variant"])

            t0 = time.time()
            audio = generate(entry["prompt"], args.seconds, seed)
            elapsed = time.time() - t0

            audio = audio[:want_samples]
            if audio.size < want_samples:
                audio = np.pad(audio, (0, want_samples - audio.size))
            audio = normalize(audio)

            sf.write(os.path.join(args.out, rel), audio, sample_rate, subtype="PCM_16")

            segments.append({
                "file": rel,
                "seed": seed,
                "duration_s": args.seconds,
                "generation_s": round(elapsed, 2),
                "rms": round(float(np.sqrt(np.mean(audio ** 2))), 4),
                "peak": round(float(np.max(np.abs(audio))), 4),
            })
            done += 1
            print(f"  [{done:>3}/{todo}] rung {entry['rung']} {entry['variant']:<9} "
                  f"v{variant_index}  {elapsed:5.1f}s  ({elapsed / args.seconds:.2f}x realtime)")

            # Write the manifest after every segment. An interrupted build then
            # leaves a valid, smaller library rather than orphaned WAVs.
            manifest["prompts"] = _merge(manifest["prompts"], entry, segments)
            _write_manifest(manifest_path, manifest, prompts)

        manifest["prompts"] = _merge(manifest["prompts"], entry, segments)

    _write_manifest(manifest_path, manifest, prompts)

    total_audio = planned * args.seconds
    print()
    print("=" * 74)
    print(f"  {planned} segments, {total_audio / 60:.1f} minutes of audio")
    print(f"  rendered {done} this run in {(time.time() - build_started) / 60:.1f} minutes")
    print(f"  manifest: {manifest_path}")
    print("=" * 74)
    return 0


def _merge(existing_entries: list, entry: dict, segments: list) -> list:
    """Replace this prompt's entry, keeping manifest order stable."""
    out = [e for e in existing_entries if e["prompt"] != entry["prompt"]]
    out.append({**entry, "segments": segments})
    return out


def _write_manifest(path: str, manifest: dict, prompt_order: list[dict]) -> None:
    order = {p["prompt"]: i for i, p in enumerate(prompt_order)}
    payload = dict(manifest)
    payload["prompts"] = sorted(manifest["prompts"], key=lambda e: order.get(e["prompt"], 999))
    payload["segment_count"] = sum(len(e["segments"]) for e in payload["prompts"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
