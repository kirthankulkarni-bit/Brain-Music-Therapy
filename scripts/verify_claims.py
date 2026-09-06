"""
verify_claims.py - regenerate every number the preprint cites, and check it.

WHY

A preprint's numbers get copied from a terminal into prose once, and then the code
moves. Six weeks later the manuscript says 1.14x and the repository says 1.05x, and
nobody knows which is right or when it changed. That is how honest projects end up
with unreproducible papers.

So every headline claim is recorded here with the value asserted in the manuscript and
the computation that produces it, straight from the artefacts on disk. Running this
before submission tells you whether the paper still describes the code.

A failure here is not necessarily a bug. It means a number moved, and the manuscript
has to move with it - or the claim was wrong. Either way it must be looked at rather
than rounded away, so the tolerances are tight.

Usage:
    python scripts/verify_claims.py
    python scripts/verify_claims.py --verbose
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


def _bench(pattern: str) -> list[dict]:
    return [json.load(open(f, encoding="utf-8"))
            for f in sorted(glob.glob(os.path.join(_ROOT, "benchmarks", pattern)))]


def _median_gen(run: dict, precision: str, duration: float) -> float:
    for row in run.get("musicgen") or []:
        if row["precision"] == precision and row["duration_s"] == duration:
            return float(row["median_generation_s"])
    return float("nan")


def _pilot() -> dict:
    # real_sessions, not glob: a --demo or --mock run must not be able to move a
    # manuscript number by existing. See session_logger._SYNTHETIC_SOURCES.
    from session_logger import load_session, real_sessions
    dirs = real_sessions(os.path.join(_ROOT, "sessions", "PILOT*"))
    if not dirs:
        raise FileNotFoundError("no real PILOT session on disk")
    return load_session(dirs[-1])


def _pilot_z() -> np.ndarray:
    z = np.asarray([w["z"] for w in _pilot()["windows"]
                    if w.get("phase") == "intervention" and w.get("valid")
                    and isinstance(w.get("z"), (int, float))], dtype=float)
    return z[np.isfinite(z)]


# ---------------------------------------------------------------- the claims


def claim_t4_best_realtime() -> tuple[float, str]:
    """Best realtime factor across all T4 runs and precisions."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    best = min(row["median_generation_s"] / row["duration_s"]
               for r in runs for row in r["musicgen"])
    return best, f"{len(runs)} T4 runs, all precisions"


def claim_t4_precision_ordering() -> tuple[float, str]:
    """Fraction of run x duration cells where fp32 < fp16-half < fp16."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    cells = ok = 0
    for r in runs:
        for d in (4.0, 8.0):
            a, b, c = (_median_gen(r, "fp32", d), _median_gen(r, "fp16-half", d),
                       _median_gen(r, "fp16", d))
            if all(np.isfinite([a, b, c])):
                cells += 1
                ok += int(a < b < c)
    return ok / max(1, cells), f"{ok} of {cells} cells"


def claim_t4_fp16half_vs_fp32() -> tuple[float, str]:
    """fp16-half speedup over fp32 at 8 s, median across runs."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    a = np.median([_median_gen(r, "fp32", 8.0) for r in runs])
    b = np.median([_median_gen(r, "fp16-half", 8.0) for r in runs])
    return a / b, "median of 3 T4 runs at 8 s"


def claim_t4_between_run_variance() -> tuple[float, str]:
    """Worst max/min across T4 runs for any configuration."""
    runs = _bench("latency_colab-tesla-t4-run*.json")
    worst = 0.0
    for p in ("fp32", "fp16", "fp16-half"):
        for d in (4.0, 8.0):
            v = [_median_gen(r, p, d) for r in runs]
            v = [x for x in v if np.isfinite(x)]
            if len(v) > 1:
                worst = max(worst, max(v) / min(v))
    return worst, "worst config across 3 T4 runs"


def claim_pilot_effective_n() -> tuple[float, str]:
    z = _pilot_z()
    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    return z.size * (1 - rho) / (1 + rho), f"AR(1) from {z.size} windows"


def claim_pilot_autocorrelation() -> tuple[float, str]:
    z = _pilot_z()
    return float(np.corrcoef(z[:-1], z[1:])[0, 1]), "lag-1 of PILOT01 intervention z"


def claim_pilot_rejection() -> tuple[float, str]:
    rows = [w for w in _pilot()["windows"] if w.get("phase") == "intervention"]
    return sum(1 for w in rows if not w.get("valid")) / len(rows), f"{len(rows)} windows"


def claim_chatter_before() -> tuple[float, str]:
    """Prompt changes actually logged in PILOT01."""
    audio = _pilot()["audio"]
    return float(sum(1 for i in range(1, len(audio))
                     if audio[i].get("prompt") != audio[i - 1].get("prompt"))), \
        f"{len(audio)} audio events"


def claim_chatter_after() -> tuple[float, str]:
    """Prompt changes when the same z is replayed through the fixed controller."""
    from music_engine import build_prompt
    z = _pilot_z()
    prev, changes = None, 0
    for v in z:
        p = build_prompt(float(v), -1.0, previous_prompt=prev)
        if prev is not None and p != prev:
            changes += 1
        prev = p
    return float(changes), f"replay of {z.size} windows"


def claim_slowest_realtime_factor() -> tuple[float, str]:
    """
    Worst realtime factor across every benchmark cell on disk.

    With claim_t4_best_realtime (the fastest, 1.05x) this pins both ends of the range the
    library argument rests on: generation is slower than realtime everywhere measured, so
    a precomputed library is not a compromise. The fastest end was asserted from the
    start; the slowest was quoted in build_library.py and checked by nothing.
    """
    cells = [(row["median_generation_s"] / row["duration_s"], os.path.basename(f))
             for f in sorted(glob.glob(os.path.join(_ROOT, "benchmarks", "latency_*.json")))
             for row in (json.load(open(f, encoding="utf-8")).get("musicgen") or [])]
    worst = max(cells)
    return float(worst[0]), f"{len(cells)} cells, slowest on {worst[1]}"


def claim_pilot_rung_occupancy() -> tuple[float, str]:
    """
    Fraction of PILOT01's intervention spent on the single dominant ladder rung.

    analysis_plan.md section 7 records this as a known limitation before the fact -
    "PILOT01 used rung 1 for 96% of the session" - and it is the basis for the
    instruction not to describe the controller as five graded levels in a methods
    section. It is in the FROZEN pre-registration and was regenerated by nothing.
    """
    from music_engine import _ENERGY_LADDER

    rows = [w for w in _pilot()["windows"]
            if w.get("phase") == "intervention" and w.get("prompt")]
    counts: dict = {}
    for w in rows:
        rung = next((i for i, base in enumerate(_ENERGY_LADDER)
                     if w["prompt"].startswith(base)), None)
        counts[rung] = counts.get(rung, 0) + 1
    top = max(counts.values())
    return top / len(rows), (f"{len(counts)} rungs used over {len(rows)} windows: "
                             + ", ".join(f"rung {k} {v / len(rows):.1%}"
                                         for k, v in sorted(counts.items())))


def claim_library_clipping_bound() -> tuple[float, str]:
    from library_engine import LibraryConfig
    man = json.load(open(os.path.join(_ROOT, "library", "manifest.json"), encoding="utf-8"))
    peaks = [s["peak"] for e in man["prompts"] for s in e["segments"]]
    return max(peaks) * (2 ** 0.5) * LibraryConfig().output_gain, \
        f"{len(peaks)} segments, gain {LibraryConfig().output_gain}"


def claim_library_dominant_variants() -> tuple[float, str]:
    """Renders available for the prompt that carries a relaxation session."""
    from music_engine import _ENERGY_LADDER
    man = json.load(open(os.path.join(_ROOT, "library", "manifest.json"), encoding="utf-8"))
    for e in man["prompts"]:
        if e["prompt"] == _ENERGY_LADDER[1]:
            return float(len(e["segments"])), "rung 1 base, 96% of PILOT01"
    return float("nan"), "not found"


def claim_latency_budget() -> tuple[float, str]:
    """End-to-end worst case with the library engine."""
    from library_engine import LibraryConfig
    runs = _bench("latency_colab-tesla-t4-run1.json")
    analysis = runs[0]["analysis"][1]["total_analysis_latency_s"]
    return analysis + LibraryConfig().crossfade_seconds, \
        f"{analysis:g} s analysis + {LibraryConfig().crossfade_seconds:g} s crossfade"


def claim_coupling_recovers_lag() -> tuple[float, str]:
    """Ground-truth check: the estimator must return the lag it was given."""
    from analyze_session import coupling_index
    from validate_coupling import build_session
    got = coupling_index(build_session(6.0, retrospective=False, seed=1),
                         n_permutations=120).get("aci_peak_lag_s", float("nan"))
    return float(got), "synthetic session, true lag +6.0 s"


def claim_alpha_validation_ratio() -> tuple[float, str]:
    """Eyes-closed alpha increase - the evidence the rig measures cortex."""
    from session_logger import load_session
    from session_logger import real_sessions
    dirs = real_sessions(os.path.join(_ROOT, "sessions", "alphatest*"))
    session = load_session(dirs[-1])
    rows = [w for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")
            and isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"])]
    a = np.log10(np.asarray([w["alpha"] for w in rows], dtype=float))
    closed = np.asarray([w["phase"] == "eyes_closed" for w in rows], dtype=bool)
    return float(10 ** (a[closed].mean() - a[~closed].mean())),         f"{closed.sum()} closed / {(~closed).sum()} open windows"


def claim_alpha_ratio_at_deployed_rejection() -> tuple[float, str]:
    """
    The eyes-closed alpha ratio at the rejection threshold the pipeline actually uses.

    The headline 2.13x is computed from the alpha session's logged windows, and that
    session was recorded at 150 uV - a threshold this project abandoned, in response to
    the very sensitivity analysis that measures it, in favour of 350 uV. At 150 uV the
    conditions retain 72 open against 142 closed windows, a 2:1 imbalance that inflates
    the contrast by stripping blink power from the open condition only.

    At the deployed 350 uV the conditions are near-balanced (1.05) and the ratio is
    1.90x. Both numbers are real; the second is the one the deployed pipeline would
    produce, so it is the one to lead with and it needs to be regenerated rather than
    remembered.
    """
    from alpha_sensitivity import ratio_at_threshold

    return ratio_at_threshold(350.0)


def claim_alpha_effect_survives_every_threshold() -> tuple[float, str]:
    """
    Fraction of rejection thresholds at which the eyes-closed effect stays significant.

    This is the claim that matters more than any single ratio: the effect is not
    manufactured by the rejection rule. It must be 1.0 - significant at every threshold
    swept, including no rejection at all.
    """
    from alpha_sensitivity import significance_across_thresholds

    frac, detail = significance_across_thresholds()
    return frac, detail


def claim_channel_mismatch_af() -> tuple[float, str]:
    """Eyes-closed alpha ratio on AF7/AF8 - the channels the study actually uses."""
    from eeg_features import FeatureConfig, FeatureExtractor
    from session_logger import load_raw, load_session

    from session_logger import real_sessions
    d = real_sessions(os.path.join(_ROOT, "sessions", "alphatest*"))[-1]
    chans = load_raw(d)[:, 1:].T.astype(float)
    session = load_session(d)
    tl = [(float(w["elapsed_s"]), w["phase"]) for w in session["windows"]
          if w.get("phase") in ("eyes_open", "eyes_closed")]

    def phase_at(t):
        prev = None
        for tt, ph in tl:
            if tt > t:
                return prev
            prev = ph
        return prev

    cfg = FeatureConfig(sampling_rate=256.0, frontal_channels=("AF7", "AF8"))
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    op, cl = [], []
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        f = ex.extract(chans[:, s0:s0 + nw])
        if not (f.valid and np.isfinite(f.alpha) and f.alpha > 0):
            continue
        ph = phase_at((s0 + nw) / 256.0 + 6.25)
        (op if ph == "eyes_open" else cl if ph == "eyes_closed" else []).append(f.alpha)
    if len(op) < 10 or len(cl) < 10:
        return float("nan"), "insufficient"
    return float(10 ** (np.log10(cl).mean() - np.log10(op).mean())),         f"{len(op)} open / {len(cl)} closed windows"


_DEAP_CACHE: dict = {}


def _deap():
    """DEAP participant 1, loaded once. The file is 103 MB and two claims read it."""
    if not _DEAP_CACHE:
        import pickle
        path = os.path.join(_ROOT, "s01.dat")
        if not os.path.exists(path):
            _DEAP_CACHE["missing"] = True
            return _DEAP_CACHE
        with open(path, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        _DEAP_CACHE.update(data=np.asarray(d["data"])[:, :32, :],
                           arousal=np.asarray(d["labels"])[:, 1])
    return _DEAP_CACHE


def claim_deap_arousal_rho() -> tuple[float, str]:
    """Spearman rho between log(beta/alpha) and DEAP self-reported arousal, AF3/AF4."""
    from scipy import stats as spstats
    from validate_index_deap import DEAP_CHANNELS, trial_index

    c = _deap()
    if c.get("missing"):
        return float("nan"), "s01.dat absent"
    picks = [DEAP_CHANNELS.index(ch) for ch in ("AF3", "AF4")]
    idx = np.array([trial_index(c["data"][t], picks) for t in range(c["data"].shape[0])])
    ok = np.isfinite(idx)
    rho, _ = spstats.spearmanr(idx[ok], c["arousal"][ok])
    return float(rho), f"{ok.sum()} trials, AF3/AF4"


def claim_deap_montages_positive() -> tuple[float, str]:
    """
    Montages where the index correlates POSITIVELY with arousal, of seven.

    The construct-validity result does not rest on any single correlation - at n = 40
    none of them individually reaches 0.05. It rests on the direction being consistent
    across independent channel pairs, which under no association is a coin flip each.
    Seven of seven is what makes the sign test significant, and it was asserted nowhere.
    """
    from scipy import stats as spstats
    from validate_index_deap import DEAP_CHANNELS, trial_index

    c = _deap()
    if c.get("missing"):
        return float("nan"), "s01.dat absent"
    rhos = []
    for pair in (("AF3", "AF4"), ("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"),
                 ("T7", "T8"), ("P3", "P4"), ("O1", "O2")):
        picks = [DEAP_CHANNELS.index(ch) for ch in pair]
        v = np.array([trial_index(c["data"][t], picks)
                      for t in range(c["data"].shape[0])])
        m = np.isfinite(v)
        if m.sum() < 12:
            continue
        rho, _ = spstats.spearmanr(v[m], c["arousal"][m])
        rhos.append(rho)
    pos = sum(r > 0 for r in rhos)
    sign_p = spstats.binomtest(pos, len(rhos), 0.5).pvalue
    return float(pos), f"of {len(rhos)} montages, sign test p = {sign_p:.3f}"


def claim_streaming_latency_budget() -> tuple[float, str]:
    """Total analysis latency of the low-latency estimator."""
    from eeg_features import FeatureConfig, StreamingBandPower
    est = StreamingBandPower(FeatureConfig(sampling_rate=256.0), tau_seconds=0.25, order=4)
    b = est.latency_budget()
    return float(b["total_analysis_latency_s"]), "order 4, tau 0.25"


# claim -> (function, value asserted in the manuscript, tolerance)
def claim_analysis_share_of_budget() -> tuple[float, str]:
    """
    Fraction of the end-to-end budget spent in the analysis path, not in audio.

    This is what makes the latency decomposition useful rather than merely present: work
    aimed at faster GENERATION optimises the smaller term. The preprint states 85% twice
    and nothing regenerated it until 9/5.
    """
    from library_engine import LibraryConfig
    analysis = _bench("latency_colab-tesla-t4-run1.json")[0]["analysis"][1][
        "total_analysis_latency_s"]
    crossfade = LibraryConfig().crossfade_seconds
    return analysis / (analysis + crossfade),         f"{analysis:g} s analysis of {analysis + crossfade:g} s total"


def claim_laptop_between_run_variance() -> tuple[float, str]:
    """
    Worst max/min across the three laptop runs of the same configuration.

    The T4 equivalent (1.18) was already asserted; this one was not, and it is the more
    striking of the pair - a single run of the probe is not a measurement on a thermally
    limited part, which is why the paper reports a range rather than a point estimate.
    """
    runs = _bench("latency_nitro5-1650ti_run*.json")
    worst = 0.0
    for p in ("fp32", "fp16", "fp16-half"):
        for d in (4.0, 8.0):
            v = [x for x in (_median_gen(r, p, d) for r in runs) if np.isfinite(x)]
            if len(v) > 1:
                worst = max(worst, max(v) / min(v))
    return worst, f"worst config across {len(runs)} laptop runs"


def claim_autocorrelation_overstatement() -> tuple[float, str]:
    """
    How much treating windows as independent overstates the evidence: sqrt(n / n_eff).

    The factor between p = 0.05 and p = 0.4. Both inputs were asserted separately; the
    ratio the paper actually quotes was not.
    """
    z = _pilot_z()
    rho = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    n_eff = z.size * (1 - rho) / (1 + rho)
    return float((z.size / n_eff) ** 0.5), f"{z.size} windows, n_eff {n_eff:.1f}"


_SWEEP_CACHE: dict = {}


def _sweep():
    """
    The estimator sweep, computed once: {name: {latency_s, d, n_eff_per_min, info}}.

    This is the evidence for C2 - "the analysis latency is a dominated configuration",
    which the preprint calls its strongest result - and nothing regenerated any of it
    until 9/5. The n = 7 projection in finding_analysis_latency.md is derived from the
    d and ind/min columns, so the study's feasibility argument rests on them too.

    The sweep refuses to report below r = 0.9 against the session log, and that gate is
    re-applied here rather than assumed.
    """
    if not _SWEEP_CACHE:
        import estimator_sweep as es
        from session_logger import real_sessions
        d = real_sessions(os.path.join(_ROOT, "sessions", "alphatest*"))[-1]
        session, chans, pair, timeline = es.load(d)
        offset, r = es.find_offset(chans, pair, session, timeline)
        if not np.isfinite(r) or r < 0.9:
            raise RuntimeError(f"sweep reproduction r = {r:.3f} < 0.9")
        rows = {}
        for name, fn, kw in es.ESTIMATORS:
            t, y = fn(chans, pair, **kw)
            sc = es.score(t, y, timeline, offset)
            sc["info"] = (sc["d"] * np.sqrt(sc["n_eff_per_min"])
                          if np.isfinite(sc["d"]) else float("nan"))
            rows[name] = sc
        _SWEEP_CACHE.update(rows=rows, r=r, names=[n for n, _, _ in es.ESTIMATORS])
    return _SWEEP_CACHE


def claim_deployed_detection_latency() -> tuple[float, str]:
    """Median seconds from a real state change to the estimator crossing the midpoint."""
    c = _sweep()
    row = c["rows"][c["names"][0]]
    return float(row["latency_s"]), f"deployed 4 s / 1 s / tau 3, r = {c['r']:.3f}"


def claim_deployed_info_per_min() -> tuple[float, str]:
    """d x sqrt(independent observations per minute) for the deployed configuration."""
    c = _sweep()
    row = c["rows"][c["names"][0]]
    return float(row["info"]), f"d {row['d']:.2f}, ind/min {row['n_eff_per_min']:.1f}"


def claim_retuned_info_per_min() -> tuple[float, str]:
    """The same for 2 s / 0.5 s / tau 0.5 - the configuration the dwell made usable."""
    c = _sweep()
    row = c["rows"]["pipeline 2s win, 0.5s hop, t=0.5"]
    return float(row["info"]), f"d {row['d']:.2f}, ind/min {row['n_eff_per_min']:.1f}"


def claim_alternatives_dominating_deployed() -> tuple[float, str]:
    """
    How many alternatives beat the deployed configuration on BOTH axes.

    Both, not either: faster to detect AND more information per minute. That is what
    "dominated" means and what makes this stronger than a trade-off.
    """
    c = _sweep()
    base = c["rows"][c["names"][0]]
    n = sum(1 for name in c["names"][1:]
            if c["rows"][name]["latency_s"] < base["latency_s"]
            and c["rows"][name]["info"] > base["info"])
    return float(n), f"of {len(c['names']) - 1} alternatives tested"


# Simulations per power cell. 1000 leaves required_n straddling a grid step (25 or 30
# depending on seed); 2000 pins it at 25 with the power estimate varying by 0.010 across
# seeds. The tolerances on the two power claims below are that measured spread, not a
# guess - a stochastic quantity asserted to more precision than its own noise is a test
# that fails for the wrong reason.
_POWER_SIMS = 2000


def _power_ctx():
    from power_analysis import session_stats
    from session_logger import real_sessions
    return session_stats(real_sessions(os.path.join(_ROOT, "sessions", "PILOT*"))[-1])


def claim_power_at_registered_n() -> tuple[float, str]:
    """
    Power to detect the smallest effect of interest at the registered n.

    analysis_plan.md section 4 states 38%, and that number is the reason this is a
    feasibility study rather than a test of H1. It is frozen, so it cannot move - but it
    can stop being reproducible, which is worse, because the registered design would then
    rest on a computation nobody can rerun. Nothing checked it until 9/5.
    """
    from power_analysis import power_for
    st = _power_ctx()
    got = power_for(np.random.default_rng(0), 10, 0.15, 0.5, st, True, "z_mean",
                    _POWER_SIMS)
    return float(got), f"paired, n=10 per arm, 0.15 z, {_POWER_SIMS} sims"


def claim_required_n_for_target_effect() -> tuple[float, str]:
    """
    Participants per arm for 80% power at 0.15 z, paired.

    This is the 25 that finding_analysis_latency.md compares its retuned projection of 7
    against, so the whole "infeasible to feasible" argument is anchored on it.
    """
    from power_analysis import required_n
    st = _power_ctx()
    got = required_n(np.random.default_rng(0), 0.15, 0.5, st, True, "z_mean",
                     _POWER_SIMS)
    return float(got), f"paired, 80% power, {_POWER_SIMS} sims"


def claim_trend_noise_to_signal() -> tuple[float, str]:
    """
    How many times larger the trend estimator's noise is than the drift it describes.

    This is the number that removed the trend suffix on 9/5, so it is regenerated rather
    than remembered. Above 1.0 means no threshold can work: placed above the noise it can
    only be crossed by noise, placed low enough to catch real drift it fires constantly.

    If a future estimator ever drives this below 1.0, the trend becomes measurable and
    the suffix is worth reconsidering. See docs/deviations.md.
    """
    z = _pilot_z()
    W = 20
    idx = np.arange(W, dtype=float)
    noise = float(np.array([np.polyfit(idx, z[i - W:i], 1)[0]
                            for i in range(W, z.size)]).std(ddof=1))
    win = 60
    real = max(abs(float(np.polyfit(np.arange(win, dtype=float), z[i:i + win], 1)[0]))
               for i in range(0, z.size - win, win // 4))
    return noise / real, f"noise {noise:.4f} vs largest 60 s drift {real:.4f}"


def claim_reachable_prompt_space() -> tuple[float, str]:
    """
    Distinct prompts build_prompt can emit. Was 20 with the trend suffix, now 5.

    Asserted because build_library derives the render list from this sweep, so a silent
    growth here means the library stops covering the controller - the exact failure
    verify_library exists to catch, one step earlier.
    """
    from build_library import enumerate_prompts
    ps = enumerate_prompts()
    n_reach = sum(1 for p in ps if p["reachable_default_targets"])
    return float(len(ps)), f"{n_reach} reachable under the two default targets"


_REPLAY_CACHE: dict = {}


def _replay_ctx():
    """
    The PILOT01 reconstruction, computed once and reused by the claims below.

    Reconstructing z from raw runs the real FeatureExtractor over ~1300 s of 256 Hz
    data, twice (two estimator configurations). Doing that per claim would put minutes
    into a suite that has to stay cheap enough to run before every commit.
    """
    if not _REPLAY_CACHE:
        import controller_replay as cr
        from session_logger import real_sessions
        d = real_sessions(os.path.join(_ROOT, "sessions", "PILOT*"))[-1]
        session, chans, ts, pair, base = cr.load(d)
        t0, v0 = cr.reconstruct(chans, ts, pair, cr.DEPLOYED[0], cr.DEPLOYED[1], 0.001)
        offset, r = cr.align(t0, v0, session)
        _REPLAY_CACHE.update(cr=cr, session=session, chans=chans, ts=ts, pair=pair,
                             base=base, offset=offset, r=r)
    return _REPLAY_CACHE


def _replay_z(config):
    c = _replay_ctx()
    key = ("z", config)
    if key not in c:
        c[key] = c["cr"].z_series(c["chans"], c["ts"], c["pair"], c["session"],
                                  c["base"], *config, c["offset"])
    return c[key]


def claim_replay_fidelity() -> tuple[float, str]:
    """
    How well the offline reconstruction reproduces the deployed pipeline.

    Every retuned-estimator number depends on this, because no session log exists for a
    configuration that was never run. Below 0.9 the replay describes a different system
    - a hand-rolled Welch scores 0.05 - so this is the load-bearing assumption behind
    the whole finding, and it is asserted rather than assumed.
    """
    c = _replay_ctx()
    return float(c["r"]), f"offset {c['offset']:+.2f} s against the session log"


def claim_retuned_chatter_no_dwell() -> tuple[float, str]:
    """Switches arriving inside a crossfade at 2 s / 0.5 s / tau 0.5, no dwell."""
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].RETUNED)
    m = c["cr"].replay(z, t, -1.0, 0.0, 0.0, 1.0)
    return float(m["under_crossfade"]), f"{m['changes']} changes over {z.size} windows"


def claim_retuned_chatter_with_dwell() -> tuple[float, str]:
    """
    The same configuration with a dwell of one crossfade.

    This is the result that makes the retuning usable at all: a dwell of at least one
    crossfade is precisely the condition for no switch arriving before the previous
    crossfade finishes, so the count is zero by construction rather than by tuning.
    """
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].RETUNED)
    m = c["cr"].replay(z, t, -1.0, 0.0, 1.0, 1.0)
    return float(m["under_crossfade"]), f"{m['changes']} changes, median gap " \
                                        f"{m['median_gap']:.1f} s"


def claim_ladder_margin_responds() -> tuple[float, str]:
    """
    Prompt changes at the deployed settings with ladder_margin 0.25.

    Asserted because the interesting failure is ZERO. build_prompt used to derive the
    previous rung from the previous PROMPT, which latched the controller on the goal
    rung and produced no changes at all across a whole session. A count above zero is
    the property that broke; the exact value is secondary.
    """
    c = _replay_ctx()
    z, t = _replay_z(c["cr"].DEPLOYED)
    m = c["cr"].replay(z, t, -1.0, 0.25, 0.0, 1.0)
    return float(m["changes"]), f"median gap {m['median_gap']:.1f} s over {z.size} windows"


CLAIMS = {
    "T4 best realtime factor":            (claim_t4_best_realtime,        1.05,   0.01),
    "T4 fp32<fp16-half<fp16 consistency": (claim_t4_precision_ordering,   1.00,   0.001),
    "T4 fp16-half vs fp32 at 8 s":        (claim_t4_fp16half_vs_fp32,     0.952,  0.01),
    "T4 between-run max/min":             (claim_t4_between_run_variance, 1.18,   0.02),
    "PILOT01 lag-1 autocorrelation":      (claim_pilot_autocorrelation,   0.953,  0.005),
    "PILOT01 effective sample size":      (claim_pilot_effective_n,       25.3,   0.5),
    "PILOT01 intervention rejection":     (claim_pilot_rejection,         0.131,  0.005),
    "prompt changes before the fix":      (claim_chatter_before,          491,    1),
    "prompt changes after the fix":       (claim_chatter_after,           24,     2),
    "library clipping bound":             (claim_library_clipping_bound,  0.980,  0.005),
    "renders on the dominant prompt":     (claim_library_dominant_variants, 32,   0),
    "end-to-end budget, library":         (claim_latency_budget,          6.5,    0.05),
    "coupling recovers a +6 s lag":       (claim_coupling_recovers_lag,   6.0,    1.0),
    "eyes-closed alpha ratio":            (claim_alpha_validation_ratio,  2.13,   0.02),
    "same effect on AF7/AF8":             (claim_channel_mismatch_af,     0.91,   0.03),
    "DEAP arousal rho (AF3/AF4)":         (claim_deap_arousal_rho,        0.303,  0.02),
    "streaming estimator budget":         (claim_streaming_latency_budget, 0.413, 0.01),
    "replay fidelity vs the log":         (claim_replay_fidelity,         0.991,  0.005),
    "retuned, no dwell: inside a xfade":  (claim_retuned_chatter_no_dwell, 136,   4),
    "retuned, 1 s dwell: inside a xfade": (claim_retuned_chatter_with_dwell, 0,   0),
    "ladder margin 0.25 still responds":  (claim_ladder_margin_responds,  8,      2),
    "trend noise / genuine drift":        (claim_trend_noise_to_signal,   1.77,   0.05),
    "distinct prompts build_prompt emits":(claim_reachable_prompt_space,  5,      0),
    "power at the registered n = 10":     (claim_power_at_registered_n,   0.389,  0.02),
    "participants per arm for 0.15 z":    (claim_required_n_for_target_effect, 25, 5),
    "deployed detection latency":         (claim_deployed_detection_latency, 5.67,  0.05),
    "deployed info per minute":           (claim_deployed_info_per_min,      2.14,  0.03),
    "retuned info per minute":            (claim_retuned_info_per_min,       4.20,  0.03),
    "alternatives dominating deployed":   (claim_alternatives_dominating_deployed, 8, 0),
    "analysis share of the budget":       (claim_analysis_share_of_budget,   0.846, 0.005),
    "laptop between-run max/min":         (claim_laptop_between_run_variance, 1.96, 0.02),
    "autocorrelation overstatement":      (claim_autocorrelation_overstatement, 6.4, 0.05),
    "DEAP montages positive (of 7)":      (claim_deap_montages_positive,     7,     0),
    "alpha ratio at deployed 350 uV":     (claim_alpha_ratio_at_deployed_rejection, 1.90, 0.02),
    "effect survives every threshold":    (claim_alpha_effect_survives_every_threshold, 1.0, 0.0),
    "slowest realtime factor measured":   (claim_slowest_realtime_factor,   6.27,   0.02),
    "PILOT01 dominant rung occupancy":    (claim_pilot_rung_occupancy,      0.959,  0.005),
}


# Claims that dominate the runtime, and what they cost. --quick skips exactly these.
#
# They are the ones that recompute something large rather than read a stored artefact:
# the power claims run Monte Carlo studies, the DEAP claims Welch a 103 MB recording
# across 40 trials, and the replay claims run the feature extractor over 20 minutes of
# 256 Hz data twice. Everything else is arithmetic on JSON and takes milliseconds.
#
# Named individually rather than flagged by a duration threshold, so adding a slow claim
# does not silently join the skip list.
_SLOW = frozenset({
    "power at the registered n = 10",
    "participants per arm for 0.15 z",
    "DEAP arousal rho (AF3/AF4)",
    "DEAP montages positive (of 7)",
    "replay fidelity vs the log",
    "retuned, no dwell: inside a xfade",
    "retuned, 1 s dwell: inside a xfade",
    "ladder margin 0.25 still responds",
    "coupling recovers a +6 s lag",
})


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every number the preprint cites")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="skip the claims that recompute from raw data. NOT a "
                             "substitute for a full run before committing or submitting.")
    args = parser.parse_args()

    print("=" * 78)
    print("CLAIM VERIFICATION - every headline number, regenerated from artefacts")
    print("=" * 78)
    if args.quick:
        print(f"  --quick: {len(_SLOW)} of {len(CLAIMS)} claims SKIPPED, not verified.")
        print("  Run without --quick before committing or submitting.")
        print()
    print(f"  {'claim':<36}{'asserted':>10}{'measured':>11}  status")
    print("  " + "-" * 74)

    failures = []
    skipped = 0
    for name, (fn, asserted, tol) in CLAIMS.items():
        if args.quick and name in _SLOW:
            skipped += 1
            continue
        try:
            measured, source = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<36}{asserted:>10.3f}{'ERROR':>11}  {type(exc).__name__}")
            failures.append(name)
            continue
        ok = np.isfinite(measured) and abs(measured - asserted) <= tol
        print(f"  {name:<36}{asserted:>10.3f}{measured:>11.3f}  "
              f"{'ok' if ok else 'MOVED'}")
        if args.verbose:
            print(f"  {'':<36}{'':<21}  source: {source}")
        if not ok:
            failures.append(name)

    print()
    print("=" * 78)
    if failures:
        print(f"  {len(failures)} CLAIM(S) MOVED - the manuscript no longer matches the code:")
        for name in failures:
            print(f"    {name}")
        print("  Update the manuscript, or find out why the number changed. Do not round.")
    else:
        n = len(CLAIMS) - skipped
        if skipped:
            print(f"  All {n} claims run reproduce, {skipped} SKIPPED by --quick.")
            print("  This is not a clean bill of health - rerun in full.")
        else:
            print(f"  All {n} claims reproduce. The manuscript matches the repository.")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
