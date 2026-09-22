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


# The pilot session the MANUSCRIPT DESCRIBES, pinned by name.
#
# These claims are statements about a particular recording, not about "whatever pilot is
# newest". claim_chatter_before asserts 491 prompt changes: that is a historical fact
# about PILOT01's pre-fix audio and the evidence behind section 6.3. PILOT02 will show
# roughly 24, because the defect was fixed.
#
# Before this was pinned, recording PILOT02 would have silently re-based a dozen claims
# onto it - and next_session.md told the operator that claims moving after a session is
# "expected and correct, update the asserted values". Following that instruction would
# have overwritten the documented pre-fix chatter figure with a post-fix one and quietly
# deleted the evidence for the finding.
#
# New pilots are ADDITIVE. To describe a different session, add claims for it rather than
# re-pointing these; to retire PILOT01 deliberately, change this constant in its own
# commit and say why.
PINNED_PILOT = "PILOT01_20260822_153652"

# The alpha-validation recording the manuscript describes - Figure 0, the 2.13x, the
# estimator sweep and every number derived from it. Pinned for the same reason as the
# pilot, decided on 2026-09-17 when a second alpha recording landed.
#
# These claims used to track the NEWEST alpha recording, on the reasoning that the next
# session would replace Figure 0. But whether a new recording replaces the manuscript's
# is a decision about the paper, not a side effect of a file appearing on disk. The
# 2026-09-17 AF7/AF8 recording is weaker and mixed - significant alpha rise, but the
# eye-closure prominence check fails on both frontal channels - so replacing Figure 0 with
# it is exactly the kind of call that must be made deliberately.
#
# To adopt a newer recording: change this constant in its own commit, say why, and update
# the manuscript to match. Until then the newer recording is described by its own claims
# (SECOND_ALPHA below), which are regenerated and visible but overwrite nothing.
PINNED_ALPHA = "alphatest_20260816_015511"

# The 2026-09-17 alpha test: AF7/AF8, good contact, two stream dropouts (11.75 s and
# 20.25 s) and heavy duplicate packets that the recorder filtered. Described separately.
SECOND_ALPHA = "alphatest_20260917_214726"


def _alpha_dir(name: str = PINNED_ALPHA) -> str:
    d = os.path.join(_ROOT, "sessions", name)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"{name} is missing; these claims describe it specifically")
    return d


def _pilot() -> dict:
    from session_logger import is_synthetic, load_session

    path = os.path.join(_ROOT, "sessions", PINNED_PILOT)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{PINNED_PILOT} is missing. These claims describe that recording "
            "specifically; they are not about whichever pilot is newest.")
    if is_synthetic(path):
        raise RuntimeError(f"{PINNED_PILOT} is a synthetic session")
    return load_session(path)


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
    """
    Prompt changes when PILOT01's z is replayed through the controller AS DEPLOYED.

    This asserted 24 until 2026-09-20, measured on the five-rung 1.0 SD ladder without a
    dwell, and that is the number section 6.3 quotes for the trend-suffix fix. Both parts
    of the controller changed: the ladder is now nine rungs at 0.5 SD, and a 30 s dwell is
    on by default. The same replay through build_prompt alone now gives 197 - the finer
    rungs change the music every few seconds - and through the deployed controller, dwell
    included, it gives 33.

    The claim tracks what a participant would hear TODAY, so it measures the governor. The
    historical 24 is preserved where it belongs: in section 6.3 and results_pilot.md, each
    now saying which ladder it was measured on.
    """
    from music_engine import PromptGovernor

    import live_music

    dwell = float(live_music.build_parser().parse_args([]).min_dwell)
    z = _pilot_z()
    gov = PromptGovernor(target_z=-1.0, min_dwell_seconds=dwell)
    prev, changes = None, 0
    for i, v in enumerate(z):
        p = gov.update(float(v), now=float(i))
        if prev is not None and p != prev:
            changes += 1
        prev = p
    return float(changes), f"replay of {z.size} windows, dwell {dwell:g} s"


def claim_analysis_latency_floor() -> tuple[float, str]:
    """
    The lowest analysis-path latency any configuration in this architecture reaches.

    The paper's headline is that the deployed 5.5 s analysis path is a configuration
    rather than a floor. This is the floor: a second-order streaming estimator at
    tau = 0.1 s, whose only delays are filter group delay and the smoother.

    It matters because of what it makes the binding constraint. At 0.189 s the analysis
    path is no longer 85% of the budget - the 1 s crossfade is - which inverts the
    architecture's whole latency story and is the reason section 3 can state a frontier
    rather than only a complaint.
    """
    from eeg_features import FeatureConfig, StreamingBandPower

    cfg = FeatureConfig(sampling_rate=256.0)
    best, where = float("inf"), ""
    for order in (2, 4):
        for tau in (0.1, 0.25, 0.5, 1.0):
            total = StreamingBandPower(cfg, band="alpha", tau_seconds=tau,
                                       order=order).latency_budget()[
                "total_analysis_latency_s"]
            if total < best:
                best, where = total, f"streaming o{order}, tau={tau:g}"
    return float(best), where


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
    """
    Renders available for the prompt that carries a relaxation session.

    The rung is DERIVED from the target rather than hardcoded. It was index 1, which was
    the relaxation goal on the five-rung ladder and is a different piece of music on the
    nine-rung one - the claim returned nan the moment the ladder changed, which is the
    right failure but the wrong cause.
    """
    from music_engine import _ENERGY_LADDER, state_rung
    man = json.load(open(os.path.join(_ROOT, "library", "manifest.json"), encoding="utf-8"))
    goal = _ENERGY_LADDER[state_rung(-1.0)]
    for e in man["prompts"]:
        if e["prompt"] == goal:
            return float(len(e["segments"])), f"the relaxation goal rung, {goal[:28]}..."
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


def _alpha_effect_size(d: str) -> tuple[float, str]:
    """Cohen's d on exactly the windows _logged_alpha_ratio uses, so the pair agrees."""
    from session_logger import load_session
    from scipy import stats as sps

    session = load_session(d)
    rows = [w for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")
            and isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"])]
    a = np.log10(np.asarray([w["alpha"] for w in rows], dtype=float))
    closed = np.asarray([w["phase"] == "eyes_closed" for w in rows], dtype=bool)
    diff = float(a[closed].mean() - a[~closed].mean())
    sd = float(np.sqrt((a[closed].var(ddof=1) + a[~closed].var(ddof=1)) / 2))
    _, p = sps.ttest_ind(a[closed], a[~closed], equal_var=False)
    return diff / sd, f"p = {p:.1e}, {closed.sum()} closed / {(~closed).sum()} open"


def _logged_alpha_ratio(d: str) -> tuple[float, str]:
    from session_logger import load_session
    session = load_session(d)
    rows = [w for w in session["windows"]
            if w.get("phase") in ("eyes_open", "eyes_closed")
            and isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"])]
    a = np.log10(np.asarray([w["alpha"] for w in rows], dtype=float))
    closed = np.asarray([w["phase"] == "eyes_closed" for w in rows], dtype=bool)
    return float(10 ** (a[closed].mean() - a[~closed].mean())),         f"{closed.sum()} closed / {(~closed).sum()} open windows"


def claim_alpha_validation_ratio() -> tuple[float, str]:
    """Eyes-closed alpha increase - the evidence the rig measures cortex."""
    return _logged_alpha_ratio(_alpha_dir())


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

    return ratio_at_threshold(350.0, session_dir=_alpha_dir())


def claim_alpha_effect_survives_every_threshold() -> tuple[float, str]:
    """
    Fraction of rejection thresholds at which the eyes-closed effect stays significant.

    This is the claim that matters more than any single ratio: the effect is not
    manufactured by the rejection rule. It must be 1.0 - significant at every threshold
    swept, including no rejection at all.
    """
    from alpha_sensitivity import significance_across_thresholds

    frac, detail = significance_across_thresholds(session_dir=_alpha_dir())
    return frac, detail


def claim_channel_mismatch_af() -> tuple[float, str]:
    """
    Eyes-closed alpha ratio on AF7/AF8 - the channels the study actually uses.

    This used to align windows to blocks with a HARDCODED +6.25 s clock offset and
    sample-index time. +6.25 s was measured on the August recording only; trap 3 in the
    handoff says the offset must be found by correlation, never assumed, and the
    2026-09-17 recording measured +6.00 s. It also had two stream dropouts (11.75 s and
    20.25 s), after which sample-index time runs behind by the dropout. Both are now taken
    from the sweep's own alignment: the correlated offset and the gap-corrected sample
    clock.
    """
    from eeg_features import FeatureConfig, FeatureExtractor
    import estimator_sweep as es

    d = _alpha_dir()
    c = _sweep(d)
    session, chans, pair, tl, t_samples = es.load(d)
    offset = c["offset"]

    cfg = FeatureConfig(sampling_rate=256.0, frontal_channels=("AF7", "AF8"))
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    op, cl = [], []
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        f = ex.extract(chans[:, s0:s0 + nw])
        if not (f.valid and np.isfinite(f.alpha) and f.alpha > 0):
            continue
        ph = es.phase_at(tl, t_samples[s0 + nw - 1] + 1.0 / 256.0 + offset)
        (op if ph == "eyes_open" else cl if ph == "eyes_closed" else []).append(f.alpha)
    if len(op) < 10 or len(cl) < 10:
        return float("nan"), "insufficient"
    return float(10 ** (np.log10(cl).mean() - np.log10(op).mean())),         f"{len(op)} open / {len(cl)} closed windows, offset {offset:+.2f} s"


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


def _sweep_dir():
    """The alpha recording the manuscript's sweep numbers describe. See PINNED_ALPHA."""
    return _alpha_dir(PINNED_ALPHA)


def _sweep(d: str = None):
    """
    The estimator sweep on one recording, computed once per recording:
    {rows: {name: {latency_s, d, n_eff_per_min, info}}, r, offset, t_samples, names}.

    This is the evidence for C2 - "the analysis latency is a dominated configuration",
    which the preprint calls its strongest result - and nothing regenerated any of it
    until 9/5. The n = 7 projection in finding_analysis_latency.md is derived from the
    d and ind/min columns, so the study's feasibility argument rests on them too.

    The sweep refuses to report below r = 0.9 against the session log, and that gate is
    re-applied here rather than assumed. Time comes from the gap-corrected sample clock;
    see session_logger.sample_clock.
    """
    d = d or _sweep_dir()
    if d not in _SWEEP_CACHE:
        import estimator_sweep as es
        session, chans, pair, timeline, t_samples = es.load(d)
        offset, r = es.find_offset(chans, pair, session, timeline, t_samples)
        if not np.isfinite(r) or r < 0.9:
            raise RuntimeError(f"sweep reproduction r = {r:.3f} < 0.9 on {os.path.basename(d)}")
        rows = {}
        for name, fn, kw in es.ESTIMATORS:
            t, y = fn(chans, pair, t_samples=t_samples, **kw)
            sc = es.score(t, y, timeline, offset)
            sc["info"] = (sc["d"] * np.sqrt(sc["n_eff_per_min"])
                          if np.isfinite(sc["d"]) else float("nan"))
            rows[name] = sc
        _SWEEP_CACHE[d] = dict(rows=rows, r=r, offset=offset, t_samples=t_samples,
                               names=[n for n, _, _ in es.ESTIMATORS])
    return _SWEEP_CACHE[d]


def _floor_info_ratio(d: str) -> tuple[float, str]:
    import estimator_sweep as es

    c = _sweep(d)
    base = c["rows"][c["names"][0]]
    base_info = base["d"] * np.sqrt(base["n_eff_per_min"])
    session, chans, pair, timeline, t_samples = es.load(d)
    t, y = es.est_streaming(chans, pair, order=2, tau_s=0.10, t_samples=t_samples)
    sc = es.score(t, y, timeline, c["offset"])
    info = sc["d"] * np.sqrt(sc["n_eff_per_min"])
    return float(info / base_info), (f"streaming o2 tau=0.1: d {sc['d']:.2f}, "
                                     f"ind/min {sc['n_eff_per_min']:.1f}")


def _eye_closure_ratio(d: str, channel: str) -> tuple[float, str]:
    """The signal_quality eye-closure prominence ratio for one channel."""
    from eeg_features import MUSE_CHANNELS
    from session_logger import load_raw, load_session
    from signal_quality import eyes_closed_ratio, labelled_blocks

    raw = load_raw(d)
    ts = raw[:, 0].astype(float)
    ts = ts - ts[0]
    op, cl, ratio = eyes_closed_ratio(ts, raw[:, 1 + MUSE_CHANNELS.index(channel)].astype(float),
                                      labelled_blocks(load_session(d)))
    return float(ratio), f"prominence open {op:.2f}, closed {cl:.2f}"


# -------------------------------------------- the 2026-09-17 AF7/AF8 recording
#
# Described here rather than substituted into the manuscript's claims. Asserted so that
# the numbers the handoff quotes for that session are regenerated, not remembered.


def claim_second_alpha_ratio() -> tuple[float, str]:
    """Logged-window eyes-closed alpha ratio on AF7/AF8, 2026-09-17."""
    return _logged_alpha_ratio(_alpha_dir(SECOND_ALPHA))


def claim_second_alpha_d() -> tuple[float, str]:
    """
    Cohen's d for the 9/17 AF7/AF8 eyes-closed rise, on the SAME windows as the ratio.

    This exists because the manuscript quoted a ratio from one analysis and a d and p
    from another, and read as a single result. The ratio (1.34x) came from every
    labelled window with a finite alpha; the d (0.61) and p (5e-7) came from the
    in-block subset, which drops the protocol's declared 3 s settle after each
    instruction and so is a slightly stronger, slightly different measurement (1.41x,
    d 0.61). Both analyses are defensible. Quoting half of each is not, and nothing
    would have caught it, because no claim asserted the effect size at all.
    """
    return _alpha_effect_size(_alpha_dir(SECOND_ALPHA))


def claim_second_rejection_rate() -> tuple[float, str]:
    """
    Fraction of 9/17 windows rejected, as recorded.

    Rejection is recorded as a non-finite alpha, not as a boolean field. Looking for a
    'rejected' key returns 0% on a session that actually rejected 6.8%, which is how a
    wrong number reached a draft table.
    """
    from session_logger import load_session

    session = load_session(_alpha_dir(SECOND_ALPHA))
    windows = [w for w in session["windows"]
               if w.get("phase") in ("eyes_open", "eyes_closed")]
    bad = [w for w in windows
           if not (isinstance(w.get("alpha"), (int, float)) and np.isfinite(w["alpha"]))]
    return len(bad) / len(windows), f"{len(bad)} of {len(windows)} labelled windows"


def claim_second_af7_eye_closure() -> tuple[float, str]:
    """AF7 prominence ratio, 2026-09-17. Below 1.2 = the eye-closure check fails."""
    return _eye_closure_ratio(_alpha_dir(SECOND_ALPHA), "AF7")


def claim_second_af8_eye_closure() -> tuple[float, str]:
    """AF8 prominence ratio, 2026-09-17. Below 1.2 = the eye-closure check fails."""
    return _eye_closure_ratio(_alpha_dir(SECOND_ALPHA), "AF8")


def claim_second_deployed_d() -> tuple[float, str]:
    """Deployed estimator's discriminability on AF7/AF8, 2026-09-17 (1.99 on TP9/TP10)."""
    c = _sweep(_alpha_dir(SECOND_ALPHA))
    row = c["rows"][c["names"][0]]
    return float(row["d"]), f"r = {c['r']:.3f}, offset {c['offset']:+.2f} s"


def claim_second_floor_info() -> tuple[float, str]:
    """Floor info rate over deployed on AF7/AF8, 2026-09-17 (1.83x on TP9/TP10)."""
    return _floor_info_ratio(_alpha_dir(SECOND_ALPHA))


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


def claim_info_rate_at_the_floor() -> tuple[float, str]:
    """
    Information per minute at the architecture's latency floor, over the deployed value.

    The number that turns the floor from a trade-off into a free lunch. Per-sample d
    collapses there - 1.99 deployed against about 0.6 - but independence rises more than
    thirtyfold, and precision goes with the product. Above 1.0 means the fastest
    configuration this architecture can reach is ALSO more informative than the one being
    shipped.

    Asserted because section 3 now leads with it, and because it is the kind of ratio
    that would move quietly if the band edges or the filter order changed.
    """
    return _floor_info_ratio(_sweep_dir())


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
    # Pinned too: the 38% power figure and the n = 25 anchor are in the FROZEN plan and
    # were computed from this recording's autocorrelation. A different pilot would give
    # different numbers and silently contradict the pre-registration.
    from power_analysis import session_stats
    return session_stats(os.path.join(_ROOT, "sessions", PINNED_PILOT))


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
        d = os.path.join(_ROOT, "sessions", PINNED_PILOT)
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
    "prompt changes, deployed ctrl":      (claim_chatter_after,           33,     3),
    "library clipping bound":             (claim_library_clipping_bound,  0.980,  0.005),
    "renders on the dominant prompt":     (claim_library_dominant_variants, 32,   0),
    "end-to-end budget, library":         (claim_latency_budget,          6.5,    0.05),
    "coupling recovers a +6 s lag":       (claim_coupling_recovers_lag,   6.0,    1.0),
    "eyes-closed alpha ratio":            (claim_alpha_validation_ratio,  2.13,   0.02),
    "same effect on AF7/AF8":             (claim_channel_mismatch_af,     0.91,   0.03),
    "DEAP arousal rho (AF3/AF4)":         (claim_deap_arousal_rho,        0.303,  0.02),
    "streaming estimator budget":         (claim_streaming_latency_budget, 0.413, 0.01),
    "replay fidelity vs the log":         (claim_replay_fidelity,         0.991,  0.005),
    "retuned, no dwell: inside a xfade":  (claim_retuned_chatter_no_dwell, 591,   8),
    "retuned, 1 s dwell: inside a xfade": (claim_retuned_chatter_with_dwell, 0,   0),
    "ladder margin 0.25 still responds":  (claim_ladder_margin_responds,  109,    4),
    "trend noise / genuine drift":        (claim_trend_noise_to_signal,   1.77,   0.05),
    "distinct prompts build_prompt emits":(claim_reachable_prompt_space,  9,      0),
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
    "analysis-path latency floor":        (claim_analysis_latency_floor,    0.189,  0.005),
    "info rate at the floor / deployed":  (claim_info_rate_at_the_floor,    1.83,   0.05),
    "9/17 AF7/AF8 alpha ratio (logged)":  (claim_second_alpha_ratio,        1.34,   0.02),
    "9/17 AF7/AF8 alpha d":               (claim_second_alpha_d,            0.53,   0.02),
    "9/17 rejection rate":                (claim_second_rejection_rate,     0.068,  0.005),
    "9/17 AF7 eye-closure ratio":         (claim_second_af7_eye_closure,    1.09,   0.03),
    "9/17 AF8 eye-closure ratio":         (claim_second_af8_eye_closure,    1.13,   0.03),
    "9/17 deployed d on AF7/AF8":         (claim_second_deployed_d,         1.06,   0.03),
    "9/17 floor info / deployed":         (claim_second_floor_info,         1.94,   0.05),
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
