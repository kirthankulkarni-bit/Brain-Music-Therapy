"""
public_analyses.py - Sections 4.7, 5.1, 5.4 and 6 of the paper, on public and synthetic data.

  autocorrelation   deployed index on every unspliced PhysioNet EEGBCI baseline run
                    (AF7/AF8), and on the 40 DEAP trials of participant 1 (AF3/AF4)
  power             the simulation of scripts/power_analysis.py, driven by the
                    PhysioNet autocorrelation instead of a pilot recording
  replay            the real controller (music_engine.PromptGovernor) driven by z
                    series reconstructed from PhysioNet and synthetic sessions
  alpha             eyes-closed alpha increase and peak prominence per PhysioNet
                    subject, on AF7/AF8 and on T9/T10 (the temporal stand-ins)

Everything runs the deployed FeatureExtractor; nothing is reimplemented except the
prominence measure, copied from scripts/signal_quality.py with the sampling rate made a
parameter (that script hardcodes the Muse's 256 Hz).

Usage:
    python scripts/public_analyses.py --out results/public/analyses.json
    python scripts/public_analyses.py --only alpha
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import signal as sps
from scipy import stats as spstats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402
from public_data import (EC, EEGBCI_CHANNELS, EEGBCI_FS, EEGBCI_SUBJECTS, EO,  # noqa: E402
                         eegbci_runs, eegbci_session, synthetic_session)

DEPLOYED = (4.0, 1.0, 3.0)
RETUNED = (2.0, 0.5, 0.5)
SETTLE_HOPS = 9                # 3 tau at the deployed 1 s hop: drop the smoother's start
SESSION_MIN = 20.0             # session length the paper's design discussion assumes
CROSSFADE_S = 1.0
_JITTER_S = 0.05

_RUNS: dict = {}


def runs(subject):
    if subject not in _RUNS:
        _RUNS[subject] = eegbci_runs(subject)
    return _RUNS[subject]


def subjects(limit=None):
    out = []
    for s in EEGBCI_SUBJECTS[:limit]:
        try:
            runs(s)
            out.append(s)
        except Exception as exc:                            # pragma: no cover
            print(f"  [skip] S{s:03d}: {exc}")
    return out


def windows(chans, fs, channels, pair, window_s, hop_s, field="log_beta_alpha"):
    """(t, values, rejected fraction) from the deployed extractor, unsmoothed."""
    cfg = FeatureConfig(sampling_rate=fs, window_seconds=window_s, hop_seconds=hop_s,
                        channels=channels, frontal_channels=pair)
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    t, v, tot = [], [], 0
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        tot += 1
        f = ex.extract(chans[:, s0:s0 + nw])
        x = getattr(f, field)
        if f.valid and np.isfinite(x) and (field != "alpha" or x > 0):
            t.append((s0 + nw) / fs)
            v.append(x)
    return np.asarray(t), np.asarray(v, dtype=float), 1 - len(v) / max(tot, 1)


def smooth(v, hop_s, tau_s):
    a = 1.0 - np.exp(-hop_s / tau_s)
    out = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc += a * (x - acc)
        out[i] = acc
    return out


def acf(x, max_lag):
    x = x - x.mean()
    den = float(np.dot(x, x))
    return np.array([1.0] + [float(np.dot(x[:-k], x[k:])) / den
                             for k in range(1, max_lag + 1)])


# ------------------------------------------------------------ 5.1 autocorrelation


def autocorrelation(subs, max_lag=20):
    """Deployed index, smoothed, per unspliced run. Runs with gaps are skipped."""
    rhos, curves, sds = [], [], []
    for s in subs:
        for run in runs(s):
            t, v, _ = windows(run, EEGBCI_FS, EEGBCI_CHANNELS, ("AF7", "AF8"),
                              DEPLOYED[0], DEPLOYED[1])
            if v.size < 40 or np.any(np.diff(t) > DEPLOYED[1] + 1e-6):
                continue                                    # a rejected window = a gap
            sm = smooth(v, DEPLOYED[1], DEPLOYED[2])[SETTLE_HOPS:]
            rhos.append(float(np.corrcoef(sm[:-1], sm[1:])[0, 1]))
            curves.append(acf(sm, max_lag))
            sds.append(float(sm.std(ddof=1) / v.std(ddof=1)))
    rho = float(np.median(rhos))
    mean_acf = np.mean(curves, axis=0)
    below = np.flatnonzero(mean_acf < np.exp(-1))
    n = int(SESSION_MIN * 60 / DEPLOYED[1])
    n_eff = n * (1 - rho) / (1 + rho)
    return {
        "n_runs": len(rhos), "rho_median": rho,
        "rho_q1": float(np.percentile(rhos, 25)), "rho_q3": float(np.percentile(rhos, 75)),
        "overstatement": float(np.sqrt(n / n_eff)),
        "session_windows": n, "session_n_eff": float(n_eff),
        "decorrelation_s": float(below[0] * DEPLOYED[1]) if below.size else None,
        "acf_mean": mean_acf.tolist(),
        "acf_q1": np.percentile(curves, 25, axis=0).tolist(),
        "acf_q3": np.percentile(curves, 75, axis=0).tolist(),
        "z_sd_median": float(np.median(sds)),
    }


def autocorrelation_deap():
    """DEAP participant 1, 40 one-minute music-video trials, AF3/AF4."""
    import pickle

    from validate_index_deap import DEAP_CHANNELS, DEAP_FS, DEAP_PRETRIAL_S

    path = os.path.join(_ROOT, "s01.dat")
    if not os.path.exists(path):
        return {"missing": True}
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"])[:, :32, int(DEAP_PRETRIAL_S * DEAP_FS):]
    chans = tuple(DEAP_CHANNELS)
    rhos = []
    for trial in data:
        t, v, _ = windows(trial, DEAP_FS, chans, ("AF3", "AF4"), DEPLOYED[0], DEPLOYED[1])
        if v.size < 40 or np.any(np.diff(t) > DEPLOYED[1] + 1e-6):
            continue
        sm = smooth(v, DEPLOYED[1], DEPLOYED[2])[SETTLE_HOPS:]
        rhos.append(float(np.corrcoef(sm[:-1], sm[1:])[0, 1]))
    return {"n_trials": len(rhos), "rho_median": float(np.median(rhos)),
            "rho_q1": float(np.percentile(rhos, 25)),
            "rho_q3": float(np.percentile(rhos, 75))}


# ---------------------------------------------------------------- 5.4 power


def power(ac, sims=2000):
    from power_analysis import power_for, required_n

    """
    The cost of dichotomising, under the public autocorrelation.

    Participant counts are deliberately NOT derived here. Within-session variance comes
    from one-minute resting runs, which cannot contain the slow drift of a 20-minute
    session, so any n computed from it would be optimistic. What survives that caveat is
    the RELATIVE cost of the two outcomes under the same structure: the smallest effect
    each detects with 80% power at n = 10, paired, on a 0.01 z grid.
    """
    from power_analysis import power_for

    st = {"rho": ac["rho_median"], "sd": ac["z_sd_median"],
          "n_windows": ac["session_windows"], "target_z": -1.0}
    tib_sims = max(200, sims // 4)

    def effect_for_80(outcome, n_sims):
        for eff in np.arange(0.01, 1.501, 0.01):
            if power_for(np.random.default_rng(0), 10, float(eff), 0.5, st, True,
                         outcome, n_sims) >= 0.80:
                return float(round(eff, 2))
        return None

    zm = effect_for_80("z_mean", sims)
    tib = effect_for_80("time_in_band", tib_sims)
    return {"stats": st, "n_per_arm": 10, "z_mean_effect_80": zm,
            "time_in_band_effect_80": tib,
            "dichotomisation_factor": (tib / zm) if zm and tib else None,
            "sims": sims, "sims_time_in_band": tib_sims}


# ------------------------------------------------------------ 4.7 replay


def replay(z, t, dwell):
    from music_engine import PromptGovernor

    gov = PromptGovernor(target_z=-1.0, ladder_margin=0.0, min_dwell_seconds=dwell)
    times, prompts, prev = [], [], None
    for value, now in zip(z, t):
        p = gov.update(float(value), now=float(now))
        if prev is not None and p != prev:
            times.append(float(now))
            prompts.append(p)
        prev = p
    gaps = np.diff(times) if len(times) > 1 else np.array([])
    return {"changes": len(times), "distinct": len(set(prompts)),
            "under_crossfade": int((gaps < CROSSFADE_S - _JITTER_S).sum())}


def z_for(sess, baseline_chans, config):
    """z of the smoothed index against an eyes-open baseline, as BaselineNormalizer does."""
    _, b, _ = windows(baseline_chans, sess["fs"], sess["channels"], ("AF7", "AF8"),
                      DEPLOYED[0], DEPLOYED[1])
    if b.size < 30:
        return None, None
    t, v, _ = windows(sess["chans"], sess["fs"], sess["channels"], ("AF7", "AF8"),
                      config[0], config[1])
    if v.size < 20:
        return None, None
    return (smooth(v, config[1], config[2]) - b.mean()) / b.std(ddof=1), t


def controller_replay(subs, seeds=50, ratio=1.34):
    out = {}
    sources = []
    for s in subs:
        eo, _ = runs(s)
        sources.append(("eegbci", eegbci_session(s), eo))
    for k in range(seeds):
        sess = synthetic_session(k, ratio)
        sources.append(("synthetic", sess, sess["chans"][:, :int(60 * sess["fs"])]))
    for label, sess, base in sources:
        minutes = sess["chans"].shape[1] / sess["fs"] / 60.0
        for cname, cfg in (("deployed", DEPLOYED), ("retuned", RETUNED)):
            z, t = z_for(sess, base, cfg)
            if z is None:
                continue
            for dwell in (0.0, CROSSFADE_S):
                m = replay(z, t, dwell)
                key = f"{label}|{cname}|dwell{dwell:g}"
                acc = out.setdefault(key, {"sessions": 0, "minutes": 0.0, "changes": 0,
                                           "under_crossfade": 0,
                                           "sessions_with_any": 0})
                acc["sessions"] += 1
                acc["minutes"] += minutes
                acc["changes"] += m["changes"]
                acc["under_crossfade"] += m["under_crossfade"]
                acc["sessions_with_any"] += int(m["under_crossfade"] > 0)
    for acc in out.values():
        acc["under_crossfade_per_min"] = acc["under_crossfade"] / acc["minutes"]
        acc["changes_per_min"] = acc["changes"] / acc["minutes"]
    return out


# ------------------------------------------------------------ 6 alpha validation


def prominence(x, fs):
    from scipy.ndimage import median_filter

    x = x - np.mean(x)
    f, p = sps.welch(x, fs=fs, nperseg=int(fs * 4), noverlap=int(fs * 2))
    ok = (f > 2) & (f < 45) & (p > 0)
    f, p = f[ok], p[ok]
    if f.size < 30:
        return float("nan")
    df = float(np.median(np.diff(f)))
    width = max(5, int(round(8.0 / df)) | 1)
    base = 10 ** median_filter(np.log10(p), size=width, mode="nearest")
    band = (f >= 8.0) & (f <= 13.0)
    return float(np.mean(p[band]) / np.mean(base[band]))


def run_prominence(run, ch, fs, seg_s=10.0, trim_s=2.0):
    x = run[EEGBCI_CHANNELS.index(ch), int(trim_s * fs):]
    step = int(seg_s * fs)
    v = [prominence(x[s0:s0 + step], fs) for s0 in range(0, x.size - step + 1, step)]
    v = [q for q in v if np.isfinite(q)]
    return float(np.median(v)) if v else float("nan")


def alpha_validation(subs):
    res = {}
    for label, pair in (("AF7/AF8", ("AF7", "AF8")), ("T9/T10", ("T9", "T10"))):
        log_ratio, d_sub, rej, prom = [], [], [], {c: [] for c in pair}
        group_t = {EO: [], EC: []}
        for s in subs:
            eo, ec = runs(s)
            to, ao, ro = windows(eo, EEGBCI_FS, EEGBCI_CHANNELS, pair, 4.0, 1.0, "alpha")
            tc, ac, rc = windows(ec, EEGBCI_FS, EEGBCI_CHANNELS, pair, 4.0, 1.0, "alpha")
            rej.append((ro + rc) / 2)
            if ao.size < 10 or ac.size < 10:
                continue
            lo, lc = np.log10(ao), np.log10(ac)
            log_ratio.append(float(lc.mean() - lo.mean()))
            pooled = np.sqrt((lo.var(ddof=1) + lc.var(ddof=1)) / 2)
            d_sub.append(float((lc.mean() - lo.mean()) / pooled))
            for c in pair:
                po, pc = run_prominence(eo, c, EEGBCI_FS), run_prominence(ec, c, EEGBCI_FS)
                if np.isfinite(po) and np.isfinite(pc) and po > 0:
                    prom[c].append(pc / po)
            # Group trace: alpha on a common 1 s grid, normalised to the subject's
            # eyes-open mean, for Figure 3.
            grid = np.arange(4.0, 61.0, 1.0)
            norm = 10 ** lo.mean()
            group_t[EO].append(np.interp(grid, to, ao / norm, left=np.nan, right=np.nan))
            group_t[EC].append(np.interp(grid, tc, ac / norm, left=np.nan, right=np.nan))
        lr = np.asarray(log_ratio)
        w = spstats.wilcoxon(lr)
        dz = float(lr.mean() / lr.std(ddof=1))
        res[label] = {
            "n_subjects": int(lr.size),
            "ratio_median": float(10 ** np.median(lr)),
            "ratio_geomean": float(10 ** lr.mean()),
            "fraction_increase": float((lr > 0).mean()),
            "wilcoxon_p": float(w.pvalue), "dz": dz,
            "within_subject_d_median": float(np.median(d_sub)),
            "rejection_median": float(np.median(rej)),
            "rejection_mean": float(np.mean(rej)),
            "prominence_ratio_median": {c: float(np.median(v)) for c, v in prom.items()},
            "prominence_pass_fraction": {c: float(np.mean(np.asarray(v) >= 1.2))
                                         for c, v in prom.items()},
            "trace_grid": list(np.arange(4.0, 61.0, 1.0)),
            "trace_eo": np.nanmedian(group_t[EO], axis=0).tolist(),
            "trace_ec": np.nanmedian(group_t[EC], axis=0).tolist(),
            "trace_eo_q": np.nanpercentile(group_t[EO], [25, 75], axis=0).tolist(),
            "trace_ec_q": np.nanpercentile(group_t[EC], [25, 75], axis=0).tolist(),
        }
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_ROOT, "results", "public", "analyses.json"))
    p.add_argument("--only", choices=["autocorrelation", "power", "replay", "alpha"])
    p.add_argument("--limit", type=int, default=None, help="first N subjects only")
    p.add_argument("--sims", type=int, default=2000)
    args = p.parse_args()

    subs = subjects(args.limit)
    print(f"  {len(subs)} EEGBCI subjects")
    res = {}
    if os.path.exists(args.out) and args.only:
        res = json.load(open(args.out, encoding="utf-8"))
    res["n_subjects_loaded"] = len(subs)
    if args.only in (None, "autocorrelation", "power"):
        res["autocorrelation"] = autocorrelation(subs)
        res["autocorrelation_deap"] = autocorrelation_deap()
        a = res["autocorrelation"]
        print(f"  autocorr: rho {a['rho_median']:.3f} over {a['n_runs']} runs, "
              f"overstatement {a['overstatement']:.2f}x, decorrelates {a['decorrelation_s']} s,"
              f" DEAP rho {res['autocorrelation_deap'].get('rho_median', float('nan')):.3f}")
    if args.only in (None, "power"):
        res["power"] = power(res["autocorrelation"], args.sims)
        pw = res["power"]
        print(f"  power: 80% at n=10 needs z_mean {pw['z_mean_effect_80']}, time_in_band "
              f"{pw['time_in_band_effect_80']} -> x{pw['dichotomisation_factor']}")
    if args.only in (None, "replay"):
        res["replay"] = controller_replay(subs)
        for k, v in res["replay"].items():
            print(f"  replay {k:<32} changes {v['changes']:>6}  under xfade "
                  f"{v['under_crossfade']:>5}  ({v['under_crossfade_per_min']:.2f}/min)")
    if args.only in (None, "alpha"):
        res["alpha"] = alpha_validation(subs)
        for k, v in res["alpha"].items():
            print(f"  alpha {k}: median ratio {v['ratio_median']:.2f}x, "
                  f"{v['fraction_increase']:.0%} increase, p {v['wilcoxon_p']:.1e}, "
                  f"dz {v['dz']:.2f}, rej {v['rejection_median']:.1%}, "
                  f"prominence {v['prominence_ratio_median']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
