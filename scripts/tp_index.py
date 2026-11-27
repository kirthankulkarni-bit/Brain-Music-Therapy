"""
tp_index.py - would the arousal index be better computed from TP9/TP10?

Section 6 finds almost no eye-state alpha at the Muse's AF7/AF8, the channels the index
is computed from, and a clear effect at TP9/TP10. Moving the index looks like the obvious
fix. This script asks what the move would change, using only public data, and adds the
check that matters most for temporal sites: TP9/TP10 sit over the temporalis muscle, and
muscle activity is broadband, so it raises beta power and with it log(beta/alpha). An
index that followed jaw tension would correlate with arousal for the wrong reason.

For each pair (AF7/AF8 and TP9/TP10, or the dataset's nearest sites) and the deployed
extractor (W = 4 s, H = 1 s):

  rejection     fraction of windows the 350 uV rule rejects
  contrast      eye-state separation of the unsmoothed index: within-session difference
                of means divided by the pooled within-state SD (HEROIC, guarded blocks)
  autocorr      lag-1 autocorrelation of the smoothed index (tau = 3 s), as in Section 5.1
  emg_r         within-state Spearman correlation, across windows, between the index and
                log gamma power (30-45 Hz), the extractor's own muscle-sensitive band.
                Computed after removing each state's mean, so the eye-state effect on
                both quantities cannot create it.

DEAP participant 1: per trial, the index and log gamma power on AF3/AF4 and on
T7/P7/T8/P8; Spearman of each with arousal, and the index-arousal correlation after
partialling out gamma (ranks residualised on ranks).

Usage:
    python scripts/tp_index.py --out results/public/tp_index.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats as spstats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eeg_features import FeatureConfig, FeatureExtractor  # noqa: E402
from heroic_analyses import GUARD_S, blocks, label  # noqa: E402
from public_analyses import DEPLOYED, SESSION_MIN, SETTLE_HOPS, runs, smooth, subjects  # noqa: E402
from public_data import EC, EEGBCI_CHANNELS, EEGBCI_FS, EO, sessions  # noqa: E402

W, H, TAU = DEPLOYED


def features(chans, fs, channels, pair, notch=60.0, bands=False):
    """(t_end, log beta/alpha, log10 gamma, rejected fraction), deployed extractor.
    With bands=True, also an (n, 2) array of log10 alpha and log10 beta."""
    cfg = FeatureConfig(sampling_rate=fs, window_seconds=W, hop_seconds=H,
                        channels=channels, frontal_channels=pair, notch_hz=notch)
    ex = FeatureExtractor(cfg)
    nw, nh = cfg.window_samples, cfg.hop_samples
    t, ix, g, ab, tot = [], [], [], [], 0
    for s0 in range(0, chans.shape[1] - nw + 1, nh):
        tot += 1
        f = ex.extract(chans[:, s0:s0 + nw])
        if f.valid and np.isfinite(f.log_beta_alpha) and f.gamma > 0:
            t.append((s0 + nw) / fs)
            ix.append(f.log_beta_alpha)
            g.append(np.log10(f.gamma))
            ab.append((np.log10(f.alpha), np.log10(f.beta)))
    out = (np.asarray(t), np.asarray(ix), np.asarray(g), 1 - len(ix) / max(tot, 1))
    return out + (np.asarray(ab),) if bands else out


def lag1(t, v):
    """Lag-1 autocorrelation of the smoothed index; None if the series has a gap."""
    if v.size < 40 or np.any(np.diff(t) > H + 1e-6):
        return None
    sm = smooth(v, H, TAU)[SETTLE_HOPS:]
    return float(np.corrcoef(sm[:-1], sm[1:])[0, 1])


def emg_r(ix, g, groups):
    """Spearman(index, log gamma) after removing each group's mean from both."""
    ix, g = ix.copy(), g.copy()
    for k in set(groups):
        m = groups == k
        ix[m] -= ix[m].mean()
        g[m] -= g[m].mean()
    return float(spstats.spearmanr(ix, g)[0])


def n_eff_summary(rhos):
    rho = float(np.median(rhos))
    n = int(SESSION_MIN * 60 / H)
    ne = n * (1 - rho) / (1 + rho)
    return {"n": len(rhos), "rho_median": rho, "session_n_eff": float(ne),
            "overstatement": float(np.sqrt(n / ne))}


def subject_median(by_subject):
    return float(np.median([np.median(v) for v in by_subject.values()]))


# --------------------------------------------------------------------- HEROIC (Muse 2)


def heroic(all_sessions):
    out = {}
    for key, pair in (("AF7/AF8", ("AF7", "AF8")), ("TP9/TP10", ("TP9", "TP10"))):
        rej, con, emg, band, rhos = {}, {}, {}, {}, []
        for s in all_sessions:
            sub = s["subject"]
            bl = blocks(s["timeline"])
            t, ix, g, r, ab = features(s["chans"], s["fs"], s["channels"], pair, bands=True)
            rej.setdefault(sub, []).append(r)
            lab = label(t - W / 2, bl)
            keep = lab != ""
            if (lab == EC).sum() >= 10 and (lab == EO).sum() >= 10:
                a, b = ix[lab == EO], ix[lab == EC]
                sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
                con.setdefault(sub, []).append(float((a.mean() - b.mean()) / sd))
                emg.setdefault(sub, []).append(emg_r(ix[keep], g[keep], lab[keep]))
                d = np.r_[ab[lab == EO].mean(0) - ab[lab == EC].mean(0),
                          g[lab == EO].mean() - g[lab == EC].mean()]
                band.setdefault(sub, []).append(d)
            # Autocorrelation exactly as heroic_analyses.py: each eye block on its own.
            for a0, b0, _ in bl:
                lo, hi = int(a0 * s["fs"]), int(b0 * s["fs"])
                tb, vb, _, _ = features(s["chans"][:, lo:hi], s["fs"], s["channels"], pair)
                rr = lag1(tb, vb)
                if rr is not None:
                    rhos.append(rr)
        out[key] = {
            "n_subjects": len(con),
            "rejection_subject_median": subject_median(rej),
            "contrast_subject_median": subject_median(con),
            "contrast_subjects_positive": int(sum(np.median(v) > 0 for v in con.values())),
            "emg_r_subject_median": subject_median(emg),
            "emg_r_per_subject": {k: float(np.median(v)) for k, v in sorted(emg.items())},
            # Eyes-open minus eyes-closed mean, log10 units, median over subjects of the
            # median over that subject's sessions. The index contrast is beta - alpha.
            "eo_minus_ec_log10": dict(zip(("alpha", "beta", "gamma"), np.median(
                [np.median(v, axis=0) for v in band.values()], axis=0).tolist())),
            "autocorrelation": n_eff_summary(rhos),
        }
    return out


# ---------------------------------------------------- PhysioNet EEGBCI resting runs


def eegbci(subs):
    out = {}
    for key, pair in (("AF7/AF8", ("AF7", "AF8")), ("T9/T10", ("T9", "T10"))):
        rej, emg, rhos = [], [], []
        for s in subs:
            for run in runs(s):
                t, ix, g, r = features(run, EEGBCI_FS, EEGBCI_CHANNELS, pair)
                rej.append(r)
                if ix.size >= 20:
                    emg.append(emg_r(ix, g, np.zeros(ix.size, dtype=int)))
                rr = lag1(t, ix)
                if rr is not None:
                    rhos.append(rr)
        out[key] = {"n_runs": len(rej), "rejection_median": float(np.median(rej)),
                    "emg_r_median": float(np.median(emg)),
                    "autocorrelation": n_eff_summary(rhos)}
    return out


# ------------------------------------------------------------------- DEAP s01


def _partial_spearman(x, y, z):
    rx, ry, rz = (spstats.rankdata(v) for v in (x, y, z))
    res = [v - np.polyval(np.polyfit(rz, v, 1), rz) for v in (rx, ry)]
    r = float(np.corrcoef(*res)[0, 1])
    n = x.size
    tstat = r * np.sqrt((n - 3) / (1 - r * r))
    return r, float(2 * spstats.t.sf(abs(tstat), n - 3))


def deap(path):
    import pickle

    from validate_index_deap import DEAP_CHANNELS, DEAP_FS, DEAP_PRETRIAL_S

    if not os.path.exists(path):
        return {"missing": True}
    with open(path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"])[:, :32, :]
    arousal = np.asarray(d["labels"])[:, 1]
    start = int(DEAP_PRETRIAL_S * DEAP_FS)
    cfg_hop = 2.0                                   # as validate_index_deap.trial_index
    out = {}
    for key, chans in (("AF3/AF4", ("AF3", "AF4")), ("T7/P7/T8/P8", ("T7", "P7", "T8", "P8"))):
        cfg = FeatureConfig(sampling_rate=DEAP_FS, window_seconds=W, hop_seconds=cfg_hop,
                            channels=tuple(DEAP_CHANNELS), frontal_channels=chans,
                            notch_hz=50.0)
        ex = FeatureExtractor(cfg)
        ix, g = [], []
        for trial in data:
            vi, vg = [], []
            for s0 in range(start, trial.shape[1] - cfg.window_samples + 1, cfg.hop_samples):
                f = ex.extract(trial[:, s0:s0 + cfg.window_samples])
                if f.valid and np.isfinite(f.log_beta_alpha) and f.gamma > 0:
                    vi.append(f.log_beta_alpha)
                    vg.append(np.log10(f.gamma))
            ok = len(vi) >= 5
            ix.append(np.median(vi) if ok else np.nan)
            g.append(np.median(vg) if ok else np.nan)
        ix, g = np.asarray(ix), np.asarray(g)
        m = np.isfinite(ix) & np.isfinite(g)
        r_ix, p_ix = spstats.spearmanr(ix[m], arousal[m])
        r_g, p_g = spstats.spearmanr(g[m], arousal[m])
        r_ig = spstats.spearmanr(ix[m], g[m])[0]
        r_part, p_part = _partial_spearman(ix[m], arousal[m], g[m])
        out[key] = {"n_trials": int(m.sum()),
                    "index_arousal_rho": float(r_ix), "index_arousal_p": float(p_ix),
                    "gamma_arousal_rho": float(r_g), "gamma_arousal_p": float(p_g),
                    "index_gamma_rho": float(r_ig),
                    "index_arousal_partial_gamma": r_part, "partial_p": p_part}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_ROOT, "results", "public", "tp_index.json"))
    p.add_argument("--deap", default=os.path.join(_ROOT, "s01.dat"))
    p.add_argument("--limit", type=int, default=None, help="first N EEGBCI subjects only")
    args = p.parse_args()

    res = {"heroic": heroic(list(sessions("heroic"))),
           "eegbci": eegbci(subjects(args.limit)),
           "deap": deap(args.deap)}
    for k, v in res["heroic"].items():
        a = v["autocorrelation"]
        e = v["eo_minus_ec_log10"]
        print(f"  HEROIC {k:9s} EO-EC log10 alpha {e['alpha']:+.3f} beta {e['beta']:+.3f} "
              f"gamma {e['gamma']:+.3f}")
        print(f"  HEROIC {k:9s} rej {v['rejection_subject_median']:.1%}  contrast "
              f"{v['contrast_subject_median']:.2f} ({v['contrast_subjects_positive']}/"
              f"{v['n_subjects']} +)  emg r {v['emg_r_subject_median']:.2f}  rho "
              f"{a['rho_median']:.3f} ({a['n']} blocks)")
    for k, v in res["eegbci"].items():
        a = v["autocorrelation"]
        print(f"  EEGBCI {k:9s} rej {v['rejection_median']:.1%}  emg r {v['emg_r_median']:.2f}"
              f"  rho {a['rho_median']:.3f} ({a['n']} runs)")
    for k, v in res["deap"].items() if isinstance(res["deap"], dict) else []:
        if k == "missing":
            print("  DEAP s01.dat absent")
            continue
        print(f"  DEAP {k:12s} index~arousal {v['index_arousal_rho']:+.3f}  gamma~arousal "
              f"{v['gamma_arousal_rho']:+.3f} (p {v['gamma_arousal_p']:.3f})  index~gamma "
              f"{v['index_gamma_rho']:+.3f}  partial {v['index_arousal_partial_gamma']:+.3f} "
              f"(p {v['partial_p']:.3f})")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
