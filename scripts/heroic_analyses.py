"""
heroic_analyses.py - the signal checks of Sections 5.1 and 6, repeated on Muse 2 data.

The PhysioNet results in the paper come from a 64-channel gel cap. HEROIC (Zenodo
10.5281/zenodo.11493897, CC BY 4.0; loader and inclusion rules in public_data.py) is the
only public source here recorded on the system's own hardware, so this script asks the
two questions a gel cap cannot answer:

  alpha            does alpha rise on eye closure at the Muse's AF7/AF8 - the channels the
                   arousal index is computed from - and at TP9/TP10? Per session, then
                   per subject (median over that subject's sessions), since subjects
                   contribute one to five sessions each. Inference is over subjects.
  autocorrelation  lag-1 autocorrelation of the smoothed deployed index within each
                   61 s eye block that lost no window to rejection.

Estimator latency on the same data is in estimator_sweep_public.py --source heroic.

Usage:
    python scripts/heroic_analyses.py --out results/public/heroic_analyses.json
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

from public_analyses import DEPLOYED, SESSION_MIN, SETTLE_HOPS, prominence, smooth, windows  # noqa: E402
from public_data import EC, EO, MUSE_NAMES, sessions  # noqa: E402

PAIRS = {"AF7/AF8": ("AF7", "AF8"), "TP9/TP10": ("TP9", "TP10")}
GUARD_S = 4.0            # window centres this close to a transition belong to neither state
SEG_S = 10.0             # prominence segment length, as in public_analyses
TRACE = np.arange(-30.0, 41.0, 1.0)


def blocks(timeline):
    """[(start, end, phase)] for the eye blocks."""
    return [(timeline[i][0], timeline[i + 1][0], timeline[i][1])
            for i in range(len(timeline) - 1) if timeline[i][1] in (EO, EC)]


def label(t_centre, bl):
    out = np.full(t_centre.size, "", dtype=object)
    for a, b, ph in bl:
        out[(t_centre >= a + GUARD_S) & (t_centre < b - GUARD_S)] = ph
    return out


def block_prominence(sess, ch, bl, phase):
    x = sess["chans"][MUSE_NAMES.index(ch)]
    fs = sess["fs"]
    v = []
    for a, b, ph in bl:
        if ph != phase:
            continue
        lo, hi = int((a + GUARD_S) * fs), int((b - GUARD_S) * fs)
        step = int(SEG_S * fs)
        v += [prominence(x[s0:s0 + step], fs) for s0 in range(lo, hi - step + 1, step)]
    v = [q for q in v if np.isfinite(q)]
    return float(np.median(v)) if v else float("nan")


def run(all_sessions):
    out = {"alpha": {}, "autocorrelation": {}}
    for key, pair in PAIRS.items():
        per_session, by_subject, rej, prom = [], {}, [], {c: {} for c in pair}
        traces = []
        for s in all_sessions:
            bl = blocks(s["timeline"])
            t, a, r = windows(s["chans"], s["fs"], s["channels"], pair, 4.0, 1.0, "alpha")
            rej.append(r)
            lab = label(t - 2.0, bl)                       # window centre
            lc, lo = np.log10(a[lab == EC]), np.log10(a[lab == EO])
            if lc.size < 10 or lo.size < 10:
                continue
            lr = float(lc.mean() - lo.mean())
            per_session.append({"session": s["name"], "subject": s["subject"], "log_ratio": lr})
            by_subject.setdefault(s["subject"], []).append(lr)
            for c in pair:
                pc = block_prominence(s, c, bl, EC)
                po = block_prominence(s, c, bl, EO)
                if np.isfinite(pc) and np.isfinite(po) and po > 0:
                    prom[c].setdefault(s["subject"], []).append(pc / po)
            # Event-locked alpha around the first eyes-closed -> eyes-open cue,
            # normalised to the eyes-closed mean of that session.
            tr = next(b[0] for b in bl if b[2] == EO)
            norm = 10 ** lc.mean()
            traces.append(np.interp(tr + TRACE, t - 2.0, a / norm, left=np.nan, right=np.nan))
        lr_sess = np.array([p["log_ratio"] for p in per_session])
        lr_subj = np.array([np.median(v) for v in by_subject.values()])
        w = spstats.wilcoxon(lr_subj)
        out["alpha"][key] = {
            "n_sessions": int(lr_sess.size), "n_subjects": int(lr_subj.size),
            "ratio_median_sessions": float(10 ** np.median(lr_sess)),
            "fraction_increase_sessions": float((lr_sess > 0).mean()),
            "ratio_median_subjects": float(10 ** np.median(lr_subj)),
            "subjects_increase": int((lr_subj > 0).sum()),
            "wilcoxon_p_subjects": float(w.pvalue),
            "dz_subjects": float(lr_subj.mean() / lr_subj.std(ddof=1)),
            "rejection_median": float(np.median(rej)),
            "prominence_ratio_median_subjects": {
                c: float(np.median([np.median(v) for v in d.values()])) for c, d in prom.items()},
            "per_subject_ratio": {k: float(10 ** np.median(v)) for k, v in sorted(by_subject.items())},
            "trace_rel_s": TRACE.tolist(),
            "trace_median": np.nanmedian(traces, axis=0).tolist(),
            "trace_q": np.nanpercentile(traces, [25, 75], axis=0).tolist(),
            "per_session": per_session,
        }

    # 5.1 on Muse data: deployed index (log beta/alpha, AF7/AF8), smoothed, within each
    # clean eye block. A block that lost any window is a gap and is skipped.
    rhos = []
    for s in all_sessions:
        for a0, b0, _ in blocks(s["timeline"]):
            lo, hi = int(a0 * s["fs"]), int(b0 * s["fs"])
            t, v, _ = windows(s["chans"][:, lo:hi], s["fs"], s["channels"], ("AF7", "AF8"),
                              DEPLOYED[0], DEPLOYED[1])
            if v.size < 40 or np.any(np.diff(t) > DEPLOYED[1] + 1e-6):
                continue
            sm = smooth(v, DEPLOYED[1], DEPLOYED[2])[SETTLE_HOPS:]
            rhos.append(float(np.corrcoef(sm[:-1], sm[1:])[0, 1]))
    rho = float(np.median(rhos))
    n = int(SESSION_MIN * 60 / DEPLOYED[1])
    n_eff = n * (1 - rho) / (1 + rho)
    out["autocorrelation"] = {"n_blocks": len(rhos), "rho_median": rho,
                              "session_n_eff": float(n_eff),
                              "overstatement": float(np.sqrt(n / n_eff))}
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_ROOT, "results", "public",
                                                 "heroic_analyses.json"))
    args = p.parse_args()
    all_sessions = list(sessions("heroic"))
    print(f"  {len(all_sessions)} HEROIC sessions, "
          f"{len({s['subject'] for s in all_sessions})} subjects")
    res = run(all_sessions)
    res["n_sessions_loaded"] = len(all_sessions)
    for k, v in res["alpha"].items():
        print(f"  alpha {k}: sessions {v['ratio_median_sessions']:.2f}x "
              f"({v['fraction_increase_sessions']:.0%} rise); subjects "
              f"{v['ratio_median_subjects']:.2f}x, {v['subjects_increase']}/{v['n_subjects']} "
              f"rise, Wilcoxon p {v['wilcoxon_p_subjects']:.3g}, dz {v['dz_subjects']:.2f}, "
              f"rej {v['rejection_median']:.1%}, prominence "
              f"{ {c: round(x, 2) for c, x in v['prominence_ratio_median_subjects'].items()} }")
    a = res["autocorrelation"]
    print(f"  autocorr AF7/AF8: rho {a['rho_median']:.3f} over {a['n_blocks']} blocks, "
          f"N_eff {a['session_n_eff']:.1f}, overstatement {a['overstatement']:.2f}x")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
