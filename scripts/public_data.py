"""
public_data.py - labelled eyes-open/eyes-closed sessions from public or synthetic sources.

Every analysis-path result in the paper is computed from sessions built here, so the
paper depends on no recording of the author.

SOURCES

  eegbci     PhysioNet EEG Motor Movement/Imagery Dataset (Schalk et al. 2004;
             Goldberger et al. 2000), 109 subjects, 64 channels at 160 Hz. Runs 1 and 2
             are one-minute eyes-open and eyes-closed resting baselines. Downloaded by
             MNE on first use into ~/mne_data; nothing is stored in this repository.

  heroic     HEROIC (Zenodo 10.5281/zenodo.11493897, CC BY 4.0): Muse 2 sessions with a
             real eyes-closed -> eyes-open transition at a spoken cue. The only source here
             recorded on the system's own hardware. Must be downloaded by hand; see the
             HEROIC section below.

  synthetic  1/f background plus an alpha-band component whose power steps by a known
             ratio between alternating 60 s eyes-open and eyes-closed blocks, at the
             Muse 2's 256 Hz. Transition times are exact by construction.

HOW AN EEGBCI SESSION IS ASSEMBLED

The two runs are recorded separately, so a labelled transition has to be made by
splicing: eyes-open 0-30 s | eyes-closed 0-60 s | eyes-open 30-60 s, giving 120 s with one
rising and one falling transition at 30 s and 90 s. Each segment is mean-centred per
channel before joining and the joins are crossfaded over 50 ms, so the splice does not add
a DC step that a filter would ring on. The transition is therefore instantaneous, which
makes measured detection latency an upper bound on the estimator alone: a real eye closure
also takes the alpha rhythm a few hundred milliseconds to build.

CHANNELS

The Muse 2 records TP9, AF7, AF8 and TP10. EEGBCI has AF7 and AF8 but no TP9/TP10; its
nearest temporal sites are T9 and T10, which stand in for them. Sessions are returned in
the Muse channel order (TP9, AF7, AF8, TP10) so the deployed FeatureExtractor runs on them
unchanged, with its channel names set to the sites actually used.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import signal as sps

EEGBCI_FS = 160.0
SYNTH_FS = 256.0
EEGBCI_SUBJECTS = tuple(range(1, 110))
EEGBCI_CHANNELS = ("T9", "AF7", "AF8", "T10")      # Muse order, temporal stand-ins
MUSE_NAMES = ("TP9", "AF7", "AF8", "TP10")
TEMPORAL = {"TP9": "T9", "TP10": "T10"}            # Muse name -> EEGBCI site

EO, EC = "eyes_open", "eyes_closed"


class Session(dict):
    """chans (4, n) in microvolts, fs, channels, timeline [(t, phase)...], name."""


def site(name: str, source: str) -> str:
    return TEMPORAL.get(name, name) if source == "eegbci" else name


def _read_run(subject: int, run: int) -> np.ndarray:
    import mne
    from mne.datasets import eegbci

    mne.set_log_level("ERROR")
    path = eegbci.load_data(subject, [run], update_path=False, verbose=False)[0]
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    eegbci.standardize(raw)
    if abs(raw.info["sfreq"] - EEGBCI_FS) > 1e-6:
        raise ValueError(f"S{subject:03d}R{run:02d}: fs {raw.info['sfreq']}")
    return raw.get_data(picks=list(EEGBCI_CHANNELS)) * 1e6         # volts -> uV


def _join(parts: list[np.ndarray], fs: float, fade_s: float = 0.05) -> np.ndarray:
    n = int(round(fade_s * fs))
    out = parts[0] - parts[0].mean(axis=1, keepdims=True)
    for p in parts[1:]:
        p = p - p.mean(axis=1, keepdims=True)
        w = np.linspace(0.0, 1.0, n)
        mid = out[:, -n:] * (1 - w) + p[:, :n] * w
        out = np.concatenate([out[:, :-n], mid, p[:, n:]], axis=1)
    return out


def eegbci_runs(subject: int) -> tuple[np.ndarray, np.ndarray]:
    """The two unspliced baseline runs, (eyes open, eyes closed), in uV."""
    return _read_run(subject, 1), _read_run(subject, 2)


def eegbci_session(subject: int) -> Session:
    eo, ec = eegbci_runs(subject)
    fs = EEGBCI_FS
    h, m = int(30 * fs), int(60 * fs)
    chans = _join([eo[:, :h], ec[:, :m], eo[:, h:m]], fs)
    return Session(chans=chans, fs=fs, channels=EEGBCI_CHANNELS, name=f"S{subject:03d}",
                   timeline=[(0.0, EO), (30.0, EC), (90.0, EO), (chans.shape[1] / fs, None)])


def _pink(rng, n: int) -> np.ndarray:
    """1/f noise by spectral shaping, unit variance."""
    f = np.fft.rfftfreq(n, 1.0)
    spec = rng.normal(size=f.size) + 1j * rng.normal(size=f.size)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0.0
    x = np.fft.irfft(spec, n)
    return x / x.std()


def synthetic_session(seed: int, ratio: float, blocks: int = 6, block_s: float = 60.0,
                      fs: float = SYNTH_FS, background_uv: float = 12.0,
                      alpha_uv: float = 6.0) -> Session:
    """
    Alternating eyes-open/eyes-closed blocks with a known alpha power ratio.

    The alpha component is narrowband noise (8-13 Hz) whose POWER is multiplied by
    `ratio` in eyes-closed blocks, applied as an instantaneous amplitude step at each block
    boundary. Background is 1/f noise, partly shared across channels as real scalp EEG is.
    """
    rng = np.random.default_rng(seed)
    n = int(blocks * block_s * fs)
    t = np.arange(n) / fs
    closed = (np.floor(t / block_s).astype(int) % 2) == 1
    gain = np.where(closed, np.sqrt(ratio), 1.0)

    bp = sps.butter(4, (8.0, 13.0), btype="bandpass", fs=fs, output="sos")
    shared_alpha = sps.sosfilt(bp, rng.normal(size=n))
    shared_alpha /= shared_alpha.std()
    shared_bg = _pink(rng, n)
    chans = np.empty((4, n))
    for c in range(4):
        own = sps.sosfilt(bp, rng.normal(size=n))
        own /= own.std()
        alpha = alpha_uv * gain * (0.8 * shared_alpha + 0.6 * own)
        bg = background_uv * (0.6 * shared_bg + 0.8 * _pink(rng, n))
        chans[c] = alpha + bg
    timeline = [(k * block_s, EC if k % 2 else EO) for k in range(blocks)]
    timeline.append((n / fs, None))
    return Session(chans=chans, fs=fs, channels=MUSE_NAMES, name=f"synth{seed:03d}",
                   timeline=timeline)


# --------------------------------------------------------------------- HEROIC (Muse 2)
#
# HEROIC (Zenodo 10.5281/zenodo.11493897, CC BY 4.0): Muse 2 recordings made at home and in
# the lab with the HEROIC platform. Every session opens with ~61 s eyes closed, a ~5 s
# spoken cue, then ~61 s eyes open (two subjects repeat the pair), recorded continuously
# at the Muse's 256 Hz on TP9, AF7, AF8, TP10. Unlike EEGBCI, the transitions are REAL:
# nothing is spliced, and the measured latency includes the participant's reaction to the
# cue. The transition is placed at cue onset (the end of the preceding block), which is
# when the participant is told to change state.
#
# Rules fixed before any estimator was run on this data:
#   * byte-identical recordings are kept once. The archive holds four sub005 sessions
#     that are copies of four sub012 sessions; those two folders are treated as ONE
#     subject, since which of them recorded it cannot be told.
#   * a session is excluded if any inter-sample gap inside the eye blocks (+-5 s)
#     exceeds 0.5 s. Shorter jitter is interpolated onto a uniform 256 Hz grid.
#
# Not stored in this repository. Download the zip from Zenodo, extract it, and point
# HEROIC_DIR at the public_software_paper_data folder (default ~/eeg_data/heroic/x/...).

HEROIC_FS = 256.0
HEROIC_MAX_GAP_S = 0.5
HEROIC_DIR = os.environ.get("HEROIC_DIR", os.path.join(
    os.path.expanduser("~"), "eeg_data", "heroic", "x", "public_software_paper_data"))
_HEROIC_SAME_PERSON = {"sub005": "sub005=sub012", "sub012": "sub005=sub012"}


def heroic_files(root: str = HEROIC_DIR) -> list[str]:
    """Unique EEG.csv paths, duplicates removed by content hash, in sorted order."""
    import glob
    import hashlib

    seen, out = set(), []
    for f in sorted(glob.glob(os.path.join(root, "sub*", "*", "*", "EEG.csv"))):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(f)
    return out


def heroic_session(path: str) -> Session:
    """One HEROIC recording on a uniform grid, or ValueError if it breaks the gap rule."""
    import json

    a = np.genfromtxt(path, delimiter=",", skip_header=1)
    t, raw = a[:, 0], a[:, 1:5].T                                  # TP9, AF7, AF8, TP10
    info = json.load(open(os.path.join(os.path.dirname(path), "session_info.json")))
    blocks = [b for b in info["task_log"] if b["name"] == "eyes_closed_eyes_open"]
    if not blocks:
        raise ValueError("no eye blocks")
    kinds = []
    for b in blocks:
        inside = (t >= b["start"]) & (t <= b["end"])
        marks = set(np.unique(a[inside, 5]).astype(int)) & {3, 4}
        if len(marks) != 1:
            raise ValueError(f"block markers {sorted(marks)}")
        kinds.append(EC if marks == {3} else EO)

    lo, hi = blocks[0]["start"] - 5.0, blocks[-1]["end"] + 5.0
    span = (t >= lo) & (t <= hi)
    gap = float(np.diff(t[span]).max())
    if gap > HEROIC_MAX_GAP_S:
        raise ValueError(f"{gap:.2f} s gap in the eye blocks")
    t0 = t[span][0]
    grid = np.arange(t0, t[span][-1], 1.0 / HEROIC_FS)
    chans = np.vstack([np.interp(grid, t, ch) for ch in raw])

    rel = lambda x: float(x - t0)                                  # noqa: E731
    timeline = [(rel(blocks[0]["start"]), kinds[0])]
    timeline += [(rel(blocks[i - 1]["end"]), kinds[i]) for i in range(1, len(blocks))]
    timeline.append((rel(blocks[-1]["end"]), None))
    parts = os.path.normpath(path).split(os.sep)
    sub, setting, rec = parts[-4], parts[-3], parts[-2][:15]
    return Session(chans=chans, fs=HEROIC_FS, channels=MUSE_NAMES,
                   name=f"{sub}/{setting}/{rec}", subject=_HEROIC_SAME_PERSON.get(sub, sub),
                   timeline=timeline)


def sessions(source: str, seeds: int = 50, ratio: float = 1.34, subjects=None):
    """Yield sessions, skipping (and naming) any subject whose files fail to load."""
    if source == "heroic":
        files = heroic_files()
        if not files:
            raise FileNotFoundError(f"no HEROIC data under {HEROIC_DIR}; see public_data.py")
        for f in files:
            try:
                yield heroic_session(f)
            except ValueError as exc:
                print(f"  [skip] {os.path.relpath(f, HEROIC_DIR)}: {exc}")
        return
    if source == "eegbci":
        for s in (subjects or EEGBCI_SUBJECTS):
            try:
                yield eegbci_session(s)
            except Exception as exc:                       # pragma: no cover
                print(f"  [skip] S{s:03d}: {exc}")
    elif source == "synthetic":
        for k in range(seeds):
            yield synthetic_session(k, ratio)
    else:
        raise ValueError(source)
