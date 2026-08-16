"""
eeg_features.py - spectral feature extraction for the live Muse 2 path.

Replaces src/analyze.py (retired to testing/analyze_legacy.py).

Three things changed relative to the legacy version, and all three matter:

1. THE SAMPLING RATE IS NEVER A LITERAL IN THIS FILE.
   FeatureConfig requires it to be passed in, and the only supported source is
   inlet.info().nominal_srate() via stream_utils.get_inlet(). The Muse 2 streams
   EEG at 256 Hz. The old live path declared 128 Hz, which halved every frequency
   axis: the 8-13 Hz "alpha" mask was really reading 16-26 Hz, the 13-30 Hz "beta"
   mask was really reading 26-60 Hz, and 60 Hz mains hum landed exactly on the
   inclusive upper edge of the beta mask. That is why the resting ratio sat near
   0.4 and needed 0.35/0.55 hysteresis to stay put. See scripts/verify_sample_rate.py.

2. Band power is INTEGRATED over the band (trapezoid over the PSD), not averaged
   across bins. Averaging makes the number depend on how many bins happen to fall
   inside the mask, which changes with window length. Integration does not.

3. The output is log10(beta/alpha), and it is NOT clamped. The legacy
   max(0, min(ratio, 5)) clamp silently discarded the most extreme observations,
   which are exactly the ones a therapeutic effect would appear in. Log scale also
   makes the distribution roughly symmetric, which is what the z-scoring in
   BaselineNormalizer assumes.

Artifact rejection is explicit: a window that fails validity returns valid=False
with a reason, and callers must NOT feed invalid windows into the smoother.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch, detrend

# Canonical band edges. Upper edges are exclusive in the mask so adjacent bands
# never double-count the boundary bin (the legacy code counted 13 Hz in both
# alpha and beta, and 30 Hz in beta where the aliased mains peak sat).
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

MUSE_CHANNELS = ("TP9", "AF7", "AF8", "TP10")


@dataclass
class FeatureConfig:
    """Everything the extractor needs. sampling_rate has no default on purpose."""

    sampling_rate: float
    window_seconds: float = 4.0
    hop_seconds: float = 1.0

    channels: Sequence[str] = MUSE_CHANNELS
    # AF7/AF8 are the frontal pair. The legacy testing/main.py read sample[0]
    # (TP9, a temporal electrode) while live_music.py averaged AF7+AF8; that
    # inconsistency is resolved here by naming the channels explicitly.
    frontal_channels: Sequence[str] = ("AF7", "AF8")

    bandpass: Tuple[float, float] = (1.0, 45.0)
    notch_hz: Optional[float] = 60.0  # 50.0 outside North America
    notch_q: float = 30.0
    filter_order: int = 4

    bands: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )

    # Artifact thresholds, applied to the band-passed signal in microvolts.
    #
    # 350 uV is derived from measurement, not chosen. From the 2026-08-16 alpha
    # validation on TP9+TP10 (315 windows, 6 minutes), window peak-to-peak was:
    #
    #                  50%     95%     99%     max
    #   eyes closed     94     163     199     201     <- clean signal ceiling
    #   eyes open      158     325     947     961     <- long tail is blinks
    #
    # Normal variation tops out around 325 and real blinks start above 900, so 350
    # sits in the gap. It keeps 96% of eyes-open windows and 100% of eyes-closed.
    #
    # The previous 150 uV was a guess, and it discarded 54% of eyes-open windows
    # against 10% of eyes-closed - a differential rejection that inflated the
    # measured alpha ratio from ~1.9x to 2.49x. See scripts/alpha_sensitivity.py.
    #
    # This value is MONTAGE-SPECIFIC (temporal sites run quieter than frontal) and
    # PARTICIPANT-SPECIFIC. It must be fixed before pre-registration rather than
    # tuned per session, otherwise the effect size is being chosen after seeing the
    # data.
    reject_peak_to_peak_uv: float = 350.0
    reject_flatline_uv: float = 0.1
    reject_max_abs_uv: float = 500.0

    @property
    def window_samples(self) -> int:
        return int(round(self.sampling_rate * self.window_seconds))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sampling_rate * self.hop_seconds))

    def frontal_indices(self) -> Tuple[int, ...]:
        return tuple(self.channels.index(name) for name in self.frontal_channels)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        d["frontal_channels"] = list(self.frontal_channels)
        d["window_samples"] = self.window_samples
        d["hop_samples"] = self.hop_samples
        return d


@dataclass
class BandFeatures:
    """One analysis window's worth of features."""

    timestamp: float
    valid: bool
    reject_reason: Optional[str] = None

    delta: float = float("nan")
    theta: float = float("nan")
    alpha: float = float("nan")
    beta: float = float("nan")
    gamma: float = float("nan")

    log_beta_alpha: float = float("nan")
    total_power: float = float("nan")
    peak_freq: float = float("nan")
    peak_to_peak_uv: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureExtractor:
    """Stateless per-window spectral extraction. Build once, call extract() repeatedly."""

    def __init__(self, config: FeatureConfig):
        if config.sampling_rate is None or not math.isfinite(config.sampling_rate):
            raise ValueError("sampling_rate must be a finite number from the LSL stream")
        if config.sampling_rate < 64:
            raise ValueError(
                f"sampling_rate={config.sampling_rate} Hz is implausibly low for a Muse 2. "
                "Run scripts/verify_sample_rate.py before collecting data."
            )
        self.cfg = config
        nyq = config.sampling_rate / 2.0

        lo, hi = config.bandpass
        hi = min(hi, nyq * 0.95)
        self._sos = butter(config.filter_order, [lo / nyq, hi / nyq], btype="band", output="sos")

        self._notch = None
        if config.notch_hz and config.notch_hz < nyq * 0.95:
            self._notch = iirnotch(config.notch_hz, config.notch_q, fs=config.sampling_rate)

        # Welch segmenting: 2 s segments give 0.5 Hz resolution, which resolves the
        # 5 Hz-wide alpha band into 10 bins. Falls back to the whole window if short.
        self._nperseg = int(min(config.window_samples, round(config.sampling_rate * 2.0)))
        self._noverlap = self._nperseg // 2

        # filtfilt needs roughly 3 * max(len(a), len(b)) samples of runway.
        self._min_samples = max(int(config.sampling_rate * 1.0), 3 * (config.filter_order * 2 + 1))

    # ---------------------------------------------------------------- public

    def filter_signal(self, x: np.ndarray) -> np.ndarray:
        """
        Linear-detrend, band-pass, and notch a single channel. Raises ValueError if
        the segment is too short for filtfilt.

        THIS IS THE ONLY PLACE FILTERING IS DEFINED. Any code that judges signal
        amplitude must call this first, because raw Muse EEG carries large slow
        drift - electrode polarization and post-donning settling - that dwarfs the
        neural signal. Mean subtraction does not remove drift; the 1 Hz high-pass
        does. Amplitude thresholds applied to unfiltered data are therefore
        measuring drift, not signal quality, and will condemn perfectly usable
        channels.

        scripts/contact_check.py calls this so its verdicts match what the pipeline
        actually accepts. Those two disagreeing is a bug, not a conservative margin.
        """
        signal = detrend(np.asarray(x, dtype=np.float64), type="linear")
        signal = sosfiltfilt(self._sos, signal)
        if self._notch is not None:
            b, a = self._notch
            signal = filtfilt(b, a, signal)
        return signal

    def extract(self, window: np.ndarray, timestamp: float = 0.0) -> BandFeatures:
        """
        window: array shaped (n_channels, n_samples) in microvolts, channel order
                matching config.channels.

        Never raises on bad data. Returns valid=False with a reason instead, so the
        caller can log the rejection (rejection rate is a reportable signal-quality
        metric) without a try/except around the control loop.
        """
        window = np.asarray(window, dtype=np.float64)
        if window.ndim != 2:
            return BandFeatures(timestamp, False, "window must be 2-D (channels, samples)")
        if window.shape[1] < self._min_samples:
            return BandFeatures(timestamp, False, f"too few samples ({window.shape[1]})")
        if not np.all(np.isfinite(window)):
            return BandFeatures(timestamp, False, "non-finite samples in window")

        idx = self.cfg.frontal_indices()
        if max(idx) >= window.shape[0]:
            return BandFeatures(timestamp, False, "window has fewer channels than configured")

        raw_frontal = window[list(idx), :].mean(axis=0)

        if float(np.std(raw_frontal)) < self.cfg.reject_flatline_uv:
            return BandFeatures(timestamp, False, "flatline (electrode not contacting)")

        try:
            signal = self.filter_signal(raw_frontal)
        except ValueError as exc:
            return BandFeatures(timestamp, False, f"filter failed: {exc}")

        p2p = float(np.ptp(signal))
        if p2p > self.cfg.reject_peak_to_peak_uv:
            return BandFeatures(timestamp, False, "amplitude artifact (blink/jaw/motion)", peak_to_peak_uv=p2p)
        if float(np.max(np.abs(signal))) > self.cfg.reject_max_abs_uv:
            return BandFeatures(timestamp, False, "saturation", peak_to_peak_uv=p2p)

        freqs, psd = welch(
            signal,
            fs=self.cfg.sampling_rate,
            nperseg=self._nperseg,
            noverlap=self._noverlap,
            window="hann",
            detrend="constant",
            scaling="density",
        )

        powers = {name: self._integrate(freqs, psd, lo, hi) for name, (lo, hi) in self.cfg.bands.items()}

        alpha = powers.get("alpha", float("nan"))
        beta = powers.get("beta", float("nan"))
        if not (alpha > 0 and beta > 0):
            return BandFeatures(timestamp, False, "zero power in alpha or beta", peak_to_peak_uv=p2p)

        analysis = (freqs >= self.cfg.bandpass[0]) & (freqs < self.cfg.bands["gamma"][1])
        peak_freq = float(freqs[analysis][np.argmax(psd[analysis])]) if analysis.any() else float("nan")

        return BandFeatures(
            timestamp=timestamp,
            valid=True,
            reject_reason=None,
            delta=powers.get("delta", float("nan")),
            theta=powers.get("theta", float("nan")),
            alpha=alpha,
            beta=beta,
            gamma=powers.get("gamma", float("nan")),
            log_beta_alpha=float(np.log10(beta / alpha)),
            total_power=float(sum(v for v in powers.values() if np.isfinite(v))),
            peak_freq=peak_freq,
            peak_to_peak_uv=p2p,
        )

    # --------------------------------------------------------------- private

    @staticmethod
    def _integrate(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
        """Trapezoidal integration of the PSD over [lo, hi). Bin-count independent."""
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() < 2:
            return float("nan")
        return float(np.trapz(psd[mask], freqs[mask]))


class ExponentialSmoother:
    """
    Single-pole low-pass on the feature stream.

    Replaces the 5-sample boxcar. A boxcar of length N at hop h costs (N-1)*h/2
    seconds of group delay - at the old 5 samples / 2 s hop that was 4 s of pure
    lag, stacked on top of window fill. This has a configurable time constant and
    a much shorter effective delay for the same amount of blink suppression.

    Invalid windows must simply not be passed to update(); the state then holds,
    which is the correct behaviour (an artifact is missing information, not a
    measurement of zero).
    """

    def __init__(self, hop_seconds: float, tau_seconds: float = 3.0):
        if hop_seconds <= 0 or tau_seconds <= 0:
            raise ValueError("hop_seconds and tau_seconds must be positive")
        self.hop_seconds = hop_seconds
        self.tau_seconds = tau_seconds
        self.alpha = 1.0 - math.exp(-hop_seconds / tau_seconds)
        self.value: Optional[float] = None
        self.n_updates = 0

    @property
    def group_delay_seconds(self) -> float:
        """Effective lag of this filter, for the latency budget table."""
        return self.tau_seconds

    def update(self, x: float) -> float:
        if not math.isfinite(x):
            return self.value if self.value is not None else float("nan")
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        self.n_updates += 1
        return self.value

    def reset(self) -> None:
        self.value = None
        self.n_updates = 0


class BaselineNormalizer:
    """
    Per-participant z-scoring against their own eyes-open resting baseline.

    Absolute beta/alpha values are not comparable across people (skull thickness,
    electrode contact, hair, baseline arousal all shift them by more than any
    intervention does). Without this you cannot pool participants, and fixed
    thresholds like 0.35/0.55 are meaningless outside the one person they were
    tuned on.

    finalize() deliberately raises when the baseline is unusable. A failed baseline
    means bad electrode contact, and catching it before the intervention starts is
    the entire point of having a baseline phase.
    """

    def __init__(self, min_windows: int = 30, min_sd: float = 1e-3):
        self.min_windows = min_windows
        self.min_sd = min_sd
        self._samples: list[float] = []
        self.mean: Optional[float] = None
        self.sd: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self._samples)

    @property
    def is_ready(self) -> bool:
        return self.mean is not None and self.sd is not None

    def add(self, x: float) -> None:
        if math.isfinite(x):
            self._samples.append(float(x))

    def finalize(self) -> Tuple[float, float]:
        if self.n < self.min_windows:
            raise RuntimeError(
                f"Baseline has only {self.n} valid windows (need {self.min_windows}). "
                "Almost always bad electrode contact - reseat the headband, wet the "
                "sensors slightly, and rerun. Do not proceed with the intervention."
            )
        arr = np.asarray(self._samples, dtype=np.float64)
        self.mean = float(arr.mean())
        self.sd = float(arr.std(ddof=1))
        if self.sd < self.min_sd:
            raise RuntimeError(f"Baseline SD is {self.sd:.2e} - the signal is not varying. Check the headset.")
        return self.mean, self.sd

    def normalize(self, x: float) -> float:
        if not self.is_ready:
            raise RuntimeError("normalize() called before finalize()")
        if not math.isfinite(x):
            return float("nan")
        return (x - self.mean) / self.sd

    def to_dict(self) -> dict:
        return {
            "baseline_mean_log_beta_alpha": self.mean,
            "baseline_sd_log_beta_alpha": self.sd,
            "baseline_n_windows": self.n,
        }


def latency_budget(config: FeatureConfig, smoother_tau: float) -> Dict[str, float]:
    """
    The irreducible analysis-path latency, in seconds, for the paper's budget table.

    This is bottleneck #1 from the roadmap and no amount of GPU optimization
    touches any line of it.
    """
    window_fill = config.window_seconds / 2.0  # centroid of the window vs. its right edge
    hop_quantization = config.hop_seconds / 2.0
    return {
        "window_centroid_delay_s": window_fill,
        "hop_quantization_s": hop_quantization,
        "smoother_group_delay_s": smoother_tau,
        "total_analysis_latency_s": window_fill + hop_quantization + smoother_tau,
    }
