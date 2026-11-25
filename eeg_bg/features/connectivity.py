"""Pairwise functional connectivity: coherence and Phase-Locking Value (PLV).

Computes magnitude-squared coherence and per-band PLV for the 8 homotopic
(bilateral) electrode pairs in ``asymmetry.SYMMETRIC_PAIRS`` across the 5
standard EEG frequency bands.

Pairs are restricted to homotopic pairs (rather than all C(19,2) = 171
possible pairs) for a physiological reason: reduced inter-hemispheric
coherence/PLV is a documented correlate of focal lateralized epileptogenic
networks. This measures inter-hemispheric *phase coupling*, which is a
distinct signal from the power-asymmetry features in ``asymmetry.py`` (same
electrode pairs, but comparing power rather than phase), so it is not
redundant with them — it is just far cheaper than scoring all 171 pairs.

Efficiency:
  - Coherence: one vectorised Welch STFT over all 19 channels, then all pair
    cross-spectral densities derived from that single FFT result.
  - PLV: per-band bandpass + Hilbert computed once per channel and reused
    across all pairs in that band.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from scipy.signal.windows import hann

from eeg_bg.features._constants import _STANDARD_19
from eeg_bg.features.asymmetry import SYMMETRIC_PAIRS
from eeg_bg.features.band_power import BANDS

ALL_PAIRS: list[tuple[str, str]] = list(SYMMETRIC_PAIRS)  # 8 homotopic pairs

def build_connectivity_names(pairs: Sequence[tuple[str, str]]) -> list[str]:
    return [
        f"{metric}_{left}_{right}_{band}"
        for left, right in pairs
        for band in BANDS
        for metric in ("coh", "plv")
    ]


CONNECTIVITY_NAMES = build_connectivity_names(ALL_PAIRS)


def _bandpass_sos(f_lo: float, f_hi: float, sfreq: float) -> np.ndarray:
    return butter(4, [f_lo, f_hi], btype="bandpass", fs=sfreq, output="sos")


def connectivity_features(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float = 125.0,
    nperseg: int = 250,
) -> np.ndarray:
    """Compute pairwise coherence and PLV for all standard channel pairs.

    Parameters
    ----------
    epoch : np.ndarray
        Shape ``(n_ch, n_times)``.
    ch_names : list[str]
        Channel names for axis 0 of *epoch*.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int
        Welch window length in samples (pipeline-consistent default 250).

    Returns
    -------
    np.ndarray
        Shape ``(80,)``, dtype float64.
        Layout: for each pair in ``ALL_PAIRS``, for each band in ``BANDS``,
        ``[coherence, PLV]``.  Absent-channel pairs → 10 zeros.
    """
    return connectivity_features_for_layout(epoch, ch_names, sfreq, _STANDARD_19, ALL_PAIRS, nperseg)


def connectivity_features_for_layout(
    epoch: np.ndarray,
    ch_names: Sequence[str],
    sfreq: float,
    channel_order: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    nperseg: int = 250,
) -> np.ndarray:
    """Compute the same connectivity blocks for an explicit channel layout."""
    layout = {name: i for i, name in enumerate(channel_order)}
    left_indices = np.array([layout[left] for left, _ in pairs], dtype=int)
    right_indices = np.array([layout[right] for _, right in pairs], dtype=int)
    ch_map = {name: i for i, name in enumerate(ch_names)}
    n_std = len(channel_order)
    n_times = epoch.shape[1]

    # ── Build (n_std, n_times) array; zero-fill absent channels ──────────────
    epoch_std = np.zeros((n_std, n_times), dtype=np.float64)
    present = np.zeros(n_std, dtype=bool)
    for si, ch in enumerate(channel_order):
        if ch in ch_map:
            epoch_std[si] = epoch[ch_map[ch]].astype(np.float64)
            present[si] = True

    # ── Vectorised Welch coherence ────────────────────────────────────────────
    # Build overlapping Hann-windowed segments: (n_std, n_seg, nperseg)
    step = nperseg // 2
    win = hann(nperseg, sym=False)
    starts = np.arange(0, n_times - nperseg + 1, step)
    seg_idx = starts[:, None] + np.arange(nperseg)[None, :]      # (n_seg, nperseg)
    segs = epoch_std[:, seg_idx]                               # (n_std, n_seg, nperseg)
    segs = segs - segs.mean(axis=2, keepdims=True)             # detrend: remove segment mean
    segs = segs * win[None, None, :]

    # FFT → (n_std, n_seg, n_freq)
    F = np.fft.rfft(segs, axis=2)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)

    # Auto-power per channel averaged over segments: (n_std, n_freq)
    Pxx = (F * F.conj()).real.mean(axis=1)

    Cxy = (F[left_indices] * F[right_indices].conj()).mean(axis=1)
    P1 = Pxx[left_indices]
    P2 = Pxx[right_indices]
    coh_all = (np.abs(Cxy) ** 2) / (P1 * P2 + 1e-30)

    # ── Per-band Hilbert phases for PLV ───────────────────────────────────────
    phases: dict[str, np.ndarray] = {}  # band_name → (n_std, n_times)
    for band_name, (f_lo, f_hi) in BANDS.items():
        sos = _bandpass_sos(f_lo, f_hi, sfreq)
        filtered = sosfiltfilt(sos, epoch_std, axis=1)  # (n_std, n_times)
        phases[band_name] = np.angle(hilbert(filtered, axis=1))  # (n_std, n_times)

    # ── Pair absent mask ──────────────────────────────────────────────────────
    pair_present = present[left_indices] & present[right_indices]

    # ── Compute per-band coh and plv arrays ──────────────────────────────────
    band_names = list(BANDS.keys())
    coh_bands: dict[str, np.ndarray] = {}
    plv_bands: dict[str, np.ndarray] = {}

    for band_name, (f_lo, f_hi) in BANDS.items():
        band_mask = (freqs >= f_lo) & (freqs <= f_hi)
        coh_b = coh_all[:, band_mask].mean(axis=1) if band_mask.any() \
                else np.zeros(len(pairs))
        ph1 = phases[band_name][left_indices]
        ph2 = phases[band_name][right_indices]
        plv_b = np.abs(np.mean(np.exp(1j * (ph1 - ph2)), axis=1))
        coh_bands[band_name] = np.where(pair_present, coh_b, 0.0)
        plv_bands[band_name] = np.where(pair_present, plv_b, 0.0)

    # ── Assemble in CONNECTIVITY_NAMES order: pair → band → [coh, plv] ───────
    feats = np.empty(len(pairs) * len(BANDS) * 2, dtype=np.float64)
    out_idx = 0
    for pi in range(len(pairs)):
        for band_name in band_names:
            feats[out_idx]     = coh_bands[band_name][pi]
            feats[out_idx + 1] = plv_bands[band_name][pi]
            out_idx += 2

    return feats
