"""Burst windows, the two encoder views, and the pre-onset noise spectrum.

A burst window is an onset-aligned complex window of N = 16384 samples stored as float32 I/Q, shape [B, 2, N]. The
pre-onset segment [0, n0) with n0 = 512 holds receiver noise only; the transient view [n0, n0 + L) with L = 2048 holds
the transmitter's turn-on. WiFi preambles of 256 samples pass every view unchanged.
"""
import numpy as np
import torch

WINDOW = 16384   # N
ONSET = 512      # n0
VIEW = 2048      # L
EPS = 1e-12


def _check(iq):
    if not isinstance(iq, torch.Tensor) or iq.dtype != torch.float32 or iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError('expected float32 I/Q of shape [B, 2, samples]')
    if not bool(torch.isfinite(iq).all()):
        raise FloatingPointError('non-finite I/Q')
    return iq


def normalize_iq(iq):
    """Complex-RMS normalization per window, computed in float64."""
    x = _check(iq).to(torch.float64)
    rms = torch.sqrt(x.square().sum(dim=1).mean(dim=1))
    if not bool(torch.isfinite(rms).all()) or bool((rms <= EPS).any()):
        raise FloatingPointError('window with zero or non-finite power')
    return (x / rms[:, None, None]).to(torch.float32)


def rotate_phase(iq, phi):
    """Rotate each window by the phase phi [B] (random-phase training views)."""
    c, s = torch.cos(phi)[:, None], torch.sin(phi)[:, None]
    return torch.stack((iq[:, 0] * c - iq[:, 1] * s, iq[:, 0] * s + iq[:, 1] * c), dim=1)


def transient_view(iq, start=ONSET, keep=VIEW):
    """View of the transient-view encoder f_T: samples [start, start + keep), tiled to N, RMS-normalized."""
    if iq.shape[-1] != WINDOW:
        return iq
    return normalize_iq(iq[..., start:start + keep].repeat(1, 1, WINDOW // keep).contiguous())


def center_frequency(iq, head=(ONSET, ONSET + VIEW)):
    """Remove the window's mean frequency, estimated from the lag-1 autocorrelation phase over the burst head."""
    x = torch.complex(iq[:, 0].double(), iq[:, 1].double())
    seg = x[:, head[0]:head[1]]
    phi = torch.angle((seg[:, 1:] * seg[:, :-1].conj()).sum(-1))
    n = torch.arange(x.shape[-1], device=iq.device, dtype=torch.float64)
    y = x * torch.exp(-1j * phi[:, None] * n[None, :])
    return torch.stack((y.real, y.imag), 1).float()


def channel_view(iq):
    """View of the channel-augmented encoder f_A at inference: frequency centering, then the transient view."""
    if iq.shape[-1] != WINDOW:
        return iq
    return transient_view(normalize_iq(center_frequency(iq).contiguous()))


def preonset_spectrum(windows, onset=ONSET, segments=8):
    """Pre-onset noise spectrum phi(u) of one recording unit (eq. (12)).

    windows: sequence of arrays [2, N] (the bursts of the unit). Each pre-onset segment [0, onset) is split into
    `segments` Hann-windowed parts; their mean power spectrum is taken in log10, and its mean over bins is removed,
    which keeps the spectral shape and discards the level. The result is averaged over the bursts. A unit whose
    windows are not full-length bursts gets NaN (it then forms the group without spectra).
    """
    width = onset // segments
    out = np.full(width, np.nan)
    if not windows or any(w is None or np.shape(w)[-1] != WINDOW for w in windows):
        return out
    hann = np.hanning(width)
    acc = []
    for w in windows:
        w = np.asarray(w, dtype=np.float64)
        x = (w[0, :onset] + 1j * w[1, :onset]).reshape(segments, width) * hann
        p = np.log10(np.fft.fftshift((np.abs(np.fft.fft(x, axis=1)) ** 2).mean(0)) + 1e-20)
        acc.append(p - p.mean())
    return np.mean(acc, 0)
