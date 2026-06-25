"""Utilities for building the single-channel FO waveform dataset (v1).

Pipeline for one event
----------------------
1. Read ``/fo/strain`` (time x channels) and ``/fo/fs_hz`` from the event
   NetCDF file.
2. Locate channel 1194 via ``geometry_fo/fo_channel_id`` (robust lookup, not a
   hard-coded array index).
3. Bandpass-filter the single channel (1-100 Hz, Butterworth, zero-phase) —
   the same filter as the tabular feature pipeline.
4. Resample 1000 Hz -> 250 Hz (polyphase ``scipy.signal.resample_poly``), which
   approximately preserves the 1-100 Hz band (Nyquist at 250 Hz = 125 Hz).
5. Crop/pad to a fixed length centred on the channel-1194 energy centroid
   (FO-only; no accelerometer information is used).  No amplitude
   normalization is applied.

The returned waveform keeps physical amplitude (strain units) so the model can
learn the amplitude -> PGV relationship directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import iirfilter, resample_poly, sosfiltfilt, zpk2sos

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _to_python_scalar(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except Exception:
            pass
    return v


def _safe_float(v: Any, default: float = np.nan) -> float:
    v = _to_python_scalar(v)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def bandpass_single_channel(
    x: np.ndarray,
    fs: float,
    freqmin: float,
    freqmax: float,
    corners: int,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass on a 1-D signal.

    Identical filter design to ``parquet_v1_utils._bandpass_fo`` but for a
    single channel (1-D input).
    """
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    z, p, k = iirfilter(
        corners, [low, high], btype="band", ftype="butter", output="zpk"
    )
    sos = zpk2sos(z, p, k)
    return sosfiltfilt(sos, x)


def resample_signal(
    x: np.ndarray,
    fs_in: float,
    fs_out: float,
) -> np.ndarray:
    """Resample a 1-D signal from ``fs_in`` to ``fs_out`` using polyphase.

    ``resample_poly`` applies an internal anti-alias FIR filter, so the
    1-100 Hz band is approximately preserved while higher-frequency content is
    suppressed before decimation.  Amplitude is preserved (no rescaling).
    """
    if fs_in == fs_out:
        return x.astype(np.float64, copy=False)
    # Reduce up/down by their gcd so the rational factor is minimal.
    up = int(round(fs_out))
    down = int(round(fs_in))
    g = gcd(up, down)
    up //= g
    down //= g
    return resample_poly(x, up=up, down=down).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Channel extraction
# ---------------------------------------------------------------------------


@dataclass
class RawChannel:
    """Raw single-channel signal pulled from one event NetCDF file."""

    signal: np.ndarray          # 1-D strain trace (native fs)
    fs_hz: float                # native sampling rate
    channel_index: int          # array index of the channel in /fo/strain
    n_samples: int              # original sample count


def extract_channel_signal(
    dataset: Any,
    target_channel: int,
) -> Tuple[Optional[RawChannel], str]:
    """Pull the single channel ``target_channel`` from one event dataset.

    Returns ``(RawChannel, "ok")`` on success or ``(None, reason)`` on failure.
    The channel is located by absolute ID via ``geometry_fo/fo_channel_id`` so
    the implementation is robust to differing stored channel ranges.
    """
    if "fo" not in dataset.groups:
        return None, "missing_fo_group"
    fo_grp = dataset.groups["fo"]
    if "strain" not in fo_grp.variables or "fs_hz" not in fo_grp.variables:
        return None, "missing_fo_strain_or_fs"

    fs_hz = _safe_float(fo_grp.variables["fs_hz"][()], default=np.nan)
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        return None, "invalid_fs"

    if "geometry_fo" not in dataset.groups:
        return None, "missing_geometry_fo"
    geom = dataset.groups["geometry_fo"]
    if "fo_channel_id" not in geom.variables:
        return None, "missing_fo_channel_id"

    ch_ids = np.asarray(geom.variables["fo_channel_id"][:], dtype=np.int64)
    matches = np.where(ch_ids == int(target_channel))[0]
    if len(matches) == 0:
        return None, "target_channel_not_stored"
    ch_index = int(matches[0])

    strain = fo_grp.variables["strain"]
    if strain.ndim != 2 or strain.shape[0] < 2:
        return None, "invalid_strain_shape"
    if ch_index >= strain.shape[1]:
        return None, "channel_index_out_of_range"

    # Read only the one channel column (avoids loading the full 51-ch block).
    sig = np.asarray(strain[:, ch_index], dtype=np.float64)
    if sig.size < 2 or not np.all(np.isfinite(sig)):
        # Replace non-finite with 0 but flag tiny signals as invalid.
        if sig.size < 2:
            return None, "signal_too_short"
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    return (
        RawChannel(
            signal=sig,
            fs_hz=float(fs_hz),
            channel_index=ch_index,
            n_samples=int(sig.size),
        ),
        "ok",
    )


# ---------------------------------------------------------------------------
# Energy-centred crop / pad
# ---------------------------------------------------------------------------


@dataclass
class CropResult:
    """Outcome of cropping/padding one processed waveform to fixed length."""

    waveform: np.ndarray        # fixed-length 1-D array (target fs)
    was_padded: bool
    was_cropped: bool
    crop_start: int             # start index in the *processed* (resampled) signal
    crop_end: int               # end index (exclusive) in the processed signal
    center_index: int           # energy centre used for the crop
    method: str


def _moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Smooth |x|^2 with a centred moving average, return the RMS envelope."""
    if win <= 1:
        return np.abs(x)
    power = x * x
    kernel = np.ones(win, dtype=np.float64) / win
    smoothed = np.convolve(power, kernel, mode="same")
    return np.sqrt(np.clip(smoothed, 0.0, None))


def _energy_center_index(env: np.ndarray, method: str) -> int:
    """Return the crop-centre index from an energy envelope."""
    if method == "peak_envelope":
        return int(np.argmax(env))
    # Default: energy centroid (sum_i i * e_i / sum_i e_i) — robust to spikes.
    total = float(env.sum())
    if total <= 0:
        return int(len(env) // 2)
    idx = np.arange(len(env), dtype=np.float64)
    return int(round(float(np.dot(idx, env) / total)))


def crop_or_pad_energy_centered(
    x: np.ndarray,
    target_len: int,
    fs: float,
    method: str = "energy_centroid",
    envelope_smooth_s: float = 0.5,
    pad_value: float = 0.0,
) -> CropResult:
    """Crop or zero-pad ``x`` to ``target_len`` centred on its energy centre.

    The energy centre is computed from a smoothed RMS envelope of ``x`` (the
    same processed signal used for the model input), so no accelerometer
    information is involved.  Amplitude is preserved; padding uses zeros.
    """
    n = int(x.size)
    smooth_win = max(1, int(round(envelope_smooth_s * fs)))
    env = _moving_rms(x, smooth_win)
    center = _energy_center_index(env, method)

    if n == target_len:
        return CropResult(
            waveform=x.astype(np.float32, copy=False),
            was_padded=False,
            was_cropped=False,
            crop_start=0,
            crop_end=n,
            center_index=center,
            method=method,
        )

    if n > target_len:
        # Crop a window of target_len centred on the energy centre, clamped.
        start = center - target_len // 2
        start = max(0, min(start, n - target_len))
        end = start + target_len
        out = x[start:end].astype(np.float32, copy=False)
        return CropResult(
            waveform=out,
            was_padded=False,
            was_cropped=True,
            crop_start=int(start),
            crop_end=int(end),
            center_index=center,
            method=method,
        )

    # n < target_len -> zero-pad.  Place the signal centred on the energy
    # centre so the passage stays near the middle of the fixed window.
    out = np.full(target_len, pad_value, dtype=np.float32)
    offset = target_len // 2 - center
    offset = max(0, min(offset, target_len - n))
    out[offset : offset + n] = x.astype(np.float32, copy=False)
    return CropResult(
        waveform=out,
        was_padded=True,
        was_cropped=False,
        crop_start=int(-offset),       # negative: signal starts after pad
        crop_end=int(n - offset),
        center_index=center,
        method=method,
    )


# ---------------------------------------------------------------------------
# Full per-event processing
# ---------------------------------------------------------------------------


@dataclass
class ProcessedEvent:
    """Final processed waveform + per-event diagnostics for one event."""

    waveform: np.ndarray            # fixed-length float32 (target fs)
    fs_out: float
    # Original-signal diagnostics
    n_samples_original: int
    duration_original_s: float
    channel_index: int
    # Crop diagnostics (in original-sample units where requested)
    was_padded: bool
    was_cropped: bool
    crop_start_sample_original: int
    crop_end_sample_original: int
    crop_center_method: str
    # For diagnostic plotting (processed, pre-crop signal + crop indices)
    processed_signal: np.ndarray    # resampled, pre-crop (target fs)
    crop_start_processed: int
    crop_end_processed: int
    center_index_processed: int


def process_event_waveform(
    dataset: Any,
    target_channel: int,
    bandpass_freqmin: float,
    bandpass_freqmax: float,
    bandpass_corners: int,
    target_fs_hz: float,
    fixed_length_samples: int,
    crop_center_method: str,
    envelope_smooth_s: float,
    pad_value: float,
) -> Tuple[Optional[ProcessedEvent], str]:
    """Run the full single-channel pipeline for one event dataset.

    Returns ``(ProcessedEvent, "ok")`` or ``(None, reason)``.
    """
    raw, reason = extract_channel_signal(dataset, target_channel)
    if raw is None:
        return None, reason

    fs_in = raw.fs_hz
    n_orig = raw.n_samples
    duration_s = n_orig / fs_in if fs_in > 0 else np.nan

    # A degenerate (near-empty) signal cannot be filtered meaningfully.
    if n_orig < int(round(0.5 * fs_in)):  # < 0.5 s of data
        return None, "signal_too_short_for_processing"

    # 1. Bandpass at native fs.
    try:
        filtered = bandpass_single_channel(
            raw.signal, fs_in, bandpass_freqmin, bandpass_freqmax, bandpass_corners
        )
    except Exception as exc:  # filter can fail on pathological lengths
        return None, f"bandpass_failed:{type(exc).__name__}"

    # 2. Resample to target fs.
    processed = resample_signal(filtered, fs_in, target_fs_hz)

    # 3. Crop / pad centred on energy.
    crop = crop_or_pad_energy_centered(
        processed,
        target_len=fixed_length_samples,
        fs=target_fs_hz,
        method=crop_center_method,
        envelope_smooth_s=envelope_smooth_s,
        pad_value=pad_value,
    )

    # Map processed-sample crop indices back to original-sample units.
    fs_ratio = fs_in / target_fs_hz
    crop_start_orig = int(round(crop.crop_start * fs_ratio))
    crop_end_orig = int(round(crop.crop_end * fs_ratio))

    return (
        ProcessedEvent(
            waveform=crop.waveform,
            fs_out=float(target_fs_hz),
            n_samples_original=n_orig,
            duration_original_s=float(duration_s),
            channel_index=raw.channel_index,
            was_padded=crop.was_padded,
            was_cropped=crop.was_cropped,
            crop_start_sample_original=crop_start_orig,
            crop_end_sample_original=crop_end_orig,
            crop_center_method=crop.method,
            processed_signal=processed.astype(np.float32, copy=False),
            crop_start_processed=crop.crop_start,
            crop_end_processed=crop.crop_end,
            center_index_processed=crop.center_index,
        ),
        "ok",
    )
