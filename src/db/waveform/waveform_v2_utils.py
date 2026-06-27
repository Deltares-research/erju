"""Utilities for building the multi-channel FO waveform dataset (v2).

Extracts a local channel window (default 1184-1204 = 21 channels) per event
and processes it into a fixed (n_channels, T) tensor.

Pipeline per event
------------------
1. Read ``/fo/strain`` (time x channels) and locate the channel window via
   ``geometry_fo/fo_channel_id`` (robust ID lookup, not array indices).
2. Bandpass-filter every channel (1-100 Hz, zero-phase Butterworth).
3. Resample 1000 -> 250 Hz (polyphase) along the time axis.
4. Compute one crop window from the centre channel (1194) energy centroid and
   apply it identically to all channels (shared time axis).  No amplitude
   normalization; zero-pad short events.

Reuses the single-channel helpers from ``waveform_v1_utils`` where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from scipy.signal import iirfilter, resample_poly, sosfiltfilt, zpk2sos

from src.db.waveform.waveform_v1_utils import (
    _energy_center_index,
    _moving_rms,
    _safe_float,
)


# ---------------------------------------------------------------------------
# Multi-channel signal processing
# ---------------------------------------------------------------------------


def bandpass_multichannel(
    block: np.ndarray,         # (n_channels, n_time)
    fs: float,
    freqmin: float,
    freqmax: float,
    corners: int,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass applied to every channel (along time)."""
    fe = 0.5 * fs
    z, p, k = iirfilter(
        corners, [freqmin / fe, freqmax / fe], btype="band", ftype="butter",
        output="zpk",
    )
    sos = zpk2sos(z, p, k)
    return sosfiltfilt(sos, block, axis=1)


def resample_multichannel(
    block: np.ndarray,         # (n_channels, n_time)
    fs_in: float,
    fs_out: float,
) -> np.ndarray:
    """Resample every channel along the time axis (axis=1)."""
    if fs_in == fs_out:
        return block.astype(np.float64, copy=False)
    from math import gcd

    up = int(round(fs_out))
    down = int(round(fs_in))
    g = gcd(up, down)
    return resample_poly(block, up=up // g, down=down // g, axis=1).astype(
        np.float64, copy=False
    )


# ---------------------------------------------------------------------------
# Channel-window extraction
# ---------------------------------------------------------------------------


@dataclass
class RawBlock:
    block: np.ndarray          # (n_channels, n_time) strain at native fs
    fs_hz: float
    channel_ids: np.ndarray    # absolute channel IDs (length n_channels)
    center_pos: int            # index of the centre channel within the block
    n_time: int


def extract_channel_window(
    dataset: Any,
    center_channel: int,
    channel_lo: int,
    channel_hi: int,
    channel_stride: int = 1,
) -> Tuple[Optional[RawBlock], str]:
    """Pull the channel window [channel_lo, channel_hi] for one event.

    Channels are ordered by ascending absolute channel ID.  Returns
    ``(RawBlock, "ok")`` or ``(None, reason)``.
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

    ch_ids_all = np.asarray(geom.variables["fo_channel_id"][:], dtype=np.int64)

    # Select channels whose absolute ID is inside the requested window.
    want = (ch_ids_all >= channel_lo) & (ch_ids_all <= channel_hi)
    sel_idx = np.where(want)[0]
    if len(sel_idx) == 0:
        return None, "no_channels_in_window"

    # Order by ascending channel ID for a consistent spatial axis.
    order = np.argsort(ch_ids_all[sel_idx])
    sel_idx = sel_idx[order]
    # Decimate by stride (every Nth channel) to reduce gauge-length redundancy.
    if channel_stride > 1:
        sel_idx = sel_idx[::channel_stride]
    sel_ids = ch_ids_all[sel_idx]

    strain = fo_grp.variables["strain"]
    if strain.ndim != 2 or strain.shape[0] < 2:
        return None, "invalid_strain_shape"

    # Read selected columns -> (n_time, n_sel) then transpose to (n_sel, n_time).
    block = np.asarray(strain[:, sel_idx], dtype=np.float64).T
    if not np.all(np.isfinite(block)):
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)

    center_matches = np.where(sel_ids == int(center_channel))[0]
    center_pos = int(center_matches[0]) if len(center_matches) else len(sel_ids) // 2

    return (
        RawBlock(
            block=block,
            fs_hz=float(fs_hz),
            channel_ids=sel_ids,
            center_pos=center_pos,
            n_time=int(block.shape[1]),
        ),
        "ok",
    )


# ---------------------------------------------------------------------------
# Full per-event processing
# ---------------------------------------------------------------------------


@dataclass
class ProcessedBlock:
    block: np.ndarray              # (n_channels, T) float32, target fs
    fs_out: float
    n_channels: int
    channel_ids: np.ndarray
    n_samples_original: int
    duration_original_s: float
    was_padded: bool
    was_cropped: bool
    crop_start_sample_original: int
    crop_end_sample_original: int
    crop_center_method: str
    # for diagnostics
    processed_block: np.ndarray    # (n_channels, n_time') pre-crop, target fs
    crop_start_processed: int
    crop_end_processed: int
    center_index_processed: int
    center_pos: int


def process_event_block(
    dataset: Any,
    center_channel: int,
    channel_lo: int,
    channel_hi: int,
    bandpass_freqmin: float,
    bandpass_freqmax: float,
    bandpass_corners: int,
    target_fs_hz: float,
    fixed_length_samples: int,
    crop_center_method: str,
    envelope_smooth_s: float,
    pad_value: float,
    expected_n_channels: int,
    channel_stride: int = 1,
) -> Tuple[Optional[ProcessedBlock], str]:
    """Full multi-channel pipeline for one event dataset."""
    raw, reason = extract_channel_window(
        dataset, center_channel, channel_lo, channel_hi, channel_stride
    )
    if raw is None:
        return None, reason

    # Require the full expected channel count so the array is rectangular.
    if raw.block.shape[0] != expected_n_channels:
        return None, f"channel_count_{raw.block.shape[0]}_expected_{expected_n_channels}"

    fs_in = raw.fs_hz
    n_orig = raw.n_time
    duration_s = n_orig / fs_in if fs_in > 0 else np.nan
    if n_orig < int(round(0.5 * fs_in)):
        return None, "signal_too_short_for_processing"

    # 1. Bandpass all channels.
    try:
        filtered = bandpass_multichannel(
            raw.block, fs_in, bandpass_freqmin, bandpass_freqmax, bandpass_corners
        )
    except Exception as exc:
        return None, f"bandpass_failed:{type(exc).__name__}"

    # 2. Resample all channels.
    processed = resample_multichannel(filtered, fs_in, target_fs_hz)  # (n_ch, n_time')
    n_proc = processed.shape[1]
    target_len = fixed_length_samples

    # 3. Crop window from the centre channel energy centroid.
    smooth_win = max(1, int(round(envelope_smooth_s * target_fs_hz)))
    env = _moving_rms(processed[raw.center_pos], smooth_win)
    center = _energy_center_index(env, crop_center_method)

    was_padded = False
    was_cropped = False
    if n_proc == target_len:
        out = processed.astype(np.float32, copy=False)
        crop_start, crop_end = 0, n_proc
    elif n_proc > target_len:
        start = center - target_len // 2
        start = max(0, min(start, n_proc - target_len))
        end = start + target_len
        out = processed[:, start:end].astype(np.float32, copy=False)
        crop_start, crop_end = int(start), int(end)
        was_cropped = True
    else:  # pad
        out = np.full((processed.shape[0], target_len), pad_value, dtype=np.float32)
        offset = target_len // 2 - center
        offset = max(0, min(offset, target_len - n_proc))
        out[:, offset : offset + n_proc] = processed.astype(np.float32, copy=False)
        crop_start, crop_end = int(-offset), int(n_proc - offset)
        was_padded = True

    fs_ratio = fs_in / target_fs_hz
    return (
        ProcessedBlock(
            block=out,
            fs_out=float(target_fs_hz),
            n_channels=out.shape[0],
            channel_ids=raw.channel_ids,
            n_samples_original=n_orig,
            duration_original_s=float(duration_s),
            was_padded=was_padded,
            was_cropped=was_cropped,
            crop_start_sample_original=int(round(crop_start * fs_ratio)),
            crop_end_sample_original=int(round(crop_end * fs_ratio)),
            crop_center_method=crop_center_method,
            processed_block=processed.astype(np.float32, copy=False),
            crop_start_processed=int(crop_start),
            crop_end_processed=int(crop_end),
            center_index_processed=int(center),
            center_pos=int(raw.center_pos),
        ),
        "ok",
    )
