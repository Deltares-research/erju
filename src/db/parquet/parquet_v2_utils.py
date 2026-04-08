"""Helpers for building Parquet v2 from event-level NetCDF files.

v2 vs v1: the only change is in spectral feature computation.
  v1 used uniform 5-Hz linear bins (fo_band_XXX_YYY_energy).
  v2 uses 1/3-octave bands (ISO 18405:2017, base-2) starting from 1 Hz,
  producing logarithmically spaced bands that better match structural
  vibration physics (fo_oct_XXXhz_energy).

All other utilities (IO, filtering, time-domain features, metadata loading,
geometry, etc.) are re-exported directly from parquet_v1_utils so there is
no duplication.
"""

from __future__ import annotations

from typing import Dict, List, Any

import numpy as np
from scipy.signal import welch

# Re-export everything from v1 that is unchanged
from src.db.parquet.parquet_v1_utils import (  # noqa: F401  (re-exports)
    SkipRecord,
    assemble_output_row,
    compute_fo_time_domain_features,
    compute_pgv_z_mms,
    create_build_folder,
    get_sensor_ids,
    load_event_metadata,
    load_sensor_geometry,
    open_netcdf_event,
    process_fo_event_signal,
    reduce_channel_features,
    save_json,
    validate_row_eligibility,
    write_build_log,
    write_parquet,
    _safe_float,
)

import netCDF4 as nc


# ---------------------------------------------------------------------------
# 1/3-octave spectral band features
# ---------------------------------------------------------------------------


def _oct_feature_name(nominal_hz: float) -> str:
    """Column name prefix for one 1/3-octave band.

    Examples:
      1.0  -> 'fo_oct_1_00hz'
      12.5 -> 'fo_oct_12_50hz'
      100  -> 'fo_oct_100hz'
    """
    if nominal_hz >= 10:
        return f"fo_oct_{int(nominal_hz):03d}hz"
    return f"fo_oct_{nominal_hz:.2f}hz".replace(".", "_")


def compute_fo_octave_band_features(
    strain_ch: np.ndarray,
    fs_hz: float,
    octave_bands: List[Dict[str, float]],
    nperseg: int,
    nfft: int,
) -> Dict[str, float]:
    """Band-integrated Welch PSD power per 1/3-octave band.

    Each band value = sum(Pxx[mask] * df)  [strain² units, length-independent]

    Parameters
    ----------
    strain_ch   : 1-D filtered strain signal for one FO channel
    fs_hz       : sampling frequency in Hz
    octave_bands: list of dicts with keys 'nominal_hz', 'lower_hz', 'upper_hz'
    nperseg     : Welch window length in samples
    nfft        : FFT length (zero-padded)
    """
    x = np.asarray(strain_ch, dtype=np.float64)

    # Return NaN columns if signal is too short for Welch
    if x.size < 2 or x.size < nperseg:
        return {
            _oct_feature_name(b["nominal_hz"]): np.nan for b in octave_bands
        }

    freqs, pxx = welch(
        x,
        fs=fs_hz,
        window="hamming",
        nperseg=nperseg,
        nfft=nfft,
        scaling="density",
        detrend="linear",
    )
    df = freqs[1] - freqs[0]

    out: Dict[str, float] = {}
    for band in octave_bands:
        name = _oct_feature_name(band["nominal_hz"])
        lo   = band["lower_hz"]
        hi   = band["upper_hz"]
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.sum(pxx[mask]) * df) if np.any(mask) else 0.0
    return out


# ---------------------------------------------------------------------------
# Event-level FO feature computation (v2 version)
# ---------------------------------------------------------------------------


def compute_event_fo_features(
    dataset: nc.Dataset,
    fo_config: Any,
    octave_bands: List[Dict[str, float]],
    reductions: List[str],
    include_time_domain: bool,
    include_spectral: bool,
) -> Dict[str, float] | None:
    """Compute event-level FO features using 1/3-octave spectral bands.

    Mirrors parquet_v1_utils.compute_event_fo_features but calls
    compute_fo_octave_band_features instead of compute_fo_spectral_band_features.
    """
    processed = process_fo_event_signal(dataset=dataset, fo_config=fo_config)
    if processed is None:
        return None

    fs_hz = _safe_float(
        dataset.groups["fo"].variables["fs_hz"][()], default=np.nan
    )

    per_ch: List[Dict[str, float]] = []
    for ch_idx in range(processed.shape[1]):
        ch_signal = processed[:, ch_idx]
        ch_feats: Dict[str, float] = {}

        if include_time_domain:
            ch_feats.update(compute_fo_time_domain_features(ch_signal))

        if include_spectral:
            ch_feats.update(
                compute_fo_octave_band_features(
                    strain_ch=ch_signal,
                    fs_hz=fs_hz,
                    octave_bands=octave_bands,
                    nperseg=fo_config.welch_nperseg,
                    nfft=fo_config.welch_nfft,
                )
            )

        per_ch.append(ch_feats)

    return reduce_channel_features(per_channel_features=per_ch, reductions=reductions)
