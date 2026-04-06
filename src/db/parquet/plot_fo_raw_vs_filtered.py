"""Plot FO strain before and after bandpass filtering for a single event.

Shows 3 rows (one per selected FO channel), 2 columns:
  - Left:  raw strain from /fo/strain (demean + Tukey + phase-to-ε, no filter)
  - Right: bandpass-filtered strain (same settings as Parquet builder)

The PNG is saved to the most recent parquet build folder under
<output_root>/plots/.  If no build folder exists yet, it falls back to
saving next to this script.

Usage (from project root):
  python src/db/parquet/plot_fo_raw_vs_filtered.py

Adjust NC_FILE and the settings block below as needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.signal import iirfilter, sosfiltfilt, zpk2sos

# ---------------------------------------------------------------------------
# Settings — edit as needed
# ---------------------------------------------------------------------------
NC_FILE = (
    r"P:\11210978-erju-ai\holten_db"
    r"\netcdf_20260405_001947_10mGL_3000channels\EVENT_0014.nc"
)

# Parquet output root — latest build folder is auto-detected from here.
PARQUET_OUTPUT_ROOT = Path(r"P:\11210978-erju-ai\holten_parquet")

# Bandpass settings (must match config_parquet.py FOProcessingConfig)
FREQMIN = 1.0   # Hz
FREQMAX = 100.0  # Hz
CORNERS = 5

# Number of FO channels to plot (picks centre-1, centre, centre+1)
N_CHANNELS = 3
# ---------------------------------------------------------------------------


def _latest_build_plots_dir(output_root: Path) -> Path:
    """Return <latest_build>/plots/, creating it if needed.

    Falls back to a plots/ subfolder next to this script when no build
    folder exists in output_root.
    """
    if output_root.exists():
        candidates = sorted(
            [d for d in output_root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )
        if candidates:
            plots_dir = candidates[0] / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            return plots_dir

    fallback = Path(__file__).parent / "plots"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _bandpass(
    data: np.ndarray, freqmin: float, freqmax: float, fs: float, corners: int
) -> np.ndarray:
    fe = 0.5 * fs
    z, p, k = iirfilter(
        corners,
        [freqmin / fe, freqmax / fe],
        btype="band",
        ftype="butter",
        output="zpk",
    )
    sos = zpk2sos(z, p, k)
    return sosfiltfilt(sos, data, axis=0)


def main() -> None:
    nc_path = Path(NC_FILE)
    with nc.Dataset(nc_path, "r") as ds:
        if "fo" not in ds.groups:
            print("No /fo group found in file.")
            return

        fo = ds.groups["fo"]
        if "strain" not in fo.variables or "time_s" not in fo.variables:
            print("Missing /fo/strain or /fo/time_s.")
            return

        time_s = np.asarray(fo.variables["time_s"][:], dtype=np.float64)
        strain = np.asarray(fo.variables["strain"][:], dtype=np.float64)
        fs_hz = float(fo.variables["fs_hz"][()])

        channel_ids = None
        centre_ch = None
        if "geometry_fo" in ds.groups:
            gfo = ds.groups["geometry_fo"]
            if "fo_channel_id" in gfo.variables:
                channel_ids = np.asarray(gfo.variables["fo_channel_id"][:], dtype=int)
            if "centre_fo_channel" in gfo.variables:
                try:
                    centre_ch = int(gfo.variables["centre_fo_channel"][()])
                except Exception:
                    pass

        event_id = getattr(ds, "event_id", nc_path.stem)

    n_ch = strain.shape[1]
    if channel_ids is not None and centre_ch is not None:
        centre_idx = int(np.argmin(np.abs(channel_ids - centre_ch)))
    else:
        centre_idx = n_ch // 2

    offsets = list(range(-(N_CHANNELS // 2), N_CHANNELS // 2 + 1))
    selected = [max(0, min(n_ch - 1, centre_idx + o)) for o in offsets]
    selected = list(dict.fromkeys(selected))

    if channel_ids is not None:
        labels = [f"ch {channel_ids[i]}" for i in selected]
    else:
        labels = [f"idx {i}" for i in selected]

    filtered = _bandpass(
        strain[:, selected],
        freqmin=FREQMIN,
        freqmax=FREQMAX,
        fs=fs_hz,
        corners=CORNERS,
    )

    n_sel = len(selected)
    fig, axes = plt.subplots(
        n_sel,
        2,
        figsize=(16, 3.5 * n_sel),
        sharex=True,
        sharey="row",
    )
    if n_sel == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"FO strain — raw vs bandpass-filtered ({FREQMIN}–{FREQMAX} Hz)\n"
        f"Event: {event_id}  |  fs={fs_hz:.0f} Hz",
        fontsize=11,
    )

    for row, (ch_col, label) in enumerate(zip(range(n_sel), labels)):
        raw_trace = strain[:, selected[ch_col]]
        filt_trace = filtered[:, ch_col]

        axes[row, 0].plot(time_s, raw_trace, lw=0.6, color="steelblue")
        axes[row, 0].set_ylabel(f"{label}\nstrain (ε)", fontsize=8)
        axes[row, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[row, 0].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 0].set_title(
                "Raw strain\n(demean + Tukey + phase→ε)", fontsize=9
            )

        axes[row, 1].plot(time_s, filt_trace, lw=0.6, color="darkorange")
        axes[row, 1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[row, 1].grid(True, alpha=0.3)
        if row == 0:
            axes[row, 1].set_title(
                f"Bandpass-filtered strain\n({FREQMIN}–{FREQMAX} Hz, order {CORNERS})",
                fontsize=9,
            )

    for ax in axes[-1]:
        ax.set_xlabel("Time relative to event_t0_utc (s)")

    fig.tight_layout()

    plots_dir = _latest_build_plots_dir(PARQUET_OUTPUT_ROOT)
    output_png = plots_dir / f"fo_raw_vs_filtered_{nc_path.stem}.png"
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()
