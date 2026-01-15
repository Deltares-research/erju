import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from loguru import logger
from scipy.signal import medfilt, welch

# --- your project imports (only what we need) ---
from src.utils.db_utils import fetch_accel_data, unpack_timeseries
from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, create_results_folder


# =========================
# SPATIAL DECONVOLUTION HELPERS
# =========================
def boxcar_samples(L_m, dz_m=1.0):
    """Normalized boxcar of length L_m in meters; dz_m is channel spacing (m)."""
    n = max(1, int(round(L_m / dz_m)))
    return np.ones(n) / n


def center_and_pad_kernel(k, n_ch):
    """Pad kernel to n_ch and center it (zero-phase) to avoid spatial shifts."""
    k_full = np.zeros(n_ch)
    k_full[:len(k)] = k
    return np.roll(k_full, -len(k) // 2)


def cosine_taper(n, frac=0.04):
    """Cosine taper at both spatial ends to reduce FFT edge artifacts."""
    frac = max(0.0, min(frac, 0.5))
    m = int(frac * n)
    w = np.ones(n)
    if m > 0:
        x = np.linspace(0, np.pi / 2, m)
        w[:m] = np.sin(x) ** 2
        w[-m:] = np.sin(x[::-1]) ** 2
    return w


def estimate_epsilon_spatial(Y, window=7):
    """Estimate epsilon = noise_power / signal_power along SPACE (per event)."""
    n_times, n_ch = Y.shape
    window = min(window, max(3, (n_ch // 2) * 2 - 1))  # keep it small vs n_ch, odd
    if window % 2 == 0:
        window += 1
    smooth = medfilt(Y, kernel_size=(1, window))  # median along space
    noise = Y - smooth
    sp = np.var(smooth)
    npow = np.var(noise)
    return 1e-6 if sp == 0 else npow / sp


def deconvolve_space_wiener(Y, k10_full, epsilon, taper_frac=0.04):
    """
    Y: (n_times, n_ch) 10 m-gauge measurements.
    k10_full: centered, padded 10 m kernel (len=n_ch).
    epsilon: scalar regularization (noise/signal power).
    """
    n_times, n_ch = Y.shape
    W = cosine_taper(n_ch, taper_frac)
    K = np.fft.rfft(k10_full)
    Kc = np.conj(K)
    denom = (np.abs(K) ** 2 + float(epsilon) + 1e-12)

    Ehat = np.empty_like(Y)
    for t in range(n_times):
        Yk = np.fft.rfft(Y[t] * W)
        Ek = Kc * Yk / denom
        e = np.fft.irfft(Ek, n=n_ch)
        Ehat[t] = e / (W + 1e-12)
    return Ehat


def deconvolve_space_cls(Y, k_full, gamma=0.03, order=2, taper_frac=0.04):
    """
    CLS: minimize ||H*e - y||^2 + gamma||D e||^2   (D = finite-difference)
    gamma ~ 0.01–0.1; order=1 (first diff) or 2 (second diff).
    """
    n_times, n_ch = Y.shape

    # taper
    def cosine_taper(n, frac=0.04):
        frac = max(0.0, min(frac, 0.5))
        m = int(frac * n)
        w = np.ones(n)
        if m > 0:
            x = np.linspace(0, np.pi / 2, m)
            w[:m] = np.sin(x) ** 2
            w[-m:] = np.sin(x[::-1]) ** 2
        return w

    W = cosine_taper(n_ch, taper_frac)

    H = np.fft.rfft(k_full)
    # build D (finite-difference) and its spectrum
    if order == 1:
        d = np.array([1, -1], float)
    else:
        d = np.array([1, -2, 1], float)
    D_full = np.zeros(n_ch);
    D_full[:len(d)] = d
    D_full = np.roll(D_full, -len(d) // 2)
    Dk = np.fft.rfft(D_full)

    denom = (np.abs(H) ** 2 + gamma * (np.abs(Dk) ** 2) + 1e-12)
    Ehat = np.empty_like(Y)
    for t in range(n_times):
        Yk = np.fft.rfft(Y[t] * W)
        Ek = np.conj(H) * Yk / denom
        e = np.fft.irfft(Ek, n=n_ch)
        Ehat[t] = e / (W + 1e-12)
    return Ehat


def apply_kernel_space(X, k):
    """Spatial convolution via FFT, time-slice by time-slice. X: (n_times, n_ch)."""
    n_times, n_ch = X.shape
    k_full = center_and_pad_kernel(k, n_ch)
    K = np.fft.rfft(k_full)
    out = np.empty_like(X)
    for t in range(n_times):
        Xk = np.fft.rfft(X[t])
        Yk = Xk * K
        out[t] = np.fft.irfft(Yk, n=n_ch)
    return out


# =========================
# PLOTTING HELPERS
# =========================
def plot_time_traces(timestamps, measured, deconv, reblur, center_channel, out_png, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(timestamps, measured, label='Measured (10 m)', alpha=0.6)
    ax.plot(timestamps, deconv, label='Deconvolved', alpha=0.6)
    ax.plot(timestamps, reblur, '--', label='Re-blurred 10 m', alpha=0.6)
    ax.set_title(f"FO Channel {center_channel} {title_suffix}")
    ax.set_xlabel("Time");
    ax.set_ylabel("Strain (a.u.)")
    ax.grid(True);
    ax.legend();
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_psd(measured, deconv, reblur, fs, out_png, fmax=100.0, nperseg=4096):
    f_m, P_m = welch(measured, fs=fs, nperseg=min(nperseg, len(measured)))
    f_d, P_d = welch(deconv, fs=fs, nperseg=min(nperseg, len(deconv)))
    f_r, P_r = welch(reblur, fs=fs, nperseg=min(nperseg, len(reblur)))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f_m, P_m, label='Measured (10 m)')
    ax.plot(f_d, P_d, label='Deconvolved')
    ax.plot(f_r, P_r, '--', label='Re-blurred 10 m')
    ax.set_xlabel("Frequency (Hz)");
    ax.set_ylabel("PSD")
    ax.set_xlim(0, fmax);
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # --- Paths & basic config ---
    path_stem_db = r"P:/11207352-stem/database/Wielrondheid_132887.db"
    path_fo_data = r"E:\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels"
    path_save_res = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\10m GL\deconv"

    start_date = '2024-08-29 08:10:00'
    end_date = '2024-08-31 09:00:00'

    location_name = ['Meetjournal_MP8_Holten_zuid_4m_C']
    campaigns = None
    traintype = "SPR(A)"
    track = "1"

    # ---- Channels (use a broad window for stable inversion) ----
    CENTER_CHANNEL = 1194
    SPAN_CHANNELS = 400  # <-- ±150 → 301 channels; increase if fast enough
    first_channel = CENTER_CHANNEL - SPAN_CHANNELS
    last_channel = CENTER_CHANNEL + SPAN_CHANNELS

    # ---- Deconvolution knobs ----
    channel_spacing_m = 1.0
    gauge_length_m = 10.0
    epsilon_global = 0.05  # try 0.03–0.08; set to None to auto-estimate (uses small spatial window)
    median_win_eps = 7  # used only if epsilon_global=None
    taper_frac = 0.04

    # ---- Output dirs ----
    results_folder = create_results_folder(base_path=path_save_res,
                                           start_date=start_date, end_date=end_date,
                                           traintype=traintype, center_channel=CENTER_CHANNEL,
                                           track=track, Fpass=[1, 100])
    plots_folder = os.path.join(results_folder, "deconv_plots")
    os.makedirs(plots_folder, exist_ok=True)

    # --- Fetch events (we only need the time windows) ---
    events, tim, _ = fetch_accel_data(db_path=path_stem_db,
                                      start_date=start_date, end_date=end_date,
                                      locations=location_name, campaigns=campaigns,
                                      traintype=traintype, track=track, get_timeseries=True)
    event_series = list(tim.values())[0]

    t0 = time.time()
    logger.info(f"Deconvolution test for {len(events)} events… Using channels [{first_channel}, {last_channel}]")

    for idx, (event, (event_id, data)) in enumerate(zip(events, event_series.items()), start=1):
        logger.info(f"[{idx}/{len(events)}] Event {event_id}")

        # 1) time window from accel metadata
        _, _, _, _, time_window = unpack_timeseries(event, data)

        # 2) Load FO files overlapping the window
        fo = BaseFOdata.create_instance(dir_path=path_fo_data,
                                        first_channel=first_channel,
                                        last_channel=last_channel,
                                        reader='optasense')
        fo_files = from_window_get_fo_file(path_fo_data, time_window)
        if not fo_files:
            logger.warning(f"No FO files for event {event_id}. Skipping.")
            continue

        fo_data_list = []
        sampling_frequency = None
        for i, fpath in enumerate(fo_files):
            if i == 0:
                fo.extract_properties_per_file(fpath)
                file_start_time = fo.properties['FileStartTime']
                sampling_frequency = int(fo.properties['SamplingFrequency[Hz]'])
            processed, _ = fo.extract_data(file_name=fpath,
                                           first_channel=first_channel, last_channel=last_channel)
            fo_data_list.append(processed.T)  # (samples, channels)

        fo_data = np.concatenate(fo_data_list, axis=0)  # (n_times, n_channels)

        # 3) Crop FO to the window
        timestamps = [file_start_time + timedelta(seconds=i / sampling_frequency)
                      for i in range(fo_data.shape[0])]
        ts_np = np.array(timestamps, dtype='datetime64[ns]')
        s_idx = int(np.argmin(np.abs(ts_np - np.datetime64(time_window[0]))))
        e_idx = int(np.argmin(np.abs(ts_np - np.datetime64(time_window[1]))))
        timestamps = timestamps[s_idx:e_idx + 1]
        fo_data = fo_data[s_idx:e_idx + 1, :]

        if fo_data.size == 0:
            logger.warning(f"Empty FO slice for event {event_id}. Skipping.")
            continue

        # 4) Spatial deconvolution across channels
        n_times_ev, n_ch_ev = fo_data.shape
        k10 = boxcar_samples(gauge_length_m, channel_spacing_m)
        k10_full = center_and_pad_kernel(k10, n_ch_ev)

        # Try CLS instead of Wiener
        strain_est = deconvolve_space_cls(fo_data, k10_full, gamma=0.03, order=2, taper_frac=0.04)
        check_10m = apply_kernel_space(strain_est, k10)

        if epsilon_global is None:
            epsilon = estimate_epsilon_spatial(fo_data, window=median_win_eps)
        else:
            epsilon = float(epsilon_global)

        logger.info(f"[Event {event_id}] epsilon={epsilon:.3e} | n_ch={n_ch_ev} | fs={sampling_frequency} Hz")

        strain_est = deconvolve_space_wiener(fo_data, k10_full, epsilon, taper_frac=taper_frac)
        check_10m = apply_kernel_space(strain_est, k10)

        # Try CLS instead of Wiener
        strain_est = deconvolve_space_cls(fo_data, k10_full, gamma=0.03, order=2, taper_frac=0.04)
        check_10m = apply_kernel_space(strain_est, k10)

        # 5) Extract center channel traces
        ch_idx = CENTER_CHANNEL - first_channel
        trace_measured = fo_data[:, ch_idx]
        trace_deconv = strain_est[:, ch_idx]
        trace_reblur = check_10m[:, ch_idx]

        # 6) Save TIME plot
        out_png_time = os.path.join(plots_folder, f"time_ch{CENTER_CHANNEL}_event_{event_id}.png")
        plot_time_traces(timestamps, trace_measured, trace_deconv, trace_reblur,
                         CENTER_CHANNEL, out_png_time, title_suffix=f"— Event {event_id}")

        # 7) Save PSD plot (temporal Welch)
        out_png_psd = os.path.join(plots_folder, f"psd_ch{CENTER_CHANNEL}_event_{event_id}.png")
        plot_psd(trace_measured, trace_deconv, trace_reblur, fs=sampling_frequency,
                 out_png=out_png_psd, fmax=100.0, nperseg=4096)

    logger.success(f"Done. Plots in: {plots_folder}. Runtime: {(time.time() - t0) / 60:.2f} min")
