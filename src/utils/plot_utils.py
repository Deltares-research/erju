import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import plotly.io as pio

from plotly.subplots import make_subplots

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objs as go
import plotly.io as pio

from src.utils.file_utils import compute_psd, create_results_folder, create_subfolder, timewindow

from SignalProcessingTools.time_signal import FilterDesign


def plot_sig_acc_fo(save_dir,
                    event_id,
                    accel_time,
                    trace_x,
                    trace_y,
                    trace_z,
                    fo_time,
                    fo_data,
                    fo_channel,
                    first_channel,
                    save_interactive=False):
    """
    Create a 3x2 plot comparing accelerometer data with FO data for a given event.
    Saves both a static PNG and optionally an interactive HTML using Plotly.

    Args:
        save_dir (str): Directory to save the plot.
        event_id (str or int): ID of the event to use in the filename.
        accel_time (list): Timestamps for the accelerometer data.
        trace_x (np.array): Accelerometer trace in X.
        trace_y (np.array): Accelerometer trace in Y.
        trace_z (np.array): Accelerometer trace in Z.
        fo_time (list): Timestamps for the FO data.
        fo_data (np.array): FO data array [timesteps, channels].
        fo_channel (int): Channel to extract from FO data.
        first_channel (int): First FO channel in the data array.
        save_interactive (bool): Whether to also save as interactive HTML with Plotly.
    """
    save_dir = create_subfolder(save_dir, "sig_acc_fo")
    ch_index = fo_channel - first_channel
    accel_traces = [trace_x, trace_y, trace_z]
    labels = ["Velocity X (mm/s)", "Velocity Y (mm/s)", "Velocity Z (mm/s)"]

    # Use viridis colormap
    viridis = cm.get_cmap('viridis')
    accel_colors = [mcolors.to_hex(viridis(i)) for i in [0.2, 0.4, 0.6]]
    fo_color = mcolors.to_hex(viridis(0.85))

    # ----------- Static PNG (Matplotlib) -----------
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Accelerometer vs FO Data - Event {event_id}")

    for i in range(3):
        # Left: Accelerometer
        axes[i, 0].plot(accel_time, accel_traces[i], color=accel_colors[i], alpha=0.8)
        axes[i, 0].set_ylabel(labels[i])
        axes[i, 0].set_title("Accelerometer")
        axes[i, 0].grid(True)
        if i == 2:
            axes[i, 0].set_xlabel("Time")

        # Right: FO
        axes[i, 1].plot(fo_time, fo_data[:, ch_index], color=fo_color, alpha=0.8)
        axes[i, 1].set_ylabel("FO strain (ε)")
        axes[i, 1].set_title(f"FO Channel {fo_channel}")
        axes[i, 1].grid(True)
        if i == 2:
            axes[i, 1].set_xlabel("Time")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    png_path = os.path.join(save_dir, f"event_{event_id}_accel_vs_fo.png")
    fig.savefig(png_path, format='png')
    plt.close(fig)
    # print(f"Saved PNG to:   {png_path}")

    # ----------- Interactive HTML (Plotly) -----------
    if save_interactive:
        fig_plotly = make_subplots(rows=3, cols=2, shared_xaxes=True, subplot_titles=[
            "Accel X", f"FO Channel {fo_channel}",
            "Accel Y", f"FO Channel {fo_channel}",
            "Accel Z", f"FO Channel {fo_channel}"
        ])

        for i, trace in enumerate(accel_traces):
            fig_plotly.add_trace(go.Scatter(
                x=accel_time,
                y=trace,
                mode='lines',
                line=dict(color=accel_colors[i], width=2),
                name=labels[i]
            ), row=i + 1, col=1)

            fig_plotly.add_trace(go.Scatter(
                x=fo_time,
                y=fo_data[:, ch_index],
                mode='lines',
                line=dict(color=fo_color, width=2),
                name=f"FO {fo_channel}"
            ), row=i + 1, col=2)

        fig_plotly.update_layout(
            height=900,
            width=1200,
            title_text=f"Accelerometer vs FO - Event {event_id}",
            showlegend=False,
            margin=dict(l=50, r=50, t=60, b=60)
        )

        for i in range(1, 4):
            fig_plotly.update_yaxes(title_text="Signal", row=i, col=1, showgrid=True)
            fig_plotly.update_yaxes(title_text="Signal", row=i, col=2, showgrid=True)
        fig_plotly.update_xaxes(title_text="Time", row=3, col=1, showgrid=True)
        fig_plotly.update_xaxes(title_text="Time", row=3, col=2, showgrid=True)

        html_path = os.path.join(save_dir, f"event_{event_id}_accel_vs_fo.html")
        pio.write_html(fig_plotly, file=html_path, auto_open=False)
        # print(f"Saved interactive HTML to: {html_path}")


def plot_sig_fo_raw_and_processed(save_dir,
                                  event_id,
                                  timestamps,
                                  raw_signal_data,
                                  processed_data,
                                  fo_channel,
                                  first_channel,
                                  save_interactive=False):
    """
    Plot FO signal before and after filtering/conversion to strain for a single channel.
    Saves both a static PNG and (optionally) an interactive HTML using Plotly.

    Args:
        save_dir (str): Directory where plots should be saved.
        event_id (str or int): ID to include in filenames.
        timestamps (list): List of datetime timestamps.
        raw_signal_data (np.array): Raw FO signal data before processing [timesteps, channels].
        processed_data (np.array): Processed FO signal after bandpass + strain [timesteps, channels].
        fo_channel (int): Channel number to extract.
        first_channel (int): First channel index in the FO data.
        save_interactive (bool): If True, also saves a Plotly HTML version.
    """
    save_dir = create_subfolder(save_dir, "sig_fo_raw and processed")
    ch_index = fo_channel - first_channel

    # Get two colors from the viridis colormap
    viridis = cm.get_cmap('viridis')
    color1 = mcolors.to_hex(viridis(0.2))
    color2 = mcolors.to_hex(viridis(0.8))

    # ----------- Static PNG (Matplotlib) -----------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)

    axes[0].plot(timestamps, raw_signal_data[:, ch_index], color=color1, alpha=0.85)
    axes[0].set_title(f"Event {event_id} - FO Channel {fo_channel} (Before Filtering)")
    axes[0].set_ylabel("Raw Signal (Optical phase)")
    axes[0].grid(True)

    axes[1].plot(timestamps, processed_data[:, ch_index], color=color2, alpha=0.85)
    axes[1].set_title(f"Event {event_id} - FO Channel {fo_channel} (After Filtering + Strain)")
    axes[1].set_ylabel("Strain (ε)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True)

    fig.tight_layout()
    png_path = os.path.join(save_dir, f"event_{event_id}_fo_before_after.png")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    # print(f"Saved static FO plot to: {png_path}")

    # ----------- Interactive HTML (Plotly) -----------
    if save_interactive:
        fig_plotly = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=(
                f"FO Channel {fo_channel} - Before Filtering",
                f"FO Channel {fo_channel} - After Filtering + Strain"
            )
        )

        fig_plotly.add_trace(go.Scatter(
            x=timestamps,
            y=raw_signal_data[:, ch_index],
            mode='lines',
            name='Before Filtering',
            line=dict(color=color1)
        ), row=1, col=1)

        fig_plotly.add_trace(go.Scatter(
            x=timestamps,
            y=processed_data[:, ch_index],
            mode='lines',
            name='After Filtering + Strain',
            line=dict(color=color2)
        ), row=2, col=1)

        fig_plotly.update_layout(
            height=600,
            width=1000,
            title_text=f"Event {event_id} - FO Channel {fo_channel} Comparison",
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=60)
        )

        fig_plotly.update_yaxes(title_text="Signal", row=1, col=1, showgrid=True)
        fig_plotly.update_yaxes(title_text="Signal", row=2, col=1, showgrid=True)
        fig_plotly.update_xaxes(title_text="Time", row=2, col=1, showgrid=True)

        html_path = os.path.join(save_dir, f"event_{event_id}_fo_before_after.html")
        pio.write_html(fig_plotly, file=html_path, auto_open=False)

        # print(f"Saved interactive FO plot to: {html_path}")


def plot_psd_comparison(
        save_dir,
        event_id,
        fx, psd_x,
        fy, psd_y,
        fz, psd_z,
        ff, psd_fo,
        freq_range=(0, 100),
        save_interactive=False
):
    """
    Plot precomputed PSDs for accelerometer axes and FO signal.
    Saves both a static PNG and optionally an interactive HTML with subplots (4 rows).

    Args:
        save_dir (str): Directory to save the plot.
        event_id (str or int): Event identifier.
        fx, fy, fz, ff (np.array): Frequency axes for X, Y, Z, and FO.
        psd_x, psd_y, psd_z, psd_fo (np.array): PSD values.
        freq_range (tuple): Frequency range to display (min, max).
        save_interactive (bool): Whether to save an interactive HTML (Plotly).
    """
    os.makedirs(save_dir, exist_ok=True)

    # Use viridis colormap
    viridis = cm.get_cmap('viridis')
    colors = [mcolors.to_hex(viridis(i)) for i in [0.2, 0.4, 0.6, 0.85]]
    labels = ["Velocity X (mm/s)", "Velocity Y (mm/s)", "Velocity Z (mm/s)", "FO strain (ε)"]

    # ----------- Static PNG (Matplotlib) -----------
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10), sharex=True)

    for ax, freq, psd, color, label in zip(
            axes, [fx, fy, fz, ff], [psd_x, psd_y, psd_z, psd_fo], colors, labels
    ):
        ax.semilogy(freq, psd, color=color, alpha=0.8)
        ax.set_ylabel("PSD")
        ax.set_title(label)
        ax.set_xlim(freq_range)
        ax.grid(True)

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle(f"PSD Comparison - Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    png_path = os.path.join(save_dir, f"event_{event_id}_psd_comparison.png")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    # print(f"Saved static PSD plot to: {png_path}")

    # ----------- Interactive HTML (Plotly) -----------
    if save_interactive:
        fig_plotly = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                   subplot_titles=labels)

        traces = [
            go.Scatter(x=fx, y=psd_x, mode='lines', name='Accel X', line=dict(color=colors[0])),
            go.Scatter(x=fy, y=psd_y, mode='lines', name='Accel Y', line=dict(color=colors[1])),
            go.Scatter(x=fz, y=psd_z, mode='lines', name='Accel Z', line=dict(color=colors[2])),
            go.Scatter(x=ff, y=psd_fo, mode='lines', name='FO', line=dict(color=colors[3]))
        ]

        for i, trace in enumerate(traces):
            fig_plotly.add_trace(trace, row=i + 1, col=1)

        fig_plotly.update_layout(
            height=1000,
            width=1000,
            title=f"PSD Comparison - Event {event_id}",
            showlegend=False,
            margin=dict(l=60, r=40, t=60, b=60)
        )

        for i in range(1, 5):
            fig_plotly.update_yaxes(title_text="PSD", type="log", row=i, col=1, showgrid=True)
            fig_plotly.update_xaxes(title_text="Frequency [Hz]", range=freq_range, row=i, col=1, showgrid=True)

        html_path = os.path.join(save_dir, f"event_{event_id}_psd_comparison.html")
        pio.write_html(fig_plotly, file=html_path, auto_open=False)
        # print(f"Saved interactive PSD plot to: {html_path}")


def plot_sig_psd_acc(
        event_id,
        accel_time,
        trace_x,
        trace_y,
        trace_z,
        fs=1000,
        save_dir=".",
        freq_range=(0, 100),
        fo_for_crop=None,
):
    """
    Plot accelerometer signals and their PSDs (X, Y, Z) in 3x2 format.

    Args:
        event_id (str/int): Identifier for the event (used in filename).
        accel_time (list of datetime): Time axis for the traces.
        trace_x/y/z (np.array): Accelerometer signals.
        fs (int): Sampling frequency in Hz.
        save_dir (str): Directory to save the output plot.
        freq_range (tuple): Frequency range to display in PSD plots (e.g., (0, 100)).
    """
    save_dir = create_subfolder(save_dir, "sig_psd_accel")
    labels = ["X", "Y", "Z"]
    traces = [trace_x, trace_y, trace_z]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8), sharex='col')

    for i, trace in enumerate(traces):
        # Left: Time signal
        axes[i, 0].plot(accel_time, trace.signal[:len(fo_for_crop)], alpha=0.8)
        axes[i, 0].set_ylabel(f"Velocity {labels[i]} (mm/s)")
        axes[i, 0].grid(True)

        # Right: PSD
        axes[i, 1].plot(trace.frequency_Pxx, trace.Pxx, alpha=0.8)
        axes[i, 1].set_ylabel(f"PSD {labels[i]}")
        axes[i, 1].set_xlim(freq_range)
        axes[i, 1].grid(True)

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Frequency [Hz]")

    fig.suptitle(f"Accelerometer Signals & PSD - Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"event_{event_id}_accel_signals_and_psd.png"
    fig.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

    # print(f"Saved accelerometer signal & PSD plot to: {os.path.join(save_dir, filename)}")


import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_utils import compute_psd


def plot_sig_psd_acc_fo(event_id,
                        save_dir,
                        accel_time,
                        trace_x,
                        trace_y,
                        trace_z,
                        fo_time,
                        fo_trace,
                        len_w,
                        fo_channel,
                        first_channel,
                        fs_accel=1000,
                        fs_fo=1000,
                        freq_range=(0, 100),
                        save_interactive=False):
    """
    Plot accelerometer (X, Y, Z) and FO signals with their PSDs in a 4x2 format.
    Optionally saves an interactive Plotly version as HTML.

    Args:
        event_id (str/int): Identifier for the event (used in filename).
        accel_time (list of datetime): Time axis for the accelerometer.
        trace_x/y/z (np.array): Accelerometer signals.
        fo_time (list of datetime): Time axis for the FO signal.
        fo_trace (np.array): FO signal array (2D: time x channels).
        fs_accel (int): Accelerometer sampling frequency.
        fs_fo (int): FO sampling frequency.
        save_dir (str): Directory to save the output plot.
        freq_range (tuple): Frequency range for PSDs.
        save_interactive (bool): If True, also save an interactive Plotly version.
        len_w (int): Length of the PSD window.
    """

    ch_index = fo_channel - first_channel
    fo_trace = fo_trace[:, ch_index]

    labels = ["X", "Y", "Z", "FO"]
    traces = [trace_x, trace_y, trace_z, fo_trace]
    time_axes = [accel_time] * 3 + [fo_time]
    sample_rates = [fs_accel] * 3 + [fs_fo]

    original_save_dir = save_dir
    for window in len_w:
        save_dir = create_subfolder(original_save_dir, f"sig_psd_accel_fo_{window}")

        # ---------- Matplotlib PNG Plot ----------
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 10), sharex='col')

        for i in range(4):
            freq, psd = compute_psd(traces[i], fs=sample_rates[i], length_w=window)

            # Time-domain plot (left)
            axes[i, 0].plot(time_axes[i], traces[i], alpha=0.8)
            axes[i, 0].set_ylabel(f"Velocity {labels[i]} (mm/s)" if labels[i] != "FO" else "FO strain (ε)")
            axes[i, 0].grid(True)

            # PSD (right)
            axes[i, 1].plot(freq, psd, alpha=0.8)
            axes[i, 1].set_ylabel(f"PSD {labels[i]}")
            axes[i, 1].set_xlim(freq_range)
            axes[i, 1].grid(True)

        axes[3, 0].set_xlabel("Time [s]")
        axes[3, 1].set_xlabel("Frequency [Hz]")

        fig.suptitle(f"Accelerometer & FO Signals with PSD - Event {event_id}")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        filename = f"event_{event_id}_accel_fo_with_psd.png"
        fig.savefig(os.path.join(save_dir, filename))
        plt.close(fig)

        # print(f"Saved static accelerometer + FO signal & PSD plot to: {os.path.join(save_dir, filename)}")

        # ---------- Plotly HTML Plot ----------
        if save_interactive:
            viridis = cm.get_cmap('viridis')
            colors = [mcolors.to_hex(viridis(i)) for i in [0.1, 0.3, 0.6, 0.85]]

            fig_plotly = make_subplots(rows=4, cols=2,
                                       shared_xaxes=True,
                                       column_widths=[0.5, 0.5],
                                       horizontal_spacing=0.08,
                                       subplot_titles=[f"{lbl} Signal" for lbl in labels] + [f"{lbl} PSD" for lbl in
                                                                                             labels])

            for i in range(4):
                freq, psd = compute_psd(traces[i], fs=sample_rates[i])
                fig_plotly.add_trace(go.Scatter(x=time_axes[i], y=traces[i],
                                                mode='lines',
                                                name=f"{labels[i]} Signal",
                                                line=dict(color=colors[i])),
                                     row=i + 1, col=1)

                fig_plotly.add_trace(go.Scatter(x=freq, y=psd,
                                                mode='lines',
                                                name=f"{labels[i]} PSD",
                                                line=dict(color=colors[i])),
                                     row=i + 1, col=2)

            fig_plotly.update_layout(height=1000,
                                     width=1200,
                                     title_text=f"Accelerometer & FO Signals with PSD - Event {event_id}",
                                     showlegend=False,
                                     margin=dict(l=60, r=40, t=60, b=60))

            for i in range(1, 5):
                fig_plotly.update_yaxes(title_text=labels[i - 1], row=i, col=1, showgrid=True)
                fig_plotly.update_yaxes(title_text=f"PSD {labels[i - 1]}", row=i, col=2, showgrid=True,
                                        type="linear")
                fig_plotly.update_xaxes(title_text="Time", row=i, col=1)
                fig_plotly.update_xaxes(title_text="Frequency [Hz]", row=i, col=2, range=freq_range)

            html_path = os.path.join(save_dir, f"event_{event_id}_accel_fo_with_psd.html")
            fig_plotly.write_html(html_path, auto_open=False)

            # print(f"Saved interactive HTML plot to: {html_path}")


def plot_sig_acc_raw_and_processed(event_id,
                                   time,
                                   trace_x,
                                   trace_y,
                                   trace_z,
                                   trace_x_filt,
                                   trace_y_filt,
                                   trace_z_filt,
                                   save_dir):
    """
    Plot unfiltered vs bandpass-filtered accelerometer signals (X, Y, Z) in 3x2 format.

    Args:
        event_id (str/int): Identifier for the event (used in filename).
        time (list of datetime): Time axis for the traces.
        trace_x/y/z (np.array): Raw accelerometer signals.
        trace_x/y/z_filt (np.array): Filtered accelerometer signals.
        save_dir (str): Directory to save the output plot.
    """
    save_dir = create_subfolder(save_dir, "sig_acc_raw_and_processed")
    labels = ["X", "Y", "Z"]
    traces_raw = [trace_x, trace_y, trace_z]
    traces_filt = [trace_x_filt, trace_y_filt, trace_z_filt]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8), sharex=True)

    for i in range(3):
        axes[i, 0].plot(time, traces_raw[i], alpha=0.8)
        axes[i, 0].set_ylabel(f"Velocity {labels[i]} (mm/s)")
        axes[i, 0].set_title("Unfiltered")
        axes[i, 0].grid(True)

        axes[i, 1].plot(time, traces_filt[i], alpha=0.8)
        axes[i, 1].set_title("Filtered (1-100 Hz)")
        axes[i, 1].grid(True)

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    fig.suptitle(f"Accelerometer Traces (Unfiltered vs Filtered) - Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"event_{event_id}_accel_filtered_comparison.png"
    fig.savefig(os.path.join(save_dir, filename))
    plt.close(fig)

    # print(f"Saved filtered comparison plot to: {os.path.join(save_dir, filename)}")


def plot_sig_acc_fo_align(
        event_id,
        time_axis,
        trace_x,
        trace_y,
        trace_z,
        fo_aligned,
        save_dir
):
    """
    Plot normalized aligned FO signal overlaid on the 3 accelerometer traces.

    Args:
        event_id (str or int): Event name or ID for filenames.
        time_axis (list): Time axis (same for all signals).
        trace_x/y/z (np.array): Accelerometer traces.
        fo_aligned (np.array): Aligned FO signal.
        save_dir (str): Directory to save the PNG plot.
    """
    save_dir = create_subfolder(save_dir, "sig_acc_fo_align")

    # Normalize all signals (zero mean, unit variance)
    def normalize(signal):
        return (signal - np.mean(signal)) / np.std(signal)

    trace_x_norm = normalize(trace_x)
    trace_y_norm = normalize(trace_y)
    trace_z_norm = normalize(trace_z)
    fo_norm = normalize(fo_aligned)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
    labels = ["X", "Y", "Z"]
    traces = [trace_x_norm, trace_y_norm, trace_z_norm]

    for i in range(3):
        axes[i].plot(time_axis, traces[i], label=f"Accel {labels[i]}", alpha=0.8)
        axes[i].plot(time_axis, fo_norm, label="FO (aligned, norm)", alpha=0.6)
        axes[i].set_ylabel(f"Velocity {labels[i]} / FO")
        axes[i].legend()
        axes[i].grid(True)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Normalized FO Overlay on Accelerometer Traces - Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"event_{event_id}_fo_overlay_on_accel.png"
    fig.savefig(os.path.join(save_dir, filename), format="png")
    plt.close(fig)

    # print(f"Saved normalized overlay plot to: {os.path.join(save_dir, filename)}")


import os
import matplotlib.pyplot as plt


def plot_cosine_sim_boxplot(sim_x, sim_y, sim_z, save_dir="."):
    """
    Create and save a boxplot comparing cosine similarity distributions for X/Y/Z axes.

    Args:
        sim_x, sim_y, sim_z (list): Lists of cosine similarity scores.
        save_dir (str): Directory to save the figure.
        filename (str): Filename to save as.
    """
    save_dir = create_subfolder(save_dir, "cosine_sim_boxplot")
    data = [sim_x, sim_y, sim_z]
    labels = ['Velocity X (mm/s)', 'Velocity Y (mm/s)', 'Velocity Z (mm/s)']

    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=labels, notch=True, patch_artist=True)
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Windowed Cosine Similarity to FO Signal')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    filename = "cosine_similarity_boxplot.png"

    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    # print(f"Saved boxplot to: {plot_path}")


import matplotlib.pyplot as plt
import os
import numpy as np


def plot_psd_summary(frequencies,
                     psds_x,
                     psds_y,
                     psds_z,
                     psds_fo,
                     save_dir):
    """
    Create subplots (1 per trace) showing mean ± std PSD across all events.
    Plots blank axes with 'No data' for missing inputs.

    Args:
        frequencies (np.array): Frequency bins (same for all).
        psds_x/y/z/fo (list of np.array or None): PSDs from each event.
        save_dir (str): Where to save the figure.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    save_dir = create_subfolder(save_dir, "psd_summary")

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 10), sharex=True)
    psd_sets = [psds_x, psds_y, psds_z, psds_fo]
    labels = ["Velocity X (mm/s)", "Velocity Y (mm/s)", "Velocity Z (mm/s)", "FO strain (ε)"]

    for i, (ax, psd_list, label) in enumerate(zip(axes, psd_sets, labels)):
        # Filter out None entries if any
        if psd_list is None:
            psd_list = []
        psd_list = [p for p in psd_list if p is not None]

        if len(psd_list) > 0:
            try:
                all_psds = np.vstack(psd_list)
                mean_psd = np.mean(all_psds, axis=0)
                std_psd = np.std(all_psds, axis=0)

                ax.plot(frequencies, mean_psd, label="Mean PSD")
                ax.fill_between(frequencies, mean_psd - std_psd, mean_psd + std_psd,
                                alpha=0.3, label="±1 STD")
                ax.legend()
            except Exception as e:
                ax.text(0.5, 0.5, f"Error plotting\n{str(e)}", transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='red')
        else:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')

        ax.set_ylabel("PSD")
        ax.set_title(label)
        ax.grid(True)
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel("Frequency [Hz]")
    fig.suptitle("PSD Summary Across All Events")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = "aggregated_psd_subplots.png"
    output_path = os.path.join(save_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fo_psd_ch_compare(
        event_id: str,
        timestamps: list,
        super_raw_data: np.ndarray,
        sampling_frequency: int,
        center_channel: int,
        first_channel: int,
        last_channel: int,
        window_size: int,
        save_dir: str,
        step: int = 5,
        freq_range: tuple = (0, 100)
):
    """
    Plot FO time signal and PSD using TimeSignalProcessing for every Nth channel from center.

    Args:
        event_id (str): Event ID for filename.
        timestamps (list of datetime): Time axis.
        super_raw_data (np.ndarray): Raw FO data, shape (T, C).
        sampling_frequency (int): Sampling frequency of FO.
        center_channel (int): Physical center channel number.
        first_channel (int): First physical channel number.
        last_channel (int): Last physical channel number.
        window_size (int): Window size for PSD.
        save_dir (str): Folder to save figure.
        step (int): Step size between channels (default 5).
        freq_range (tuple): Frequency axis limits for PSD.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows

    save_dir = create_subfolder(save_dir, f"fo_grid_signal_psd_{step}m")

    total_channels = last_channel - first_channel + 1
    center_idx = center_channel - first_channel

    # Select symmetrical channels around center every `step`
    offsets = []
    i = 0
    while True:
        up = center_idx + i * step
        down = center_idx - i * step
        if up < total_channels:
            offsets.append(up)
        if down >= 0 and down != up:
            offsets.append(down)
        i += 1
        if len(offsets) >= 10:
            break

    offsets = sorted(offsets)
    n_rows = len(offsets)

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 2.5 * n_rows), sharex=False)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, ch_idx in enumerate(offsets):
        ch_number = ch_idx + first_channel
        raw_signal = super_raw_data[:, ch_idx]

        # Create TimeSignalProcessing object and compute PSD
        signal = TimeSignalProcessing(
            time=timestamps,
            signal=raw_signal,
            Fs=sampling_frequency,
            window=Windows.HAMMING,
            window_size=window_size
        )

        signal.filter(Fpass=[10, 100], N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)

        from scipy.signal import iirfilter, zpk2sos, sosfilt, windows

        # timewindowtukey = windows.tukey(M=signal.signal.shape[0], alpha=0.25)
        timewindowtukey = windows.tukey(M=len(signal.signal), alpha=0.5)

        # plot the tukey for a quick visualization but clear it or close it after each to avoind messing with the other plots
        # plt.figure()
        # plt.plot(timewindowtukey)
        # plt.title(f"Tukey Window for Channel {ch_number}")
        # plt.show()
        # plt.close()

        signal.signal = signal.signal * timewindowtukey

        signal.psd()

        # crop the signal to the same length as the timestamps
        if len(signal.signal) > len(timestamps):
            signal.signal = signal.signal[:len(timestamps)]
        elif len(signal.signal) < len(timestamps):
            signal.signal = np.pad(signal.signal, (0, len(timestamps) - len(signal.signal)), 'edge')

        # Determine color: red for center channel, blue for others
        color_sig = "tab:red" if ch_number == center_channel else "tab:blue"
        color_psd = "tab:red" if ch_number == center_channel else "tab:blue"

        # Left: signal in time
        ax_sig = axes[row_idx, 0]
        ax_sig.plot(timestamps, signal.signal, color=color_sig, lw=0.8)
        ax_sig.set_title(f"FO Signal - Channel {ch_number}")
        ax_sig.set_ylabel("Strain (ε)")
        ax_sig.grid(True)

        # Right: PSD
        ax_psd = axes[row_idx, 1]
        ax_psd.plot(signal.frequency_Pxx, signal.Pxx, color=color_psd, lw=0.8)
        ax_psd.set_xlim(*freq_range)
        ax_psd.set_title(f"PSD - Channel {ch_number}")
        ax_psd.set_ylabel("Power")
        ax_psd.grid(True)

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"FO Signals and PSDs around Center Channel {center_channel} — Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"event_{event_id}_fo_signal_psd_grid.png"
    fig.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close(fig)


def plot_sig_fft_acc_fo(event_id,
                        save_dir,
                        trace_x,
                        trace_y,
                        trace_z,
                        accel_time,
                        fo_trace,
                        fo_time,
                        fo_channel,
                        first_channel,
                        fs_accel=1000,
                        fs_fo=1000,
                        freq_range=(0, 100),
                        fo_for_crop=None,
                        ):
    """
    Plot accelerometer (X, Y, Z) and FO signals with their PSDs in a 4x2 format.
    Optionally saves an interactive Plotly version as HTML.

    Args:
        event_id (str/int): Identifier for the event (used in filename).
        accel_time (list of datetime): Time axis for the accelerometer.
        trace_x/y/z (np.array): Accelerometer signals.
        fo_time (list of datetime): Time axis for the FO signal.
        fo_trace (np.array): FO signal array (2D: time x channels).
        fs_accel (int): Accelerometer sampling frequency.
        fs_fo (int): FO sampling frequency.
        save_dir (str): Directory to save the output plot.
        freq_range (tuple): Frequency range for PSDs.
        save_interactive (bool): If True, also save an interactive Plotly version.
        len_w (int): Length of the PSD window.
    """
    trace_x.reset()
    trace_y.reset()
    trace_z.reset()
    fo_trace.reset()
    trace_x.filter([1, 100], 4, type_filter="bandpass")
    trace_y.filter([1, 100], 4, type_filter="bandpass")
    trace_z.filter([1, 100], 4, type_filter="bandpass")
    fo_trace.filter(Fpass=[10, 100], N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
    trace_x.fft(half_representation=True)
    trace_y.fft(half_representation=True)
    trace_z.fft(half_representation=True)
    fo_trace.fft(half_representation=True)

    fft_x = trace_x.amplitude
    fft_y = trace_y.amplitude
    fft_z = trace_z.amplitude
    fft_fo = fo_trace.amplitude
    freq = trace_x.frequency

    ch_index = fo_channel - first_channel

    traces = [trace_x.signal[:len(fo_for_crop)], trace_y.signal[:len(fo_for_crop)], trace_z.signal[:len(fo_for_crop)],
              fo_trace.signal]
    fft_list = [fft_x, fft_y, fft_z, fft_fo]

    labels = ["X", "Y", "Z", "FO"]
    time_axes = [accel_time] * 3 + [fo_time]

    original_save_dir = save_dir

    save_dir = create_subfolder(original_save_dir, f"sig_fft_accel_fo")

    # ---------- Matplotlib PNG Plot ----------
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 10), sharex='col')

    for i in range(4):
        # Time-domain plot (left)
        axes[i, 0].plot(time_axes[i], traces[i], alpha=0.8)
        axes[i, 0].set_ylabel(f"Velocity {labels[i]} (mm/s)" if labels[i] != "FO" else "FO strain (ε)")
        axes[i, 0].grid(True)

        # FFT (right)
        axes[i, 1].plot(freq, fft_list[i], alpha=0.8)
        axes[i, 1].set_ylabel(f"FFT {labels[i]}")
        axes[i, 1].set_xlim(freq_range)
        axes[i, 1].grid(True)

    axes[3, 0].set_xlabel("Time [s]")
    axes[3, 1].set_xlabel("Frequency [Hz]")

    fig.suptitle(f"Accelerometer & FO Signals with FFT - Event {event_id}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"event_{event_id}_accel_fo_with_psd.png"
    fig.savefig(os.path.join(save_dir, filename))
    plt.close(fig)
