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

from src.utils.file_utils import compute_psd


def plot_accel_vs_fo(
        save_dir,
        event_id,
        accel_time,
        trace_x,
        trace_y,
        trace_z,
        fo_time,
        fo_data,
        fo_channel=1194,
        first_channel=1189,
        save_interactive=False
):
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
    os.makedirs(save_dir, exist_ok=True)
    ch_index = fo_channel - first_channel
    accel_traces = [trace_x, trace_y, trace_z]
    labels = ["Trace X", "Trace Y", "Trace Z"]

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
        axes[i, 1].set_title(f"FO Channel {fo_channel}")
        axes[i, 1].grid(True)
        if i == 2:
            axes[i, 1].set_xlabel("Time")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    png_path = os.path.join(save_dir, f"event_{event_id}_accel_vs_fo.png")
    fig.savefig(png_path, format='png')
    plt.close(fig)
    print(f"Saved PNG to:   {png_path}")

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
        print(f"Saved interactive HTML to: {html_path}")


def plot_fo_before_after(
        save_dir,
        event_id,
        timestamps,
        raw_signal_data,
        processed_data,
        fo_channel=1194,
        first_channel=1189,
        save_interactive=False
):
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
    os.makedirs(save_dir, exist_ok=True)
    ch_index = fo_channel - first_channel

    # Get two colors from the viridis colormap
    viridis = cm.get_cmap('viridis')
    color1 = mcolors.to_hex(viridis(0.2))
    color2 = mcolors.to_hex(viridis(0.8))

    # ----------- Static PNG (Matplotlib) -----------
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6), sharex=True)

    axes[0].plot(timestamps, raw_signal_data[:, ch_index], color=color1, alpha=0.85)
    axes[0].set_title(f"Event {event_id} - FO Channel {fo_channel} (Before Filtering)")
    axes[0].set_ylabel("Raw Signal")
    axes[0].grid(True)

    axes[1].plot(timestamps, processed_data[:, ch_index], color=color2, alpha=0.85)
    axes[1].set_title(f"Event {event_id} - FO Channel {fo_channel} (After Filtering + Strain)")
    axes[1].set_ylabel("Processed Signal")
    axes[1].set_xlabel("Time")
    axes[1].grid(True)

    fig.tight_layout()
    png_path = os.path.join(save_dir, f"event_{event_id}_fo_before_after.png")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    print(f"Saved static FO plot to: {png_path}")

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

        print(f"Saved interactive FO plot to: {html_path}")


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
    labels = ["Accel X", "Accel Y", "Accel Z", "FO"]

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
    print(f"Saved static PSD plot to: {png_path}")

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
        print(f"Saved interactive PSD plot to: {html_path}")


def plot_accel_signals_and_psd(
        event_id,
        accel_time,
        trace_x,
        trace_y,
        trace_z,
        fs=1000,
        save_dir=".",
        freq_range=(0, 500)
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
    os.makedirs(save_dir, exist_ok=True)
    labels = ["X", "Y", "Z"]
    traces = [trace_x, trace_y, trace_z]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 8), sharex='col')

    for i, trace in enumerate(traces):
        # Left: Time signal
        axes[i, 0].plot(accel_time, trace, alpha=0.8)
        axes[i, 0].set_ylabel(f"{labels[i]} (m/s²)")
        axes[i, 0].grid(True)

        # Right: PSD
        freq, psd = compute_psd(trace, fs=fs)
        axes[i, 1].plot(freq, psd, alpha=0.8)
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

    print(f"Saved accelerometer signal & PSD plot to: {os.path.join(save_dir, filename)}")


import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils.file_utils import compute_psd


def plot_accel_fo_with_psd(
        event_id,
        accel_time,
        trace_x,
        trace_y,
        trace_z,
        fo_time,
        fo_trace,
        fo_channel=1194,
        first_channel=1189,
        fs_accel=1000,
        fs_fo=1000,
        save_dir=".",
        freq_range=(0, 500),
        save_interactive=False
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
    """
    os.makedirs(save_dir, exist_ok=True)

    ch_index = fo_channel - first_channel
    fo_trace = fo_trace[:, ch_index]

    labels = ["X", "Y", "Z", "FO"]
    traces = [trace_x, trace_y, trace_z, fo_trace]
    time_axes = [accel_time] * 3 + [fo_time]
    sample_rates = [fs_accel] * 3 + [fs_fo]

    # ---------- Matplotlib PNG Plot ----------
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 10), sharex='col')

    for i in range(4):
        freq, psd = compute_psd(traces[i], fs=sample_rates[i])

        # Time-domain plot (left)
        axes[i, 0].plot(time_axes[i], traces[i], alpha=0.8)
        axes[i, 0].set_ylabel(f"{labels[i]} (m/s²)" if labels[i] != "FO" else "FO")
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

    print(f"Saved static accelerometer + FO signal & PSD plot to: {os.path.join(save_dir, filename)}")

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

        print(f"Saved interactive HTML plot to: {html_path}")
