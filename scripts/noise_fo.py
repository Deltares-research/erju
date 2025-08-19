"""
This script processes FO data to extract noise signals, computes their Power Spectral Density (PSD),
and visualizes the results. It handles multiple files, extracts a specified number of signals, and saves the
results in a structured manner. It chooses random segments from the data files, applies a bandpass filter,
and plots both the time-domain signals and their PSDs. The results are saved in a specified directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import timedelta
from loguru import logger

from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows, FilterDesign
from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import get_files_in_dir

# === Parameters =======================================================================================================
path_fo = r"E:\recording_2024-08-29T08_01_16Z_5kHzping_1kHzlog_1mCS_10mGL_3000channels"
save_path = r"N:\Projects\11210000\11210064\B. Measurements and calculations\holten\noise"
os.makedirs(save_path, exist_ok=True)

first_channel = 1994
last_channel = 1994
selected_channel = 1994
duration_sec = 60  # Duration of signal to extract in seconds
n_total = 100  # Total signals to process
batch_size = 5  # Number of signals per plot
window_size = 1024

# === Setup FO Reader ==================================================================================================
fo = BaseFOdata.create_instance(dir_path=path_fo, first_channel=first_channel,
                                last_channel=last_channel, reader='optasense')

fo_files = get_files_in_dir(path_fo, ".h5")
logger.info(f"Found {len(fo_files)} FO files.")

# === Storage for PSDs =================================================================================================
all_psds = []
freqs_reference = None
signal_counter = 0
batch_counter = 0

while signal_counter < n_total:
    fig, axes = plt.subplots(nrows=batch_size, ncols=2, figsize=(12, 2.5 * batch_size), sharex=False)
    row = 0

    while row < batch_size and signal_counter < n_total:
        random_file = random.choice(fo_files)
        logger.info(f"[{signal_counter + 1}/{n_total}] Using FO file: {random_file}")

        try:
            fo.extract_properties_per_file(random_file)
            sampling_frequency = int(fo.properties['SamplingFrequency[Hz]'])
            file_start_time = fo.properties['FileStartTime']

            _, raw_signal_data = fo.extract_data(file_name=random_file,
                                                 first_channel=first_channel,
                                                 last_channel=last_channel)
            raw_signal_data = raw_signal_data.T

            total_samples = raw_signal_data.shape[0]
            samples_needed = sampling_frequency * duration_sec
            if total_samples < samples_needed:
                logger.warning(f"Skipping file {random_file}, too short.")
                continue

            start_idx = random.randint(0, total_samples - samples_needed)
            end_idx = start_idx + samples_needed

            slice_data = raw_signal_data[start_idx:end_idx, :]
            timestamps = [file_start_time + timedelta(seconds=j / sampling_frequency)
                          for j in range(start_idx, end_idx)]

            ch_idx = selected_channel - first_channel

            # Create TimeSignalProcessing object and process the raw FO signal
            ts_obj = TimeSignalProcessing(
                time=timestamps,
                signal=slice_data[:, ch_idx],
                Fs=sampling_frequency,
                window=Windows.HAMMING,
                window_size=window_size
            )
            # Apply bandpass filter
            ts_obj.filter(Fpass=[10, 100], N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
            # Compute Power Spectral Density (PSD)
            ts_obj.psd()

            # Store PSD
            all_psds.append(ts_obj.Pxx)
            if freqs_reference is None:
                freqs_reference = ts_obj.frequency_Pxx

            # Align signal to timestamps
            if len(ts_obj.signal) > len(timestamps):
                ts_obj.signal = ts_obj.signal[:len(timestamps)]
            elif len(ts_obj.signal) < len(timestamps):
                ts_obj.signal = np.pad(ts_obj.signal, (0, len(timestamps) - len(ts_obj.signal)), mode='edge')

            # Plot signal
            axes[row, 0].plot(timestamps, ts_obj.signal)
            axes[row, 0].set_title(f"Signal {signal_counter + 1}")
            axes[row, 0].set_ylabel("Strain")
            axes[row, 0].grid(True)

            # Plot PSD
            axes[row, 1].plot(ts_obj.frequency_Pxx, ts_obj.Pxx)
            axes[row, 1].set_title("PSD")
            axes[row, 1].set_xlim(0, 100)
            axes[row, 1].set_ylabel("Power")
            axes[row, 1].grid(True)

            row += 1
            signal_counter += 1

        except Exception as e:
            logger.warning(f"Skipping file due to error: {e}")
            continue

    # Label axes and save
    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"FO Noise – Channel {selected_channel} (Signals {signal_counter - row + 1} to {signal_counter})",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(save_path, f"fo_noise_batch_{batch_counter + 1:02}.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    batch_counter += 1

# === Final Summary Plot ===
all_psds = np.array(all_psds)
mean_psd = np.mean(all_psds, axis=0)
std_psd = np.std(all_psds, axis=0)

plt.figure(figsize=(10, 5))
plt.plot(freqs_reference, mean_psd, label="Mean PSD", color="blue")
plt.fill_between(freqs_reference, mean_psd - std_psd, mean_psd + std_psd,
                 color="blue", alpha=0.3, label="±1 std dev")
plt.title(f"Average PSD over {len(all_psds)} FO Signals – Channel {selected_channel}")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 100)
plt.grid(True)
plt.legend()
plt.tight_layout()

summary_plot = os.path.join(save_path, f"fo_noise_channel_{selected_channel}_avg_psd.png")
plt.savefig(summary_plot, dpi=300)
plt.close()

print(f"All signal batches saved to: {save_path}")
print(f"Average PSD plot saved to: {summary_plot}")
