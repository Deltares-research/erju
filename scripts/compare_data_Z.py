import os
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from src.utils.db_utils import fetch_accel_data, estimate_sampling_frequency
from src.erju.process_FO_base import BaseFOdata
from src.utils.file_utils import from_window_get_fo_file, align_signals, compute_cosine_similarity_windows, \
    create_results_folder
from src.utils.plot_utils import plot_cosine_sim_boxplot, plot_psd_summary
from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows, FilterDesign


def get_config():
    return {
        "db_path": r"P:/11207352-stem/database/Wielrondheid_132887.db",
        "fo_data_path": r"C:/fo_samples/holten",
        "results_path": r"P:/11210064-erju/holten",
        "start_date": '2024-09-07 06:00:00',
        "end_date": '2024-09-07 08:10:00',
        "location": ['Meetjournal_MP8_Holten_zuid_4m_C'],
        "campaigns": None,
        "traintype": "ICM",
        "track": "1",
        "first_channel": 800,
        "center_channel": 1194,
        "last_channel": 1400,
        "window_size": 1024,
        "save_plots": True,
        "save_csv": True,
        "plot_config": {
            "sig_acc_fo": True,
            "sig_fo_raw_and_processed": True,
            "sig_psd_acc": True,
            "sig_psd_acc_fo": True,
            "sig_fft_acc_fo": True,
            "sig_acc_raw_and_processed": False,
            "sig_acc_fo_align": True,
            "cosine_boxplot": True,
            "psd_summary": True,
        }
    }


def fetch_and_prepare_accel_data(config):
    events, tim, _ = fetch_accel_data(
        db_path=config["db_path"],
        start_date=config["start_date"],
        end_date=config["end_date"],
        locations=config["location"],
        campaigns=config["campaigns"],
        traintype=config["traintype"],
        track=config["track"],
        get_timeseries=True
    )
    accel_records = []
    event_series = list(tim.values())[0]

    for event, (event_id, data) in zip(events, event_series.items()):
        start_time = datetime.strptime(event[2], "%Y-%m-%d %H:%M:%S")
        absolute_time = [start_time + timedelta(seconds=t) for t in data['TIME']]
        time_window = (absolute_time[0], absolute_time[-1])

        record = {
            'event_id': event_id,
            'absolute_time': absolute_time,
            'time_window': time_window
        }
        for axis in ['X', 'Y', 'Z']:
            if f'TRACE_{axis}' in data:
                trace = data[f'TRACE_{axis}']
                freq = estimate_sampling_frequency(absolute_time)
                ts = TimeSignalProcessing(
                    absolute_time, trace, Fs=freq,
                    window=Windows.HAMMING,
                    window_size=config["window_size"]
                )
                ts.filter([1, 100], 4, type_filter="bandpass")
                record[f'trace_{axis.lower()}'] = ts
        accel_records.append(record)
    return accel_records


def fetch_and_prepare_fo_data(config, time_window):
    fo = BaseFOdata.create_instance(
        dir_path=config['fo_data_path'],
        first_channel=config['first_channel'],
        last_channel=config['last_channel'],
        reader='optasense'
    )

    files = from_window_get_fo_file(config['fo_data_path'], list(time_window))

    data_list = []
    raw_list = []

    for i, file in enumerate(files):
        if i == 0:
            fo.extract_properties_per_file(file)
            start_time = fo.properties['FileStartTime']
            fs = int(fo.properties['SamplingFrequency[Hz]'])

        d_proc, d_raw = fo.extract_data(file, config['first_channel'], config['last_channel'])
        data_list.append(d_proc.T)
        raw_list.append(d_raw.T)

    data_all = np.concatenate(data_list, axis=0)
    raw_all = np.concatenate(raw_list, axis=0)

    timestamps = [start_time + timedelta(seconds=i / fs) for i in range(data_all.shape[0])]
    t_arr = np.array(timestamps, dtype='datetime64[ns]')

    start_idx = np.argmin(np.abs(t_arr - np.datetime64(time_window[0])))
    end_idx = np.argmin(np.abs(t_arr - np.datetime64(time_window[1]))) + 1

    return {
        "timestamps": timestamps[start_idx:end_idx],
        "data": data_all[start_idx:end_idx, :],
        "raw": raw_all[start_idx:end_idx, :],
        "fs": fs
    }


def analyze_event(accel, fo_data, config, results_folder):
    ch_idx = config['center_channel'] - config['first_channel']
    fo_signal = fo_data['raw'][:, ch_idx]
    ts = TimeSignalProcessing(
        fo_data['timestamps'],
        fo_signal,
        Fs=fo_data['fs'],
        window=Windows.HAMMING,
        window_size=config['window_size']
    )
    ts.filter([10, 100], N=5, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)

    results = {
        'event_id': accel['event_id'],
        'fo': ts,
        'similarity': {},
        'psd': {}
    }

    fo_aligned, lag = align_signals(accel['trace_z'].signal, ts.signal)

    for axis in ['x', 'y', 'z']:
        if f'trace_{axis}' in accel:
            acc = accel[f'trace_{axis}']
            acc.psd(nb_points=10000)
            results['psd'][axis] = acc.Pxx
            results['similarity'][axis] = compute_cosine_similarity_windows(acc.signal[:len(fo_aligned)], fo_aligned)
    ts.psd(nb_points=10000)
    results['psd']['fo'] = ts.Pxx
    results['freqs'] = ts.frequency_Pxx
    return results


def main():
    config = get_config()
    results_folder = create_results_folder(
        base_path=config['results_path'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        traintype=config['traintype'],
        center_channel=config['center_channel'],
        track=config['track']
    )

    start = time.time()
    logger.info("Fetching accelerometer data...")
    accel_data = fetch_and_prepare_accel_data(config)
    logger.info(f"Found {len(accel_data)} events")

    sim_scores = {'x': [], 'y': [], 'z': []}
    psd_all = {'x': [], 'y': [], 'z': [], 'fo': []}
    freqs_shared = None

    for record in accel_data:
        logger.info(f"Processing event {record['event_id']}")
        fo = fetch_and_prepare_fo_data(config, record['time_window'])
        result = analyze_event(record, fo, config, results_folder)

        for axis in sim_scores:
            if axis in result['similarity']:
                sim_scores[axis].append(result['similarity'][axis])
                psd_all[axis].append(result['psd'][axis])
        psd_all['fo'].append(result['psd']['fo'])

        if freqs_shared is None:
            freqs_shared = result['freqs']

    if config['save_plots']:
        if config['plot_config']['cosine_boxplot']:
            plot_cosine_sim_boxplot(sim_scores['x'], sim_scores['y'], sim_scores['z'], save_dir=results_folder)
        if config['plot_config']['psd_summary']:
            plot_psd_summary(freqs_shared,
                             psds_x=psd_all['x'],
                             psds_y=psd_all['y'],
                             psds_z=psd_all['z'],
                             psds_fo=psd_all['fo'],
                             save_dir=results_folder)

    duration = time.time() - start
    logger.success(f"Finished processing all events in {duration:.2f} seconds ({duration / 60:.2f} min)")


if __name__ == "__main__":
    main()
