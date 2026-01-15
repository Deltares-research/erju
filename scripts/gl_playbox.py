import numpy as np
import matplotlib.pyplot as plt
from SignalProcessingTools.time_signal import TimeSignalProcessing, Windows


def create_vehicle_signal(time, frequency, amplitude, nb_axles, speed, distances, noise=False):
    """
    Create a time signal for a vehicle moving over a fibre optic cable.
    The vehicle is represented by a series of sinusoidal signals
    corresponding to its axles.

    Parameters:
    -----------
    time : np.ndarray
        Time vector in seconds.
    frequency : float
        Frequency of the vehicle load in Hz.
    amplitude : float
        Amplitude of the vehicle load.
    nb_axles : int
        Number of axles of the vehicle.
    speed : float
        Speed of the vehicle in m/s.
    distances : np.ndarray
        Distances along the fibre optic cable in meters.

    Returns:
    --------
    np.ndarray
        A 2D array where each column corresponds to a distance along the fibre optic cable,
        and each row corresponds to a time step.
    """

    # define 1 load cycle in time
    time_1_cycle = 1 / frequency  # Duration of one cycle in seconds
    idx = np.where((time >= 0) & (time <= nb_axles * time_1_cycle))[0]  # Adjust time to fit one cycle
    load_1_axle = np.sin(2 * np.pi * frequency * time[idx]) * amplitude
    load_1_axle[load_1_axle > 0] = 0

    signal = np.zeros((len(time), len(distances)))
    for i, x in enumerate(distances):
        t_ini = x / speed
        idx_t = np.where(time >= t_ini)[0][0]
        if idx_t + len(idx) >= len(time):
            break
        signal[idx_t:idx_t + len(idx), i] = load_1_axle

    if noise:
        noise_level = 0.01 * np.max(np.abs(signal))
        noise = np.random.normal(0, noise_level, signal.shape)
        signal += noise
    return signal


def apply_gauge_length(signal, distances, gauge_length):
    """
    Apply a gauge length to the fibre optic signal by averaging the signal over the distance.

    Parameters:
    -----------
    signal : np.ndarray
        A 2D array where each column corresponds to a distance along the fibre optic cable,
        and each row corresponds to a time step.
    distances : np.ndarray
        Distances along the fibre optic cable in meters.
    distance_point : float
        The distance point at which to apply the gauge length.
    gauge_length : float
        The gauge length in meters over which to average the signal.

    Returns:
    --------
    np.ndarray
        The averaged signal over the specified gauge length.
    """

    dx = distances[1] - distances[0]
    half_window_pts = int(round((gauge_length / dx) / 2))

    # n_time, n_dist = signal.shape
    gauged_signal = np.zeros_like(signal)

    n_dist = signal.shape[1]
    for i in range(n_dist):
        i_start = max(0, i - half_window_pts)
        i_end = min(n_dist, i + half_window_pts + 1)
        gauged_signal[:, i] = np.mean(signal[:, i_start:i_end], axis=1)

    return gauged_signal


if __name__ == "__main__":

    # Fibre optic cable
    
    length_fo = 100  # Length of the fibre optic cable in meters
    dx = 1  # Spatial resolution in meters
    distances = np.arange(0, length_fo, dx)  # Distance along the fibre optic cable

    # Vehicle parameters
    load_frequency = 50  # Frequency of the vehicle load in Hz
    load_amplitude = -2.0  # Amplitude of the vehicle load
    load_nb_axles = 8  # Number of axles of the vehicle
    load_speed = 100 / 3.6  # Speed of the vehicle in m/s

    # time
    Fs = 1000  # Sampling frequency in Hz
    time = np.arange(0, length_fo / load_speed, 1 / Fs)  # Time vector based on speed and sampling frequency
    moving_load = create_vehicle_signal(time, load_frequency, load_amplitude, load_nb_axles, load_speed, distances,
                                        noise=True)

    # plt.plot(time, moving_load[:, 0])
    # plt.plot(time, moving_load[:, 50])
    # plt.plot(time, moving_load[:, 70])
    # plt.show()

    # process the fibre optic signal by averaging the signal over the distance
    gauge_length = 2

    gauged_signal = apply_gauge_length(moving_load, distances, gauge_length=gauge_length)

    idx_to_analyse = [50]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    for idx in idx_to_analyse:
        moving = TimeSignalProcessing(time, moving_load[:, idx], window=Windows.HAMMING, window_size=1024)
        moving.fft()
        moving.psd()
        gauged = TimeSignalProcessing(time, gauged_signal[:, idx], window=Windows.HAMMING, window_size=1024)
        gauged.fft()
        gauged.psd()

        ax[0].plot(time, moving_load[:, idx], label=f'Raw at {distances[idx]}m')
        ax[0].plot(time, gauged_signal[:, idx], label=f'Gauged at {distances[idx]}m', linestyle='-')

        ax[1].plot(moving.frequency, moving.Pxx[: -1], label=f'PSD Raw at {distances[idx]}m')
        ax[1].plot(gauged.frequency, gauged.Pxx[: -1], label=f'PSD Gauged at {distances[idx]}m', linestyle='-')

    ax[0].set_title('Time Domain Signal')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[1].set_title('Frequency Domain Signal')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Amplitude')
    ax[1].legend()
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(bottom=0)
    ax[0].set_xlim(left=0)
    ax[0].grid()
    ax[1].grid()
    plt.suptitle(f"Gauge length {gauge_length}m", fontsize=10)
    plt.show()
    # plt.savefig(f"gauge_length_{gauge_length}m.png")
    plt.close()
