import pickle
import numpy as np
import matplotlib.pyplot as plt
from SignalProcessingTools.time_signal import TimeSignalProcessing
from scipy.signal import medfilt


def wiener_deconvolution(signal, kernel, lambd):
    "lambd is the SNR"
    kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel))))  # zero pad the kernel to same length
    H = np.fft.fft(kernel)
    deconvolved = np.real(np.fft.ifft(np.fft.fft(signal) * np.conj(H) / (H * np.conj(H) + lambd ** 2)))
    return deconvolved


def estimate_epsilon(signal, kernel_len=10):
    """
    Estimate noise-to-signal power ratio epsilon for Wiener deconvolution.

    Parameters:
    - signal: np.ndarray, the observed (smoothed) signal
    - kernel_len: int, window size of the smoothing kernel (used for median filter)

    Returns:
    - epsilon: float, estimated regularization parameter
    """
    # Estimate "clean" signal using median filter
    smooth_est = medfilt(signal, kernel_size=kernel_len + 1)

    # Residual is treated as noise
    noise_est = signal - smooth_est

    signal_power = np.var(smooth_est)
    noise_power = np.var(noise_est)

    # Avoid division by zero
    if signal_power == 0:
        return 1e-3

    epsilon = noise_power / signal_power
    return epsilon


if __name__ == "__main__":
    with open(
            r"P:\11210064-erju\holten\res-20240827_20240828-SPRA-ch_1194-dir_1\processed_data_event_20240828_112307.mat.pickle",
            "rb") as f:
        data = pickle.load(f)
        signal_fo = data['trace_fo'][250:-100]
        time_datetime = data['time'][250:-100]
        signal_geophone = data['trace_x'][0][250:-100]
        start_time = time_datetime[0]
        time = [(dt - start_time).total_seconds() for dt in time_datetime]

    N = 10  # number of samples for moving average
    kernel = np.ones(N) / N  # Moving average kernel

    epsilon = estimate_epsilon(signal_fo, kernel_len=N)
    ttt = wiener_deconvolution(signal_fo, kernel, epsilon)

    fo = TimeSignalProcessing(np.array(time), signal_fo)
    fo.fft()

    dec_fo = TimeSignalProcessing(np.array(time), ttt)
    dec_fo.fft()

    geophone = TimeSignalProcessing(np.array(time), signal_geophone)
    geophone.fft()

    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    ax[0, 0].plot(time, ttt, label='FO Deconvolved', linestyle='-', alpha=0.7)
    ax[0, 0].plot(time, signal_fo, label='FO Original', alpha=0.7)
    ax[0, 0].set_title('Time Domain Signal')
    ax[0, 0].set_xlabel('Time (s)')
    ax[0, 0].set_ylabel('Amplitude')
    ax[0, 0].grid()
    ax[0, 0].legend()

    ax[0, 1].plot(dec_fo.frequency, dec_fo.amplitude, label='FFT FO Deconvolved', linestyle='-', alpha=0.7)
    ax[0, 1].plot(fo.frequency, fo.amplitude, label='FFT FO Original', alpha=0.7)
    ax[0, 1].set_title('Frequency Domain Signal')
    ax[0, 1].set_xlabel('Frequency (Hz)')
    ax[0, 1].set_ylabel('Amplitude')
    ax[0, 1].legend()
    ax[0, 1].set_xlim(0, 100)
    ax[0, 1].set_ylim(bottom=0)
    ax[0, 1].grid()

    ax[1, 0].plot(time, signal_geophone, label='Geophone')
    ax[1, 0].set_title('Time Domain Signal')
    ax[1, 0].set_xlabel('Time (s)')
    ax[1, 0].set_ylabel('Amplitude')
    ax[1, 0].grid()
    ax[1, 0].legend()

    ax[1, 1].plot(geophone.frequency, geophone.amplitude, label='FFT Geophone')
    ax[1, 1].set_title('Frequency Domain Signal')
    ax[1, 1].set_xlabel('Frequency (Hz)')
    ax[1, 1].set_ylabel('Amplitude')
    ax[1, 1].legend()
    ax[1, 1].set_xlim(0, 100)
    ax[1, 1].set_ylim(bottom=0)
    ax[1, 1].grid()
    plt.tight_layout()
    plt.show()
    plt.close()
