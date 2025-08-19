import numpy as np
import matplotlib.pyplot as plt

# =============================
# PLAYBOX PARAMETERS (edit these)
# =============================
fiber_length_m = 1000  # total fiber length (m)
channel_spacing_m = 1  # spatial sampling (1 m => 1 channel per meter)
gauge_length_measured_m = 50  # gauge length used in acquisition
gauge_length_target_m = 1  # hypothetical (desired) gauge length to reconstruct

train_speed_m_s = 30  # train speed (m/s) ~108 km/h
num_bogies = 6  # number of bogies
bogie_spacing_m = 18.0  # distance between bogie centers (m)
axle_spacing_within_bogie_m = 2.5  # distance between the two axles in a bogie (m)

axle_gaussian_sigma_m = 1.0  # spatial width of each axle response (Gaussian sigma)
axle_amplitude = 1.0  # amplitude per axle

sleeper_spacing_m = 0.6  # sleeper spacing (m)
sleeper_modulation_amp = 0.2  # amplitude of periodic modulation due to sleepers (0 = off)

noise_std_fraction = 0.05  # noise level relative to max(|true_strain|)
regularization_alpha = 0.05  # Tikhonov regularization parameter
time_step_s = 0.2  # temporal sampling

# =============================
# DERIVED SETUP
# =============================
positions = np.arange(0, fiber_length_m, channel_spacing_m)
n_positions = positions.size

# Axle relative positions: two axles per bogie at +/- axle_spacing/2
bogie_centers = np.arange(num_bogies) * bogie_spacing_m
axle_offsets = [-axle_spacing_within_bogie_m / 2, axle_spacing_within_bogie_m / 2]
axle_relative_positions = np.array([c + o for c in bogie_centers for o in axle_offsets])
train_length_m = axle_relative_positions.max()

# Time until last axle leaves the fiber
t_end = (fiber_length_m + train_length_m) / train_speed_m_s
times = np.arange(0, t_end, time_step_s)
n_times = times.size


# =============================
# UTILITY FUNCTIONS
# =============================
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_true_strain_field():
    """Underlying strain before gauge averaging."""
    strain_true = np.zeros((n_times, n_positions))
    for ti, t in enumerate(times):
        x_front = train_speed_m_s * t  # position of first axle
        axle_positions = x_front + axle_relative_positions

        field = np.zeros(n_positions)
        for ax_pos in axle_positions:
            if -10 < ax_pos < fiber_length_m + 10:
                field += axle_amplitude * gaussian(positions, ax_pos, axle_gaussian_sigma_m)

        # sleeper modulation (periodic)
        modulation = 1 + sleeper_modulation_amp * np.sin(2 * np.pi * positions / sleeper_spacing_m)
        strain_true[ti] = field * modulation
    return strain_true


def boxcar_kernel(gauge_length_m):
    n = int(round(gauge_length_m / channel_spacing_m))
    n = max(n, 1)
    return np.ones(n) / n


def apply_gauge_length(strain_true, gauge_length_m):
    k = boxcar_kernel(gauge_length_m)
    out = np.array([np.convolve(row, k, mode='same') for row in strain_true])
    return out


def add_noise(signal, frac):
    max_abs = np.max(np.abs(signal))
    noise = np.random.normal(0, frac * max_abs, size=signal.shape)
    return signal + noise


def deconvolve_tikhonov(y, kernel, alpha):
    """
    Tikhonov deconvolution in spatial frequency domain.
    y: (n_times, n_positions)
    kernel: 1D averaging kernel
    alpha: regularization parameter
    """
    n = y.shape[1]
    k_full = np.zeros(n)
    k_full[:kernel.size] = kernel
    # center the kernel for FFT
    k_full = np.roll(k_full, -kernel.size // 2)
    K = np.fft.rfft(k_full)

    estimated = np.zeros_like(y)
    for ti in range(y.shape[0]):
        Y = np.fft.rfft(y[ti])
        H_conj = np.conj(K)
        denom = (np.abs(K) ** 2 + alpha)
        E_hat = H_conj * Y / denom
        estimated[ti] = np.fft.irfft(E_hat, n=n)
    return estimated


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


# =============================
# SIMULATION
# =============================
np.random.seed(42)
strain_true = build_true_strain_field()

measurement_10m = apply_gauge_length(strain_true, gauge_length_measured_m)
measurement_10m_noisy = add_noise(measurement_10m, noise_std_fraction)

# Deconvolve to estimate underlying strain
kernel_10m = boxcar_kernel(gauge_length_measured_m)
strain_estimated = deconvolve_tikhonov(measurement_10m_noisy, kernel_10m, regularization_alpha)

# Predict 2 m gauge measurement from estimated strain
measurement_2m_predicted = apply_gauge_length(strain_estimated, gauge_length_target_m)

# True 2 m gauge (for comparison)
measurement_2m_true = apply_gauge_length(strain_true, gauge_length_target_m)

# =============================
# PLOTS (single time snapshot)
# =============================
mid_index = n_times // 2

plt.figure(figsize=(12, 6))
plt.title("Full Fiber: Spatial Strain Profiles (time index = mid)")
plt.plot(positions, strain_true[mid_index], label="True Strain (Underlying)")
plt.plot(positions, measurement_10m_noisy[mid_index], label="Measured 10 m Gauge (noisy)")
plt.plot(positions, measurement_2m_true[mid_index], label="True 2 m Gauge")
plt.plot(positions, measurement_2m_predicted[mid_index], '--', label="Predicted 2 m Gauge (from deconvolution)")
plt.xlabel("Position (m)")
plt.ylabel("Strain (arb. units)")
plt.legend()
plt.tight_layout()

# Zoomed 200 m window around center
window_center = fiber_length_m / 2
window_half_width = 100
mask = (positions >= window_center - window_half_width) & (positions <= window_center + window_half_width)
x_subset = positions[mask]

plt.figure(figsize=(12, 6))
plt.title("Zoomed 200 m Window")
plt.plot(x_subset, strain_true[mid_index, mask], label="True Strain (Underlying)")
plt.plot(x_subset, measurement_10m_noisy[mid_index, mask], label="Measured 10 m Gauge (noisy)")
plt.plot(x_subset, measurement_2m_true[mid_index, mask], label="True 2 m Gauge")
plt.xlabel("Position (m)")
plt.ylabel("Strain (arb. units)")
plt.legend()
plt.tight_layout()

# Difference plot
plt.figure(figsize=(12, 4))
plt.title("Difference: True 2 m Gauge - Measured 10 m Gauge (Zoomed)")
plt.plot(x_subset, measurement_2m_true[mid_index, mask] - measurement_10m_noisy[mid_index, mask])
plt.xlabel("Position (m)")
plt.ylabel("Strain Difference")
plt.tight_layout()

plt.show()

# =============================
# METRICS
# =============================
print(f"RMSE underlying strain (deconvolved vs true): {rmse(strain_estimated, strain_true):.4f}")
print(f"RMSE 2 m gauge (predicted vs true): {rmse(measurement_2m_predicted, measurement_2m_true)::.4f}")

print("\nAdjust 'regularization_alpha', 'sleeper_modulation_amp', etc., to see how smoothing and recovery change.")
