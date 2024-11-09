import numpy as np
from scipy.signal import hilbert, stft
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import find_peaks, hilbert, welch
from scipy.fft import fft, fftfreq

def compute_fft_features(signal):
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    peak_idx = np.argmax(magnitude)
    center_frequency = peak_idx
    peak_power = 20 * np.log10(magnitude[peak_idx])
    avg_power = 20 * np.log10(np.mean(magnitude))
    std_dev_power = 20 * np.log10(np.std(magnitude))
    return center_frequency, peak_power, avg_power, std_dev_power


def compute_instantaneous_features(signal):
    analytic_signal = hilbert(np.real(signal))
    instantaneous_amplitude = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase)
    instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode="edge")
    return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency


def autocorrelation(signal):
    result = np.correlate(signal, signal, mode="full")
    return result[result.size // 2:]


def is_digital_signal(autocorr_signal, threshold=0.1):
    normalized_autocorr = autocorr_signal / np.max(autocorr_signal)
    return 1 if np.any(normalized_autocorr < threshold) else 0


def compute_kurtosis(signal):
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    return np.mean((signal - mean_signal) ** 4) / (std_signal ** 4)


def compute_skewness(signal):
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    return np.mean((signal - mean_signal) ** 3) / (std_signal ** 3)


def compute_spectral_energy_concentration(signal, center_freq_idx, bandwidth):
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    lower_bound = max(0, center_freq_idx - bandwidth // 2)
    upper_bound = min(len(magnitude), center_freq_idx + bandwidth // 2)
    spectral_energy = np.sum(magnitude[lower_bound:upper_bound] ** 2)
    total_energy = np.sum(magnitude ** 2)
    return spectral_energy / (total_energy + 1e-10)


def compute_zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)


def compute_instantaneous_frequency_jitter(instantaneous_frequency):
    return np.std(instantaneous_frequency)


def augment_data_progressive(X, current_epoch, total_epochs, augmentation_params=None):
    if augmentation_params is None:
        augmentation_params = {
            "noise_level": 0.001,
            "scale_range": (0.99, 1.01),
            "shift_range": (-0.01, 0.01),
            "augment_percent": 0.5,
        }

    noise_level, scale_range, shift_range, augment_percent = (
        augmentation_params["noise_level"],
        augmentation_params["scale_range"],
        augmentation_params["shift_range"],
        augmentation_params["augment_percent"],
    )

    scale_factor = 1 - (current_epoch / total_epochs)
    noise_level *= scale_factor
    scale_range = (
        1 + (scale_range[0] - 1) * scale_factor,
        1 + (scale_range[1] - 1) * scale_factor,
    )
    shift_range = (shift_range[0] * scale_factor, shift_range[1] * scale_factor)

    num_samples = X.shape[0]
    num_to_augment = int(num_samples * augment_percent * scale_factor)
    indices_to_augment = np.random.choice(num_samples, num_to_augment, replace=False)

    noise = np.random.normal(0, noise_level, (num_to_augment, X.shape[1], X.shape[2]))
    scale = np.random.uniform(
        scale_range[0], scale_range[1], (num_to_augment, X.shape[1], X.shape[2])
    )
    shift = np.random.uniform(
        shift_range[0], shift_range[1], (num_to_augment, X.shape[1], X.shape[2])
    )

    X[indices_to_augment] = X[indices_to_augment] * scale + noise + shift
    print(f"Data augmented progressively for {num_to_augment} samples.")
    return X


def cyclical_lr(epoch, base_lr=1e-6, max_lr=1e-3, step_size=10):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    return base_lr + (max_lr - base_lr) * max(0, (1 - x))


def compute_spectral_kurtosis(signal, fs=1.0, nperseg=128):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    power_spectral_density = np.abs(Zxx) ** 2
    psd_mean = np.mean(power_spectral_density, axis=1)
    psd_variance = np.var(power_spectral_density, axis=1)
    return psd_variance / (psd_mean ** 2) - 1


def compute_higher_order_cumulants(signal, order=4):
    if order == 2:
        return np.var(signal)
    elif order == 3:
        return skew(signal)
    elif order == 4:
        return kurtosis(signal, fisher=False)
    raise ValueError("Only cumulants up to order 4 are supported.")


def compute_spectral_flatness(signal):
    magnitude = np.abs(np.fft.fft(signal))
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude + 1e-10)
    return geometric_mean / arithmetic_mean


def compute_instantaneous_envelope_mean(signal):
    envelope = np.abs(hilbert(signal))
    return np.mean(envelope)


def compute_variance_of_phase(signal):
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    return np.var(instantaneous_phase)


def compute_crest_factor(signal):
    peak_amplitude = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal ** 2))
    return peak_amplitude / (rms_value + 1e-10)


def compute_spectral_entropy(signal, num_bins=128):
    magnitude = np.abs(np.fft.fft(signal, n=num_bins))
    power_spectrum = magnitude ** 2
    power_spectrum /= np.sum(power_spectrum)
    return -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))


def compute_energy_spread(signal, center_freq_idx, bandwidth):
    fft_result = np.abs(np.fft.fft(signal))
    lower_bound = max(0, center_freq_idx - bandwidth // 2)
    upper_bound = min(len(fft_result), center_freq_idx + bandwidth // 2)
    spectral_energy = np.sum(fft_result[lower_bound:upper_bound] ** 2)
    total_energy = np.sum(fft_result ** 2)
    return spectral_energy / (total_energy + 1e-10)


def compute_autocorrelation_decay(signal):
    autocorr = np.correlate(signal, signal, mode="full")[len(signal)-1:]
    normalized_autocorr = autocorr / (autocorr[0] + 1e-10)
    decay_rate = -np.gradient(np.log(normalized_autocorr + 1e-10))
    return np.mean(decay_rate[1:])


def compute_rms_of_instantaneous_frequency(signal):
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    instantaneous_frequency = np.diff(instantaneous_phase)
    return np.sqrt(np.mean(instantaneous_frequency ** 2))


def compute_entropy_of_instantaneous_frequency(signal):
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    instantaneous_frequency = np.diff(instantaneous_phase)
    hist, _ = np.histogram(instantaneous_frequency, bins=64, density=True)
    hist += 1e-10
    return -np.sum(hist * np.log2(hist))


def compute_spectral_asymmetry(signal):
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    midpoint = len(magnitude) // 2
    lower_half_energy = np.sum(magnitude[:midpoint] ** 2)
    upper_half_energy = np.sum(magnitude[midpoint:] ** 2)
    return (upper_half_energy - lower_half_energy) / (upper_half_energy + lower_half_energy + 1e-10)


def compute_envelope_variance(signal):
    envelope = np.abs(signal)
    return np.var(envelope)


def compute_papr(signal):
    power = np.abs(signal) ** 2
    peak_power = np.max(power)
    average_power = np.mean(power)
    return peak_power / (average_power + 1e-10)


def compute_modulation_index(signal):
    envelope = np.abs(signal)
    peak_amplitude = np.max(envelope)
    average_amplitude = np.mean(envelope)
    return (peak_amplitude - average_amplitude) / average_amplitude if average_amplitude != 0 else 0

from scipy.signal import hilbert
import numpy as np

# Instantaneous Amplitude, Phase, and Frequency
def compute_instantaneous_features(signal):
    analytic_signal = hilbert(signal)
    amplitude = np.abs(analytic_signal)
    phase = np.unwrap(np.angle(analytic_signal))
    frequency = np.diff(phase) / (2.0 * np.pi)
    frequency = np.pad(frequency, (0, 1), mode="edge")  # To match original signal length
    return amplitude, phase, frequency

# Modulation Index
def compute_modulation_index(signal):
    amplitude, _ = compute_instantaneous_features(signal)[:2]
    peak_amplitude = np.max(amplitude)
    average_amplitude = np.mean(amplitude)
    return (peak_amplitude - average_amplitude) / (average_amplitude + 1e-10)

# Spectral Asymmetry (for AM-SSB)
def compute_spectral_asymmetry(signal):
    magnitude_spectrum = np.abs(np.fft.fft(signal))
    midpoint = len(magnitude_spectrum) // 2
    lower_half_energy = np.sum(magnitude_spectrum[:midpoint] ** 2)
    upper_half_energy = np.sum(magnitude_spectrum[midpoint:] ** 2)
    return (upper_half_energy - lower_half_energy) / (upper_half_energy + lower_half_energy + 1e-10)

# Instantaneous Frequency Deviation (already defined if using for WBFM)
def instantaneous_frequency_deviation(signal):
    analytic_signal = hilbert(signal)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi)
    return np.std(inst_freq)

# Spectral Entropy (already defined)
def spectral_entropy(signal, fs=1.0):
    from scipy.signal import welch
    from scipy.stats import entropy
    freqs, power_spectrum = welch(signal, fs=fs)
    power_spectrum /= np.sum(power_spectrum)
    return entropy(power_spectrum)

# Envelope Mean and Variance (already defined)
def envelope_mean_variance(signal):
    amplitude = np.abs(hilbert(signal))
    return np.mean(amplitude), np.var(amplitude)

# Spectral Flatness (already defined)
def spectral_flatness(signal):
    magnitude = np.abs(np.fft.fft(signal))
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude)
    return geometric_mean / (arithmetic_mean + 1e-10)

# Spectral Peaks and Bandwidth (already defined)
def spectral_peaks_bandwidth(signal, threshold_ratio=0.5):
    magnitude_spectrum = np.abs(np.fft.fft(signal))
    max_magnitude = np.max(magnitude_spectrum)
    threshold = threshold_ratio * max_magnitude
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_count = len(peaks)
    bandwidth_indices = np.where(magnitude_spectrum >= threshold)[0]
    if len(bandwidth_indices) == 0:
        return peak_count, 0.0  # No bandwidth if no peaks above threshold
    lower_freq, upper_freq = min(bandwidth_indices), max(bandwidth_indices)
    bandwidth = upper_freq - lower_freq
    return peak_count, bandwidth

# Zero Crossing Rate (already defined)
def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

def instantaneous_frequency_deviation(signal):
    """
    Calculates the standard deviation of instantaneous frequency
    derived from the phase of the analytic signal.
    """
    # Unwrap the phase to obtain the instantaneous phase
    inst_phase = np.unwrap(np.angle(signal))
    # Calculate the instantaneous frequency as the derivative of phase
    inst_freq = np.diff(inst_phase) / (2.0 * np.pi)
    return np.std(inst_freq)



def spectral_entropy(signal, fs=1.0):
    freqs, power_spectrum = welch(signal, fs=fs)
    power_spectrum /= np.sum(power_spectrum)
    return entropy(power_spectrum)

def envelope_mean_variance(signal):
    envelope = np.abs(hilbert(signal))
    return np.mean(envelope), np.var(envelope)

def spectral_flatness(signal):
    magnitude_spectrum = np.abs(fft(signal))
    geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10)))
    arithmetic_mean = np.mean(magnitude_spectrum)
    return geometric_mean / arithmetic_mean

def spectral_peaks_bandwidth(signal, threshold_ratio=0.5):
    magnitude_spectrum = np.abs(fft(signal))
    freqs = fftfreq(len(magnitude_spectrum))
    
    # Bandwidth calculation
    max_magnitude = np.max(magnitude_spectrum)
    threshold = threshold_ratio * max_magnitude
    bandwidth_indices = np.where(magnitude_spectrum >= threshold)[0]
    low_freq, high_freq = freqs[bandwidth_indices[0]], freqs[bandwidth_indices[-1]]
    bandwidth = high_freq - low_freq
    
    # Peaks calculation
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_freqs = freqs[peaks]
    return len(peak_freqs), bandwidth

def zero_crossing_rate(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)