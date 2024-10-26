import numpy as np
from scipy.signal import hilbert

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
    instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode='edge')
    return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

def autocorrelation(signal):
    result = np.correlate(signal, signal, mode='full')
    return result[result.size // 2:]

def is_digital_signal(autocorr_signal, threshold=0.1):
    normalized_autocorr = autocorr_signal / np.max(autocorr_signal)
    is_digital = np.any(normalized_autocorr < threshold)
    return 1 if is_digital else 0

def compute_kurtosis(signal):
    """
    Compute the kurtosis of a signal.
    Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued signal.
    """
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    kurtosis = np.mean((signal - mean_signal)**4) / (std_signal**4)
    return kurtosis

def compute_skewness(signal):
    """
    Compute the skewness of a signal.
    Skewness is a measure of the asymmetry of the probability distribution of a real-valued signal.
    """
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    skewness = np.mean((signal - mean_signal)**3) / (std_signal**3)
    return skewness

def compute_spectral_energy_concentration(signal, center_freq_idx, bandwidth):
    """
    Compute the spectral energy concentration around the peak center frequency.
    This measures how concentrated the energy is around the peak frequency within a specified bandwidth.
    
    :param signal: The input IQ signal (real + imaginary parts)
    :param center_freq_idx: The index of the center frequency (in terms of FFT bin)
    :param bandwidth: The bandwidth (in terms of number of bins) around the center frequency
    """
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    
    # Select bins within the specified bandwidth around the center frequency
    lower_bound = max(0, center_freq_idx - bandwidth // 2)
    upper_bound = min(len(magnitude), center_freq_idx + bandwidth // 2)
    
    # Compute the energy concentration within the specified bandwidth
    spectral_energy = np.sum(magnitude[lower_bound:upper_bound]**2)
    total_energy = np.sum(magnitude**2)
    
    energy_concentration = spectral_energy / total_energy
    return energy_concentration

def compute_zero_crossing_rate(signal):
    """
    Compute the zero-crossing rate of a signal.
    Zero-crossing rate is the rate at which the signal changes sign.
    """
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zcr = len(zero_crossings) / len(signal)
    return zcr

def compute_instantaneous_frequency_jitter(instantaneous_frequency):
    """
    Compute the instantaneous frequency jitter, which is the standard deviation of instantaneous frequency.
    :param instantaneous_frequency: Array of instantaneous frequency values
    """
    freq_jitter = np.std(instantaneous_frequency)
    return freq_jitter


def augment_data_progressive(
        X, current_epoch, total_epochs, augmentation_params=None
    ):
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
        indices_to_augment = np.random.choice(
            num_samples, num_to_augment, replace=False
        )

        noise = np.random.normal(
            0, noise_level, (num_to_augment, X.shape[1], X.shape[2])
        )
        scale = np.random.uniform(
            scale_range[0], scale_range[1], (num_to_augment, X.shape[1], X.shape[2])
        )
        shift = np.random.uniform(
            shift_range[0], shift_range[1], (num_to_augment, X.shape[1], X.shape[2])
        )

        X[indices_to_augment] = X[indices_to_augment] * scale + noise + shift
        print(f"Data augmented progressively for {num_to_augment} samples.")
        return X