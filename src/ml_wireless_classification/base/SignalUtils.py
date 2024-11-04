import numpy as np
from scipy.signal import hilbert, stft
from scipy.stats import kurtosis, skew


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
    return result[result.size // 2 :]


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
    kurtosis = np.mean((signal - mean_signal) ** 4) / (std_signal**4)
    return kurtosis


def compute_skewness(signal):
    """
    Compute the skewness of a signal.
    Skewness is a measure of the asymmetry of the probability distribution of a real-valued signal.
    """
    mean_signal = np.mean(signal)
    std_signal = np.std(signal)
    skewness = np.mean((signal - mean_signal) ** 3) / (std_signal**3)
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
    spectral_energy = np.sum(magnitude[lower_bound:upper_bound] ** 2)
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
    lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
    print(f"Learning rate for epoch {epoch+1}: {lr}")
    return lr


def compute_spectral_kurtosis(signal, fs=1.0, nperseg=128):
    """
    Computes the Spectral Kurtosis of a given signal.

    Parameters:
    - signal (np.array): The input signal.
    - fs (float): Sampling frequency of the signal.
    - nperseg (int): Number of samples per segment for STFT.

    Returns:
    - spectral_kurtosis (np.array): The Spectral Kurtosis values across frequencies.
    """
    # Compute the Short-Time Fourier Transform (STFT) of the signal
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)

    # Calculate the power spectral density
    power_spectral_density = np.abs(Zxx) ** 2

    # Calculate the mean and variance of the power spectral density across time
    psd_mean = np.mean(power_spectral_density, axis=1)
    psd_variance = np.var(power_spectral_density, axis=1)

    # Compute Spectral Kurtosis
    spectral_kurtosis = psd_variance / (psd_mean**2) - 1

    return spectral_kurtosis


def compute_higher_order_cumulants(signal, order=4):
    """
    Computes the higher-order cumulants (up to fourth-order by default) of the input signal.

    Parameters:
    - signal (np.array): The input signal (real or complex).
    - order (int): The order of the cumulant (default is 4 for kurtosis).

    Returns:
    - cumulant (float): The cumulant value for the given order.
    """
    if order == 2:
        # Second-order cumulant is simply the variance
        cumulant = np.var(signal)
    elif order == 3:
        # Third-order cumulant is skewness
        cumulant = skew(signal)
    elif order == 4:
        # Fourth-order cumulant is kurtosis (excess kurtosis)
        cumulant = kurtosis(signal, fisher=False)  # Using population kurtosis
    else:
        raise ValueError("Currently, only cumulants up to order 4 are supported.")
    cumulant = np.nan_to_num(cumulant, nan=0.0, posinf=0.0, neginf=0.0)
    return cumulant


import numpy as np
from scipy.signal import hilbert


def compute_spectral_flatness(signal):
    """
    Compute the spectral flatness of a signal.
    Spectral flatness is a measure of the noise-like characteristics of a signal.
    A flat spectrum (high spectral flatness) indicates noise, while low flatness indicates a tonal signal.

    :param signal: The input signal (1D numpy array)
    :return: Spectral flatness value (float)
    """
    magnitude = np.abs(np.fft.fft(signal))
    geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
    arithmetic_mean = np.mean(magnitude + 1e-10)
    spectral_flatness = geometric_mean / arithmetic_mean
    return spectral_flatness


def compute_instantaneous_envelope_mean(signal):
    """
    Compute the mean of the instantaneous envelope of a signal.
    The envelope represents the instantaneous amplitude of the signal, and its mean is useful for AM signals.

    :param signal: The input signal (1D numpy array)
    :return: Instantaneous envelope mean (float)
    """
    envelope = np.abs(hilbert(signal))
    envelope_mean = np.mean(envelope)
    return envelope_mean


def compute_variance_of_phase(signal):
    """
    Compute the variance of the instantaneous phase of a signal.
    Variance in phase can help differentiate signals with high or low phase stability.

    :param signal: The input signal (1D numpy array)
    :return: Variance of instantaneous phase (float)
    """
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    phase_variance = np.var(instantaneous_phase)
    return phase_variance


def compute_crest_factor(signal):
    """
    Compute the crest factor of a signal.
    Crest factor is the ratio of the peak amplitude to the RMS value of the signal,
    and it indicates the signal's amplitude variability.

    :param signal: The input signal (1D numpy array)
    :return: Crest factor (float)
    """
    peak_amplitude = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))
    crest_factor = peak_amplitude / (rms_value + 1e-10)
    return crest_factor


def compute_spectral_entropy(signal, num_bins=128):
    """
    Compute the spectral entropy of a signal.
    Spectral entropy measures the randomness in the spectrum, helping distinguish structured from random signals.

    :param signal: The input signal (1D numpy array)
    :param num_bins: Number of bins for the FFT (default 128)
    :return: Spectral entropy value (float)
    """
    magnitude = np.abs(np.fft.fft(signal))
    power_spectrum = magnitude**2
    power_spectrum /= np.sum(power_spectrum)
    spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-10))
    return spectral_entropy


def compute_energy_spread(signal, center_freq_idx, bandwidth):
    """
    Compute the energy spread around the center frequency in a specified bandwidth.
    This measures energy concentration around a frequency and is useful for understanding signal bandwidth.

    :param signal: The input IQ signal (1D numpy array)
    :param center_freq_idx: Center frequency index in FFT bins (int)
    :param bandwidth: Bandwidth in FFT bins around the center frequency (int)
    :return: Energy spread value (float)
    """
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    lower_bound = max(0, center_freq_idx - bandwidth // 2)
    upper_bound = min(len(magnitude), center_freq_idx + bandwidth // 2)
    energy_concentration = np.sum(magnitude[lower_bound:upper_bound] ** 2)
    total_energy = np.sum(magnitude**2)
    energy_spread = energy_concentration / (total_energy + 1e-10)
    return energy_spread


def compute_autocorrelation_decay(signal):
    """
    Compute the decay rate of the autocorrelation of a signal.
    Autocorrelation decay rate helps identify periodicity and persistence in the signal.

    :param signal: The input signal (1D numpy array)
    :return: Decay rate of autocorrelation (float)
    """
    autocorr = np.correlate(signal, signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    normalized_autocorr = autocorr / (autocorr[0] + 1e-10)
    decay_rate = -np.gradient(np.log(normalized_autocorr + 1e-10))
    return np.mean(decay_rate[1:])


def compute_rms_of_instantaneous_frequency(signal):
    """
    Compute the RMS (Root Mean Square) of the instantaneous frequency of a signal.
    This feature captures the phase stability of a signal.

    :param signal: The input signal (1D numpy array)
    :return: RMS of instantaneous frequency (float)
    """
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    instantaneous_frequency = np.diff(instantaneous_phase)
    rms_instantaneous_frequency = np.sqrt(np.mean(instantaneous_frequency**2))
    return rms_instantaneous_frequency


def compute_entropy_of_instantaneous_frequency(signal):
    """
    Compute the entropy of the instantaneous frequency of a signal.
    Instantaneous frequency entropy quantifies the complexity of phase modulation.

    :param signal: The input signal (1D numpy array)
    :return: Entropy of instantaneous frequency (float)
    """
    instantaneous_phase = np.unwrap(np.angle(hilbert(signal)))
    instantaneous_frequency = np.diff(instantaneous_phase)
    hist, _ = np.histogram(instantaneous_frequency, bins=64, density=True)
    hist += 1e-10  # Avoid log of zero
    entropy_instantaneous_frequency = -np.sum(hist * np.log2(hist))
    return entropy_instantaneous_frequency


def compute_spectral_asymmetry(signal):
    """
    Compute the spectral asymmetry of a signal.
    Spectral asymmetry distinguishes signals with uneven power distribution across the frequency spectrum.

    :param signal: The input signal (1D numpy array)
    :return: Spectral asymmetry value (float)
    """
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    midpoint = len(magnitude) // 2
    lower_half_energy = np.sum(magnitude[:midpoint] ** 2)
    upper_half_energy = np.sum(magnitude[midpoint:] ** 2)
    spectral_asymmetry = (upper_half_energy - lower_half_energy) / (
        upper_half_energy + lower_half_energy + 1e-10
    )
    return spectral_asymmetry


def compute_zero_crossing_rate(signal):
    """
    Compute the zero-crossing rate of a signal.
    Zero-crossing rate indicates the rate at which the signal changes sign, useful for understanding signal structure.

    :param signal: The input signal (1D numpy array)
    :return: Zero-crossing rate (float)
    """
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    zero_crossing_rate = len(zero_crossings) / len(signal)
    return zero_crossing_rate


def compute_envelope_variance(iq_signal):
    """
    Compute the variance of the envelope (magnitude) of the IQ signal.
    AM signals exhibit significant envelope variance due to amplitude modulation.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - envelope_variance (float): The variance of the envelope of the signal.
    """
    envelope = np.abs(iq_signal)
    envelope_variance = np.var(envelope)
    return envelope_variance

def compute_instantaneous_frequency_variance(iq_signal):
    """
    Compute the variance of the instantaneous frequency, which highlights frequency-modulated signals.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - instantaneous_frequency_variance (float): The variance of instantaneous frequency.
    """
    instantaneous_phase = np.unwrap(np.angle(iq_signal))
    instantaneous_frequency = np.diff(instantaneous_phase)
    instantaneous_frequency_variance = np.var(instantaneous_frequency)
    return instantaneous_frequency_variance

def compute_spectral_energy_concentration(iq_signal, bandwidth=10):
    """
    Calculate the concentration of spectral energy around the peak frequency.
    AM signals have concentrated energy around a central frequency, while WBFM is more spread.

    Parameters:
    - iq_signal (np.array): The input IQ signal.
    - bandwidth (int): The bandwidth around the center frequency to compute energy concentration.

    Returns:
    - energy_concentration (float): Ratio of spectral energy around the peak to total energy.
    """
    fft_result = np.fft.fft(iq_signal)
    magnitude = np.abs(fft_result)
    peak_idx = np.argmax(magnitude)
    lower_bound = max(0, peak_idx - bandwidth // 2)
    upper_bound = min(len(magnitude), peak_idx + bandwidth // 2)
    spectral_energy = np.sum(magnitude[lower_bound:upper_bound] ** 2)
    total_energy = np.sum(magnitude**2)
    energy_concentration = spectral_energy / total_energy
    return energy_concentration

def compute_modulation_index(iq_signal):
    """
    Estimate the modulation index for AM signals by calculating the ratio of peak to mean amplitude.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - modulation_index (float): The estimated modulation index.
    """
    envelope = np.abs(iq_signal)
    peak_amplitude = np.max(envelope)
    average_amplitude = np.mean(envelope)
    modulation_index = (peak_amplitude - average_amplitude) / average_amplitude if average_amplitude != 0 else 0
    return modulation_index

def compute_papr(iq_signal):
    """
    Calculate the peak-to-average power ratio, useful for distinguishing AM signals.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - papr (float): The peak-to-average power ratio.
    """
    power = np.abs(iq_signal) ** 2
    peak_power = np.max(power)
    average_power = np.mean(power)
    papr = peak_power / average_power if average_power != 0 else 0
    return papr

def compute_spectral_flatness(iq_signal):
    """
    Calculate the spectral flatness, which can help differentiate between QAM8 and QAM16.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - spectral_flatness (float): The spectral flatness measure.
    """
    fft_result = np.fft.fft(iq_signal)
    power_spectrum = np.abs(fft_result) ** 2
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))  # +1e-10 to avoid log(0)
    arithmetic_mean = np.mean(power_spectrum)
    spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean != 0 else 0
    return spectral_flatness

def compute_fourth_order_cumulant(iq_signal):
    """
    Compute the fourth-order cumulant, useful for distinguishing constellation densities.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - fourth_order_cumulant (float): The fourth-order cumulant (kurtosis).
    """
    cumulant = kurtosis(iq_signal, fisher=False)  # Using population kurtosis
    return np.nan_to_num(cumulant, nan=0.0, posinf=0.0, neginf=0.0)

def compute_phase_jitter(iq_signal):
    """
    Compute the phase jitter, the standard deviation of phase differences.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - phase_jitter (float): The standard deviation of instantaneous phase differences.
    """
    instantaneous_phase = np.unwrap(np.angle(iq_signal))
    phase_diff = np.diff(instantaneous_phase)
    phase_jitter = np.std(phase_diff)
    return phase_jitter

def compute_frequency_spread(iq_signal):
    """
    Compute the frequency spread of a signal, useful for distinguishing WBFM signals.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - frequency_spread (float): The width of the frequency spectrum.
    """
    fft_result = np.fft.fft(iq_signal)
    magnitude = np.abs(fft_result)
    threshold = 0.1 * np.max(magnitude)
    spread_indices = np.where(magnitude > threshold)[0]
    frequency_spread = spread_indices[-1] - spread_indices[0] if spread_indices.size > 0 else 0
    return frequency_spread

def compute_autocorrelation_peak_to_average_ratio(iq_signal):
    """
    Compute the peak-to-average ratio of the autocorrelation function.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - peak_to_average_ratio (float): The peak-to-average ratio of the autocorrelation.
    """
    autocorr = np.correlate(iq_signal, iq_signal, mode="full")
    autocorr_half = autocorr[autocorr.size // 2:]
    peak = np.max(autocorr_half)
    average = np.mean(autocorr_half)
    peak_to_average_ratio = peak / average if average != 0 else 0
    return peak_to_average_ratio

def compute_zero_crossing_rate(iq_signal):
    """
    Compute the zero-crossing rate, useful for differentiating FM-modulated signals.
    
    Parameters:
    - iq_signal (np.array): The input IQ signal.

    Returns:
    - zcr (float): The zero-crossing rate of the signal.
    """
    zero_crossings = np.where(np.diff(np.sign(iq_signal.real)))[0]  # Count crossings in the real part
    zcr = len(zero_crossings) / len(iq_signal)
    return zcr
