import numpy as np
from scipy.signal import hilbert, stft
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import find_peaks, hilbert, welch
from scipy.fft import fft, fftfreq
# Feature Extraction Helper Functions
from scipy.signal import butter, filtfilt
import pickle
from scipy.signal import hilbert, welch
from scipy.ndimage import gaussian_filter1d
import pywt

ALL_FEATURES = [
    # Instantaneous Frequency Features
    "Inst. Freq. Dev (Magnitude)", "Inst. Freq. Dev (Phase)",
    "Inst. Freq. Dev Std (Magnitude)", "Inst. Freq. Dev Std (Phase)",
    "Inst. Freq. Rate of Change",
    "Inst. Freq. Asymmetry (Cubic)",

    # Phase Features
    "Inst. Phase Deviation Rate (Magnitude)", "Inst. Phase Deviation Rate (Phase)",
    "Phase Variance",
    "Phase Change Rate (Quartic) (Magnitude)", "Phase Change Rate (Quartic) (Phase)",
    "Skewness of Phase Changes (Cubic) (Magnitude)", "Skewness of Phase Changes (Cubic) (Phase)",
    "Phase Modulation Skewness (Quartic) (Magnitude)", "Phase Modulation Skewness (Quartic) (Phase)",
    "Phase-Envelope Correlation (Polynomial)",

    # Amplitude Features
    "Avg Symbol Power",
    "PAPR",
    "RMS of Signal Envelope",
    "Instantaneous Amplitude Asymmetry",
    "RMS of Signal Envelope (Polynomial)",
    "Inst. Amplitude Asymmetry (Cubic)",
    "Band-Pass Filtered RMS of Signal Envelope",

    # Statistical Features
    "Kurtosis Magnitude",
    "Skewness Magnitude",
    "PSD Kurtosis (Magnitude)", "PSD Kurtosis (Phase)",
    "Autocorrelation Skewness (Quartic) (Magnitude)", "Autocorrelation Skewness (Quartic) (Phase)",
    "Autocorrelation Energy Spread (Magnitude)", "Autocorrelation Energy Spread (Phase)",
    "Fifth Order Cumulant (Magnitude)", "Fifth Order Cumulant (Phase)",

    # Spectral Features
    "Spectral Energy Density (Magnitude)", "Spectral Energy Density (Phase)",
    "Spectral Peak Ratio (Magnitude)", "Spectral Peak Ratio (Phase)",
    "Amplitude Spectral Flatness (Magnitude)", "Amplitude Spectral Flatness (Phase)",
    "High-Frequency Spectral Entropy (Cubic)",
    "Low-Frequency Spectral Flatness (Quartic)",
    "Spectral Concentration Around Center (Magnitude)", "Spectral Concentration Around Center (Phase)",
    "Frequency Spread Log (Cubic) (Magnitude)", "Frequency Spread Log (Cubic) (Phase)",
    "Spectral Modulation Bandwidth (Quadratic)",

    # Entropy Features
    "Frequency Domain Entropy with High-Frequency Emphasis",
    "Wavelet Entropy Multiple Scales (Quadratic)",

    # Temporal Features
    "Temporal Peak Density (Quadratic)",
    "Interquartile Range of Envelope Peaks (Cubic)",
    "Time-Frequency Energy Concentration (Magnitude)", "Time-Frequency Energy Concentration (Phase)",
    "Energy Spread Time-Frequency (Gaussian) (Magnitude)", "Energy Spread Time-Frequency (Gaussian) (Phase)",
    "Energy Spread Time-Frequency",
    "Temporal Energy Variance (Gaussian)",

    # Frequency Features
    "Frequency Modulation Rate",
    "Frequency Spread Variability (Magnitude)", "Frequency Spread Variability (Phase)",

    # Wavelet Features
    "Wavelet Transform (Cubic Kernel) (Magnitude)", "Wavelet Transform (Cubic Kernel) (Phase)",
    "High-Frequency Wavelet Coefficient Mean (Quartic)",
    "Wavelet Energy Concentration (High Frequency)",

    # Spread Features
    "Frequency Spread Log (Cubic) (Magnitude)", "Frequency Spread Log (Cubic) (Phase)",
    "Normalized High-Frequency Power Ratio",

    # Envelope Features
    "Envelope Power Variability in Frequency Bands",

    # Adaptive Features
    "Adaptive Bandwidth Concentration (Gaussian) (Magnitude)", "Adaptive Bandwidth Concentration (Gaussian) (Phase)",
    "Adaptive Gaussian Filtering (Frequency Domain)",

    # Energy Concentration Features
    "Mid-Band Energy Concentration (Polynomial) (Magnitude)", "Mid-Band Energy Concentration (Polynomial) (Phase)",
    
     # Spectral Flatness
    "Spectral Flatness (Magnitude)",
    "Spectral Flatness (Phase)",
    
    # Spectral Asymmetry
    "Spectral Asymmetry (Magnitude)",
    "Spectral Asymmetry (Phase)",
    
    # Frequency Spread Log (Cubic)
    "Frequency Spread Log Cubic (Magnitude)",
    "Frequency Spread Log Cubic (Phase)",
    
    # Envelope Power Variability
    "Envelope Power Variability (Magnitude)",
    "Envelope Power Variability (Phase)",
    
    # Temporal Energy Variance (Gaussian)
    "Temporal Energy Variance Gaussian (Magnitude)",
    "Temporal Energy Variance Gaussian (Phase)",
    
    # Band-Pass Filtered RMS of Signal Envelope
    "Band-Pass Filtered RMS Envelope (Magnitude)",
    "Band-Pass Filtered RMS Envelope (Phase)",
    
    # Adaptive Gaussian Filtering (Frequency Domain)
    "Adaptive Gaussian Filtering (Magnitude)",
    "Adaptive Gaussian Filtering (Phase)",
    
    # Mid-Band Energy Concentration (Polynomial)
    "Mid-Band Energy Concentration (Magnitude)",
    "Mid-Band Energy Concentration (Phase)"
]
EXPERIMENTAL_FEATURES = [
    "Frequency Domain Entropy with High-Frequency Emphasis",
    "Frequency Domain Entropy with Mid-Frequency Emphasis",
    "Frequency Domain Entropy with Low-Frequency Emphasis",
    "Spectral Peak Count (Magnitude)",
    "Spectral Peak Average Amplitude (Magnitude)",
    "Spectral Peak Energy Ratio (Magnitude)",
    "Spectral Peak Width (Magnitude)",
    "Peak-to-Average Ratio (Spectral Peaks)",
    "Spectral Kurtosis (Peaks Only)",
    "Peak Frequency Variation (Magnitude)",
    "Total Spectral Energy in Peaks (Magnitude)",
    "Spectral Flatness in Peak Band (Magnitude)",
    "Spectral Energy Spread in Peaks (Magnitude)",
    "Maximum Peak Amplitude (Magnitude)",
    "Minimum Peak Amplitude (Magnitude)",
    "Spectral Peak Density (Peaks per Hz)",
    "Frequency Range of Spectral Peaks"
]


feature_dict = {}

def add_feature(name, func, *args):
    """Try to add a feature by checking the shape and ensuring it’s a scalar."""
    try:
        value = func(*args)
        if np.isscalar(value):
            feature_dict[name] = value
        elif isinstance(value, (list, tuple, np.ndarray)) and value.size == 1:
            feature_dict[name] = value.item()
        else:
            print(f"Warning: Feature '{name}' has incorrect shape and was not added.")
    except Exception as e:
        print(f"Error computing feature '{name}': {e}")
        

def extract_comprehensive_features(complex_signal, selected_features=None):
    # If no specific features are provided, compute all by default
    compute_all_features = selected_features is None
    real_signal = np.real(complex_signal)
    
    def add_feature_if_selected(name, func, *args):
        """Compute and add the feature if either all features are being computed or it's in the selected_features list."""
        if compute_all_features or name in selected_features:
            add_feature(name, func, *args)

    # Instantaneous Frequency Features
    add_feature_if_selected("Inst. Freq. Dev (Magnitude)", lambda x: np.abs(instantaneous_frequency_deviation(x)), complex_signal)
    add_feature_if_selected("Inst. Freq. Dev (Phase)", lambda x: np.angle(instantaneous_frequency_deviation(x)), complex_signal)
    add_feature_if_selected("Inst. Freq. Dev Std (Magnitude)", lambda x: np.abs(instantaneous_frequency_deviation_std(x)), complex_signal)
    add_feature_if_selected("Inst. Freq. Dev Std (Phase)", lambda x: np.angle(instantaneous_frequency_deviation_std(x)), complex_signal)
    add_feature_if_selected("Inst. Freq. Rate of Change", instantaneous_frequency_rate_of_change, real_signal)
    add_feature_if_selected("Inst. Freq. Asymmetry (Cubic)", lambda x: frequency_asymmetry(apply_kernel(x, kernel="cubic")), real_signal)

    # Phase Features
    add_feature_if_selected("Inst. Phase Deviation Rate (Magnitude)", lambda x: np.abs(instantaneous_phase_deviation_rate(x)), complex_signal)
    add_feature_if_selected("Inst. Phase Deviation Rate (Phase)", lambda x: np.angle(instantaneous_phase_deviation_rate(x)), complex_signal)
    add_feature_if_selected("Phase Variance", lambda x: np.var(compute_instantaneous_features(x)[1]), real_signal)
    add_feature_if_selected("Phase Change Rate (Quartic) (Magnitude)", lambda x: np.abs(phase_change_rate(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Phase Change Rate (Quartic) (Phase)", lambda x: np.angle(phase_change_rate(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Skewness of Phase Changes (Cubic) (Magnitude)", lambda x: np.abs(skewness_of_phase_changes(apply_kernel(x, kernel="cubic"))), complex_signal)
    add_feature_if_selected("Skewness of Phase Changes (Cubic) (Phase)", lambda x: np.angle(skewness_of_phase_changes(apply_kernel(x, kernel="cubic"))), complex_signal)
    add_feature_if_selected("Phase Modulation Skewness (Quartic) (Magnitude)", lambda x: np.abs(phase_modulation_skewness(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Phase Modulation Skewness (Quartic) (Phase)", lambda x: np.angle(phase_modulation_skewness(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Phase-Envelope Correlation (Polynomial)", lambda x: correlation(phase_envelope(x), apply_kernel(x, kernel="polynomial")), real_signal)

    # Amplitude Features
    add_feature_if_selected("Avg Symbol Power", lambda x: np.mean(np.abs(x)**2), complex_signal)
    add_feature_if_selected("PAPR", lambda x: np.max(np.abs(x)**2) / np.mean(np.abs(x)**2), complex_signal)
    add_feature_if_selected("RMS of Signal Envelope", rms_signal_envelope, real_signal)
    add_feature_if_selected("Instantaneous Amplitude Asymmetry", amplitude_asymmetry, real_signal)
    add_feature_if_selected("RMS of Signal Envelope (Polynomial)", lambda x: rms_signal_envelope(apply_kernel(x, kernel="polynomial")), real_signal)
    add_feature_if_selected("Inst. Amplitude Asymmetry (Cubic)", lambda x: amplitude_asymmetry(apply_kernel(x, kernel="cubic")), real_signal)
    add_feature_if_selected("Band-Pass Filtered RMS of Signal Envelope", band_pass_filtered_rms, real_signal)

    # Statistical Features
    add_feature_if_selected("Kurtosis Magnitude", lambda x: compute_kurtosis(np.abs(x)), complex_signal)
    add_feature_if_selected("Skewness Magnitude", lambda x: compute_skewness(np.abs(x)), complex_signal)
    add_feature_if_selected("PSD Kurtosis (Magnitude)", lambda x: np.abs(kurtosis(np.abs(np.fft.fft(x))**2)), complex_signal)
    add_feature_if_selected("PSD Kurtosis (Phase)", lambda x: np.angle(kurtosis(np.abs(np.fft.fft(x))**2)), complex_signal)
    add_feature_if_selected("Autocorrelation Skewness (Quartic) (Magnitude)", lambda x: np.abs(autocorrelation_skewness(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Autocorrelation Skewness (Quartic) (Phase)", lambda x: np.angle(autocorrelation_skewness(apply_kernel(x, kernel="quartic"))), complex_signal)
    add_feature_if_selected("Autocorrelation Energy Spread (Magnitude)", lambda x: np.abs(autocorrelation_energy_spread(x)), complex_signal)
    add_feature_if_selected("Autocorrelation Energy Spread (Phase)", lambda x: np.angle(autocorrelation_energy_spread(x)), complex_signal)
    add_feature_if_selected("Fifth Order Cumulant (Magnitude)", lambda x: np.abs(fifth_order_cumulant(x)), complex_signal)
    add_feature_if_selected("Fifth Order Cumulant (Phase)", lambda x: np.angle(fifth_order_cumulant(x)), complex_signal)

    # Spectral Features
    add_feature_if_selected("Spectral Energy Density (Magnitude)", lambda x: np.abs(spectral_energy_density(x)), complex_signal)
    add_feature_if_selected("Spectral Energy Density (Phase)", lambda x: np.angle(spectral_energy_density(x)), complex_signal)
    add_feature_if_selected("Spectral Peak Ratio (Magnitude)", lambda x: np.abs(spectral_peak_ratio(x)), complex_signal)
    add_feature_if_selected("Spectral Peak Ratio (Phase)", lambda x: np.angle(spectral_peak_ratio(x)), complex_signal)
    add_feature_if_selected("Amplitude Spectral Flatness (Magnitude)", lambda x: np.abs(amplitude_spectral_flatness(x)), complex_signal)
    add_feature_if_selected("Amplitude Spectral Flatness (Phase)", lambda x: np.angle(amplitude_spectral_flatness(x)), complex_signal)
    add_feature_if_selected("High-Frequency Spectral Entropy (Cubic)", lambda x: spectral_entropy(apply_kernel(x, kernel="cubic", band="high")), complex_signal)
    add_feature_if_selected("Low-Frequency Spectral Flatness (Quartic)", lambda x: spectral_flatness(apply_kernel(x, kernel="quartic", band="low")), complex_signal)
    add_feature_if_selected("Spectral Concentration Around Center (Magnitude)", lambda x: np.abs(spectral_concentration_center(x)), complex_signal)
    add_feature_if_selected("Spectral Concentration Around Center (Phase)", lambda x: np.angle(spectral_concentration_center(x)), complex_signal)
    add_feature_if_selected("Frequency Spread Log (Cubic) (Magnitude)", lambda x: np.abs(np.log(frequency_spread(apply_kernel(x, kernel="cubic")))), complex_signal)
    add_feature_if_selected("Frequency Spread Log (Cubic) (Phase)", lambda x: np.angle(np.log(frequency_spread(apply_kernel(x, kernel="cubic")))), complex_signal)
    add_feature_if_selected("Spectral Modulation Bandwidth (Quadratic)", lambda x: spectral_modulation_bandwidth(apply_kernel(x, kernel="quadratic")), complex_signal)

    # Entropy Features
    add_feature_if_selected("Frequency Domain Entropy with High-Frequency Emphasis", high_freq_emphasis_entropy, complex_signal)
    add_feature_if_selected("Wavelet Entropy Multiple Scales (Quadratic)", lambda x: wavelet_entropy(x, kernel="quadratic"), complex_signal)

    # Temporal Features
    add_feature_if_selected("Temporal Peak Density (Quadratic)", lambda x: peak_density(apply_kernel(x, kernel="quadratic")), complex_signal)
    add_feature_if_selected("Interquartile Range of Envelope Peaks (Cubic)", lambda x: interquartile_range(envelope_peaks(apply_kernel(x, kernel="cubic"))), real_signal)
    add_feature_if_selected("Time-Frequency Energy Concentration (Magnitude)", lambda x: np.abs(time_frequency_energy_concentration(x)), complex_signal)
    add_feature_if_selected("Time-Frequency Energy Concentration (Phase)", lambda x: np.angle(time_frequency_energy_concentration(x)), complex_signal)
    add_feature_if_selected("Energy Spread Time-Frequency (Gaussian) (Magnitude)", lambda x: np.abs(energy_spread_time_frequency(apply_kernel(x, kernel="gaussian"))), complex_signal)
    add_feature_if_selected("Energy Spread Time-Frequency (Gaussian) (Phase)", lambda x: np.angle(energy_spread_time_frequency(apply_kernel(x, kernel="gaussian"))), complex_signal)
    add_feature_if_selected("Temporal Energy Variance (Gaussian)", temporal_energy_variance_gaussian, real_signal)

    # Frequency Features
    add_feature_if_selected("Frequency Modulation Rate", frequency_modulation_rate, real_signal)
    add_feature_if_selected("Frequency Spread Variability (Magnitude)", lambda x: np.abs(frequency_spread_variability(x)), complex_signal)
    add_feature_if_selected("Frequency Spread Variability (Phase)", lambda x: np.angle(frequency_spread_variability(x)), complex_signal)

    # Wavelet Features
    add_feature_if_selected("Wavelet Transform (Cubic Kernel) (Magnitude)", lambda x: np.abs(wavelet_high_order_kernel(x, kernel="cubic")), real_signal)
    add_feature_if_selected("Wavelet Transform (Cubic Kernel) (Phase)", lambda x: np.angle(wavelet_high_order_kernel(x, kernel="cubic")), real_signal)
    add_feature_if_selected("High-Frequency Wavelet Coefficient Mean (Quartic)", lambda x: wavelet_coefficient_mean(x, freq_band="high", kernel="quartic"), complex_signal)
    add_feature_if_selected("Wavelet Energy Concentration (High Frequency)", high_freq_wavelet_energy_concentration, complex_signal)

    # Spread Features
    add_feature_if_selected("Frequency Spread Log (Cubic) (Magnitude)", lambda x: np.abs(frequency_spread_log_cubic(x)), complex_signal)
    add_feature_if_selected("Frequency Spread Log (Cubic) (Phase)", lambda x: np.angle(frequency_spread_log_cubic(x)), complex_signal)
    add_feature_if_selected("Normalized High-Frequency Power Ratio", normalized_high_freq_power_ratio, complex_signal)

    # Envelope Features
    add_feature_if_selected("Envelope Power Variability in Frequency Bands", envelope_power_variability, real_signal)

    # Adaptive Features
    add_feature_if_selected("Adaptive Bandwidth Concentration (Gaussian) (Magnitude)", lambda x: np.abs(bandwidth_concentration(apply_kernel(x, kernel="gaussian", adaptive=True))), complex_signal)
    add_feature_if_selected("Adaptive Bandwidth Concentration (Gaussian) (Phase)", lambda x: np.angle(bandwidth_concentration(apply_kernel(x, kernel="gaussian", adaptive=True))), complex_signal)
    add_feature_if_selected("Adaptive Gaussian Filtering (Frequency Domain)", adaptive_gaussian_filtering, complex_signal)

    # Energy Concentration Features
    add_feature_if_selected("Mid-Band Energy Concentration (Polynomial) (Magnitude)", lambda x: np.abs(energy_concentration(apply_kernel(x, kernel="polynomial", band="mid"))), complex_signal)
    add_feature_if_selected("Mid-Band Energy Concentration (Polynomial) (Phase)", lambda x: np.angle(energy_concentration(apply_kernel(x, kernel="polynomial", band="mid"))), complex_signal)

    # Spectral Flatness 
    add_feature_if_selected("Spectral Flatness (Magnitude)", lambda x: np.abs(spectral_flatness(x)), complex_signal)
    add_feature_if_selected("Spectral Flatness (Phase)", lambda x: np.angle(spectral_flatness(x)), complex_signal)

    # Spectral Asymmetry 
    add_feature_if_selected("Spectral Asymmetry (Magnitude)", lambda x: np.abs(compute_spectral_asymmetry(x)), complex_signal)
    add_feature_if_selected("Spectral Asymmetry (Phase)", lambda x: np.angle(compute_spectral_asymmetry(x)), complex_signal)

    # Frequency Spread Log (Cubic) 
    add_feature_if_selected("Frequency Spread Log Cubic (Magnitude)", lambda x: np.abs(frequency_spread_log_cubic(x)), complex_signal)
    add_feature_if_selected("Frequency Spread Log Cubic (Phase)", lambda x: np.angle(frequency_spread_log_cubic(x)), complex_signal)

    # Envelope Power Variability 
    add_feature_if_selected("Envelope Power Variability (Magnitude)", lambda x: np.abs(envelope_power_variability(x)), complex_signal)
    add_feature_if_selected("Envelope Power Variability (Phase)", lambda x: np.angle(envelope_power_variability(x)), complex_signal)

    # Temporal Energy Variance (Gaussian) 
    add_feature_if_selected("Temporal Energy Variance Gaussian (Magnitude)", lambda x: np.abs(temporal_energy_variance_gaussian(x)), complex_signal)
    add_feature_if_selected("Temporal Energy Variance Gaussian (Phase)", lambda x: np.angle(temporal_energy_variance_gaussian(x)), complex_signal)

    # Band-Pass Filtered RMS of Signal Envelope 
    add_feature_if_selected("Band-Pass Filtered RMS Envelope (Magnitude)", lambda x: np.abs(band_pass_filtered_rms(x)), complex_signal)
    add_feature_if_selected("Band-Pass Filtered RMS Envelope (Phase)", lambda x: np.angle(band_pass_filtered_rms(x)), complex_signal)

    # Adaptive Gaussian Filtering (Frequency Domain) 
    add_feature_if_selected("Adaptive Gaussian Filtering (Magnitude)", lambda x: np.abs(adaptive_gaussian_filtering(x)), complex_signal)
    add_feature_if_selected("Adaptive Gaussian Filtering (Phase)", lambda x: np.angle(adaptive_gaussian_filtering(x)), complex_signal)

    # Mid-Band Energy Concentration (Polynomial) 
    add_feature_if_selected("Mid-Band Energy Concentration (Magnitude)", lambda x: np.abs(energy_concentration(x)), complex_signal)
    add_feature_if_selected("Mid-Band Energy Concentration (Phase)", lambda x: np.angle(energy_concentration(x)), complex_signal)


    # Experimental WBFM Features
    add_feature_if_selected("Frequency Domain Entropy with High-Frequency Emphasis", lambda x: frequency_domain_entropy(x, low_cut=0.5, high_cut=1.0), complex_signal)
    add_feature_if_selected("Frequency Domain Entropy with Mid-Frequency Emphasis", lambda x: frequency_domain_entropy(x, low_cut=0.25, high_cut=0.5), complex_signal)
    add_feature_if_selected("Frequency Domain Entropy with Low-Frequency Emphasis", lambda x: frequency_domain_entropy(x, low_cut=0.0, high_cut=0.25), complex_signal)
    add_feature_if_selected("Spectral Peak Count (Magnitude)", spectral_peak_count, complex_signal)
    add_feature_if_selected("Spectral Peak Average Amplitude (Magnitude)", spectral_peak_average_amplitude, complex_signal)
    add_feature_if_selected("Spectral Peak Energy Ratio (Magnitude)", spectral_peak_energy_ratio, complex_signal)
    add_feature_if_selected("Spectral Peak Width (Magnitude)", spectral_peak_width, complex_signal)
    add_feature_if_selected("Peak-to-Average Ratio (Spectral Peaks)", peak_to_average_ratio_spectral, complex_signal)
    add_feature_if_selected("Spectral Kurtosis (Peaks Only)", spectral_kurtosis_peaks, complex_signal)
    add_feature_if_selected("Peak Frequency Variation (Magnitude)", peak_frequency_variation, complex_signal)
    add_feature_if_selected("Total Spectral Energy in Peaks (Magnitude)", total_spectral_energy_peaks, complex_signal)
    add_feature_if_selected("Spectral Flatness in Peak Band (Magnitude)", spectral_flatness_peak_band, complex_signal)
    add_feature_if_selected("Spectral Energy Spread in Peaks (Magnitude)", spectral_energy_spread_peaks, complex_signal)
    add_feature_if_selected("Maximum Peak Amplitude (Magnitude)", max_peak_amplitude, complex_signal)
    add_feature_if_selected("Minimum Peak Amplitude (Magnitude)", min_peak_amplitude, complex_signal)
    add_feature_if_selected("Spectral Peak Density (Peaks per Hz)", spectral_peak_density, complex_signal)
    add_feature_if_selected("Frequency Range of Spectral Peaks", frequency_range_of_peaks, complex_signal)

    return feature_dict




# 1-3. Frequency Domain Entropy with High-Frequency, Mid-Frequency, and Low-Frequency Emphasis
def frequency_domain_entropy(signal, fs=1.0, low_cut=0.0, high_cut=1.0):
    freqs, power_spectrum = welch(signal, fs=fs)
    band_mask = (freqs >= low_cut) & (freqs <= high_cut)
    power_band = power_spectrum[band_mask] / np.sum(power_spectrum[band_mask] + 1e-10)
    return entropy(power_band)

# 4. Spectral Peak Count (Magnitude)
def spectral_peak_count(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return len(peaks)

# 5. Spectral Peak Average Amplitude (Magnitude)
def spectral_peak_average_amplitude(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return np.mean(magnitude_spectrum[peaks]) if peaks.size > 0 else 0

# 6. Spectral Peak Energy Ratio (Magnitude)
def spectral_peak_energy_ratio(signal):
    magnitude_spectrum = np.abs(fft(signal))**2
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_energy = np.sum(magnitude_spectrum[peaks])
    total_energy = np.sum(magnitude_spectrum)
    return peak_energy / (total_energy + 1e-10)

# 7. Spectral Peak Width (Magnitude)
def spectral_peak_width(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, properties = find_peaks(magnitude_spectrum, width=5)
    widths = properties['widths'] if 'widths' in properties else np.array([0])
    return np.mean(widths) if widths.size > 0 else 0

# 8. Peak-to-Average Ratio (Spectral Peaks)
def peak_to_average_ratio_spectral(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_amplitude = np.max(magnitude_spectrum[peaks]) if peaks.size > 0 else 0
    average_amplitude = np.mean(magnitude_spectrum)
    return peak_amplitude / (average_amplitude + 1e-10)

# 9. Spectral Kurtosis (Peaks Only)
def spectral_kurtosis_peaks(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return np.mean((magnitude_spectrum[peaks] - np.mean(magnitude_spectrum[peaks]))**4) / (np.std(magnitude_spectrum[peaks])**4 + 1e-10) if peaks.size > 0 else 0

# 10. Peak Frequency Variation (Magnitude)
def peak_frequency_variation(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_freqs = np.diff(peaks)
    return np.var(peak_freqs) if len(peak_freqs) > 1 else 0

# 11. Total Spectral Energy in Peaks (Magnitude)
def total_spectral_energy_peaks(signal):
    magnitude_spectrum = np.abs(fft(signal))**2
    peaks, _ = find_peaks(magnitude_spectrum)
    return np.sum(magnitude_spectrum[peaks]) if peaks.size > 0 else 0

# 12. Spectral Flatness in Peak Band (Magnitude)
def spectral_flatness_peak_band(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_band = magnitude_spectrum[peaks]
    geometric_mean = np.exp(np.mean(np.log(peak_band + 1e-10)))  # Avoid log(0) by adding a small value
    arithmetic_mean = np.mean(peak_band)
    return geometric_mean / (arithmetic_mean + 1e-10) if peaks.size > 0 else 0

# 13. Spectral Energy Spread in Peaks (Magnitude)
def spectral_energy_spread_peaks(signal):
    magnitude_spectrum = np.abs(fft(signal))**2
    peaks, _ = find_peaks(magnitude_spectrum)
    peak_band_energy = magnitude_spectrum[peaks] if peaks.size > 0 else np.array([0])
    return np.std(peak_band_energy)

# 14. Maximum Peak Amplitude (Magnitude)
def max_peak_amplitude(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return np.max(magnitude_spectrum[peaks]) if peaks.size > 0 else 0

# 15. Minimum Peak Amplitude (Magnitude)
def min_peak_amplitude(signal):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return np.min(magnitude_spectrum[peaks]) if peaks.size > 0 else 0

# 16. Spectral Peak Density (Peaks per Hz)
def spectral_peak_density(signal, fs=1.0):
    magnitude_spectrum = np.abs(fft(signal))
    peaks, _ = find_peaks(magnitude_spectrum)
    return len(peaks) / fs if fs > 0 else 0

# 17. Frequency Range of Spectral Peaks
def frequency_range_of_peaks(signal, fs=1.0):
    magnitude_spectrum = np.abs(fft(signal))
    freqs = fftfreq(len(magnitude_spectrum), 1/fs)
    peaks, _ = find_peaks(magnitude_spectrum)
    if peaks.size > 0:
        min_peak_freq = freqs[peaks].min()
        max_peak_freq = freqs[peaks].max()
        return max_peak_freq - min_peak_freq
    return 0


def magnitude_phase_features(complex_signal, feature_func):
    """
    Helper function to compute the magnitude and phase of a feature derived from a complex signal.
    
    Parameters:
        complex_signal (ndarray): The complex signal.
        feature_func (function): The feature extraction function to apply to the complex signal.
    
    Returns:
        tuple: Magnitude and phase of the feature.
    """
    feature_value = feature_func(complex_signal)
    magnitude = np.abs(feature_value)
    phase = np.angle(feature_value)
    return magnitude, phase

        
def spectral_energy_density(signal):
    """Compute the spectral energy density of a signal."""
    spectrum = np.abs(fft(signal))**2
    return np.sum(spectrum) / len(spectrum)

def spectral_peak_ratio(signal):
    """Compute the spectral peak ratio of a signal."""
    spectrum = np.abs(fft(signal))
    peak_magnitude = np.max(spectrum)
    avg_magnitude = np.mean(spectrum)
    return peak_magnitude / avg_magnitude if avg_magnitude != 0 else 0

def amplitude_spectral_flatness(signal):
    """Compute the spectral flatness of the amplitude spectrum."""
    spectrum = np.abs(fft(signal))
    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))  # Avoid log(0) by adding a small value
    arithmetic_mean = np.mean(spectrum)
    return geometric_mean / arithmetic_mean if arithmetic_mean != 0 else 0

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


# Kernel application function
def apply_kernel(signal, kernel="linear", band="full", adaptive=False):
    """Applies a specified kernel to the signal."""
    if kernel == "cubic":
        return signal ** 3
    elif kernel == "quartic":
        return signal ** 4
    elif kernel == "polynomial":
        poly = np.poly1d([1, 0, -1])  # Example polynomial
        return poly(signal)
    elif kernel == "gaussian":
        sigma = 1 if not adaptive else max(0.1, min(2, np.std(signal) / np.mean(signal)))
        return gaussian_filter1d(signal, sigma=sigma)
    else:
        return signal  # Linear (default) does nothing


def apply_bandpass_filter(signal, low_cut, high_cut, fs=1.0, order=4):
    """
    Apply a Butterworth bandpass filter to a signal.
    
    Parameters:
    - signal: array-like, the input signal
    - low_cut: float, the low cutoff frequency as a fraction of the Nyquist rate
    - high_cut: float, the high cutoff frequency as a fraction of the Nyquist rate
    - fs: float, the sampling frequency of the signal (default 1.0)
    - order: int, the order of the Butterworth filter (default 4)
    
    Returns:
    - filtered_signal: array-like, the bandpass-filtered signal
    """
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

# Kernelized Frequency Domain Features
def spectral_entropy(signal):
    psd = np.abs(fft(signal)) ** 2
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log(psd_norm + 1e-10))

def spectral_flatness(signal):
    psd = np.abs(fft(signal)) ** 2
    geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
    arithmetic_mean = np.mean(psd)
    return geometric_mean / (arithmetic_mean + 1e-10)

def energy_concentration(signal, low_cut=0.1, high_cut=0.5):
    f, Pxx = welch(signal)
    band_mask = (f >= low_cut) & (f <= high_cut)
    return np.sum(Pxx[band_mask])

def bandwidth_concentration(signal):
    psd = np.abs(fft(signal)) ** 2
    smooth_psd = gaussian_filter1d(psd, sigma=1)
    return np.sum(smooth_psd)

def frequency_spread(signal):
    f, Pxx = welch(signal)
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    return np.log(np.sqrt(np.sum((f - mean_freq) ** 2 * Pxx) / np.sum(Pxx)))

# Wavelet-Based Multi-Scale Analysis
def wavelet_power_variance(signal, kernel="morlet"):
    coeffs = pywt.wavedec(signal, kernel, level=4)
    power_peaks = [np.max(np.abs(c)**2) for c in coeffs[1:]]
    return np.var(power_peaks)

def wavelet_entropy(signal, kernel="quadratic"):
    coeffs = pywt.wavedec(signal, 'db1', level=4)
    entropy_sum = np.sum([-np.sum(np.abs(c) * np.log(np.abs(c) + 1e-10)) for c in coeffs])
    return np.power(entropy_sum, 2)  # Quadratic kernel

def wavelet_coefficient_mean(signal, freq_band="high", kernel="quartic"):
    coeffs = pywt.wavedec(signal, 'db1', level=4)
    high_band = coeffs[-1]  # Assuming the highest band for simplicity
    return np.mean(high_band)

# Time-Domain Features with Higher-Order Kernels
def zero_crossing_rate(signal):
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)

def modulation_index(signal):
    envelope = np.abs(hilbert(signal))
    return np.var(envelope) / np.mean(envelope)

def phase_change_rate(signal):
    phase_diff = np.diff(np.angle(signal))
    return np.var(phase_diff)

def frequency_asymmetry(signal):
    inst_freq = np.diff(np.unwrap(np.angle(hilbert(signal))))
    return np.mean(inst_freq ** 3)

def rms_signal_envelope(signal):
    envelope = np.abs(hilbert(signal))
    return np.sqrt(np.mean(envelope ** 2))

# Higher-Order Statistical Features
def fourth_order_amplitude_moment(signal):
    centered_signal = signal - np.mean(signal)
    return np.mean(centered_signal**4)

def skewness_of_phase_changes(signal):
    phase_diff = np.diff(np.angle(signal))
    return np.power(skew(phase_diff), 3)

def cumulant(signal, order=6):
    return np.mean(signal ** order)

# Cross-Domain Features (Time-Frequency)
def frequency_variance(signal):
    inst_freq = np.diff(np.unwrap(np.angle(signal)))
    return np.var(inst_freq)

def spectral_modulation_bandwidth(signal):
    f, Pxx = welch(signal)
    return np.sqrt(np.mean(f ** 2 * Pxx) - np.mean(f * Pxx) ** 2)

# Phase and Instantaneous Amplitude-Related Features
def phase_modulation_skewness(signal):
    phase_mod = np.angle(signal)
    return np.power(skew(phase_mod), 4)

def amplitude_asymmetry(signal):
    envelope = np.abs(hilbert(signal))
    return skew(envelope)

def correlation(phase, envelope):
    return np.corrcoef(phase, envelope)[0, 1]

# Energy Spread in Time-Frequency
def energy_spread_time_frequency(signal):
    f, Pxx = welch(signal)
    return np.std(Pxx)

# Temporal peak density
def peak_density(signal):
    peaks = np.diff(np.sign(np.diff(signal))) < 0
    return np.sum(peaks) / len(signal)

# Autocorrelation Skewness
def autocorrelation_skewness(signal):
    autocorr = np.correlate(signal, signal, mode='full')
    return skew(autocorr)

# Interquartile Range (IQR) of Envelope Peaks
def interquartile_range(signal):
    envelope = np.abs(hilbert(signal))
    return np.percentile(envelope, 75) - np.percentile(envelope, 25)

# Interquartile range (IQR) of envelope peaks
def envelope_peaks(signal):
    envelope = np.abs(hilbert(signal))
    return envelope

# Correlation between phase and envelope
def phase_envelope(signal):
    return np.abs(hilbert(signal)), np.angle(signal)


# WBFM-Specific Feature Functions

def frequency_modulation_rate(signal):
    inst_phase = np.unwrap(np.angle(signal))
    inst_freq = np.diff(inst_phase)
    return np.mean(np.abs(np.diff(inst_freq)))

def instantaneous_frequency_deviation_std(signal):
    inst_phase = np.unwrap(np.angle(signal))
    inst_freq = np.diff(inst_phase)
    return np.std(inst_freq)

def high_frequency_power_ratio(signal, cutoff=0.5):
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    high_freq_power = np.sum(np.abs(fft_values[np.abs(freqs) > cutoff])**2)
    total_power = np.sum(np.abs(fft_values)**2)
    return high_freq_power / total_power if total_power > 0 else 0

def spectral_concentration_center(signal, center_freq=0):
    fft_values = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    center_band = (freqs >= center_freq - 0.1) & (freqs <= center_freq + 0.1)
    center_energy = np.sum(np.abs(fft_values[center_band])**2)
    total_energy = np.sum(np.abs(fft_values)**2)
    return center_energy / total_energy if total_energy > 0 else 0

def energy_spread_time_frequency(signal, fs=1.0):
    f, t, Zxx = stft(signal, fs=fs, nperseg=128)
    return np.std(np.abs(Zxx))

def zero_crossing_density_frequency(signal):
    fft_values = np.fft.fft(signal)
    return np.mean(np.diff(np.sign(np.real(fft_values))) != 0)

def frequency_spread_log_cubic(signal):
    f, Pxx = welch(signal)
    mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
    spread = np.log(np.sqrt(np.sum((f - mean_freq) ** 2 * Pxx) / np.sum(Pxx)) + 1e-10)
    return spread ** 3  # Cubic kernel applied

def adaptive_gaussian_filtering(signal):
    psd = np.abs(np.fft.fft(signal)) ** 2
    sigma = max(0.1, min(2, np.std(psd) / np.mean(psd)))
    filtered_psd = gaussian_filter1d(psd, sigma=sigma)
    return np.sum(filtered_psd)

def wavelet_high_order_kernel(signal, kernel="cubic"):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    if kernel == "cubic":
        return np.mean([np.power(np.abs(c), 3).mean() for c in coeffs[1:]])
    elif kernel == "quartic":
        return np.mean([np.power(np.abs(c), 4).mean() for c in coeffs[1:]])

def frequency_modulation_rate(signal, fs=1.0):
    inst_freq = np.diff(np.unwrap(np.angle(hilbert(signal))))
    return np.mean(np.abs(np.diff(inst_freq))) * fs

def high_frequency_power_ratio(signal, cutoff=0.5, fs=1.0):
    f, Pxx = welch(signal, fs=fs)
    high_freq_power = np.sum(Pxx[f > cutoff])
    total_power = np.sum(Pxx)
    return high_freq_power / total_power if total_power > 0 else 0

def zero_crossing_density_frequency(signal):
    fft_signal = np.fft.fft(signal)
    crossings = np.where(np.diff(np.sign(np.real(fft_signal))))[0]
    return len(crossings) / len(signal)

def zero_crossing_density_frequency(signal):
    fft_signal = np.fft.fft(signal)
    crossings = np.where(np.diff(np.sign(np.real(fft_signal))))[0]
    return len(crossings) / len(signal)

def spectral_concentration_center(signal, center_freq=0.5, bandwidth=0.1, fs=1.0):
    f, Pxx = welch(signal, fs=fs)
    center_band = (f > center_freq - bandwidth) & (f < center_freq + bandwidth)
    return np.sum(Pxx[center_band]) / np.sum(Pxx)

def inst_freq_deviation_std(signal):
    inst_freq = np.diff(np.unwrap(np.angle(hilbert(signal))))
    return np.std(inst_freq)

def rms_signal_envelope(signal):
    envelope = np.abs(hilbert(signal))
    return np.sqrt(np.mean(envelope ** 2))

def amplitude_asymmetry(signal):
    envelope = np.abs(hilbert(signal))
    return skew(envelope)

def spectral_modulation_bandwidth(signal, fs=1.0):
    f, Pxx = welch(signal, fs=fs)
    return np.sqrt(np.mean(f ** 2 * Pxx) - np.mean(f * Pxx) ** 2)

def wavelet_transform_cubic(signal, level=4):
    coeffs = pywt.wavedec(signal, 'db1', level=level)
    return np.mean(np.array([np.mean(np.abs(c) ** 3) for c in coeffs]))


# Band-Pass Filtered RMS of Signal Envelope
def band_pass_filtered_rms(signal, low_cut=0.05, high_cut=0.3):
    filtered_signal = apply_bandpass_filter(signal, low_cut, high_cut)
    envelope = np.abs(hilbert(filtered_signal))
    return np.sqrt(np.mean(envelope ** 2))

# Time-Frequency Energy Concentration
def time_frequency_energy_concentration(signal, low_freq=0.05, high_freq=0.3):
    f, t, Sxx = stft(signal, nperseg=256)
    band_mask = (f >= low_freq) & (f <= high_freq)
    energy_concentration = np.sum(Sxx[band_mask, :]) / np.sum(Sxx)
    return energy_concentration

# Peak Density in Filtered Frequency Domain
def peak_density_frequency_domain(signal):
    fft_signal = np.abs(fft(signal))
    smooth_fft = gaussian_filter1d(fft_signal, sigma=3)
    peaks = find_peaks(smooth_fft, height=0.1 * np.max(smooth_fft))[0]
    return len(peaks) / len(smooth_fft)

# Normalized High-Frequency Power Ratio
def normalized_high_freq_power_ratio(signal, high_cut=0.4):
    fft_signal = np.abs(fft(signal)) ** 2
    freqs = np.fft.fftfreq(len(signal))
    high_freq_power = np.sum(fft_signal[np.abs(freqs) > high_cut])
    total_power = np.sum(fft_signal)
    return high_freq_power / (total_power * (1 - high_cut))

# Frequency Domain Entropy with High-Frequency Emphasis
def high_freq_emphasis_entropy(signal, high_freq_cut=0.3):
    fft_signal = np.abs(fft(signal)) ** 2
    psd = fft_signal / np.sum(fft_signal)
    high_freqs = psd[len(psd) // 2:]
    high_freq_entropy = -np.sum(high_freqs * np.log2(high_freqs + 1e-10))
    return high_freq_entropy

# Autocorrelation Energy Spread
def autocorrelation_energy_spread(signal):
    autocorr = np.correlate(signal, signal, mode="full")
    return np.var(autocorr)

# Instantaneous Frequency Standard Deviation
def instantaneous_frequency_std(signal):
    inst_freq = np.diff(np.angle(hilbert(signal)))
    return np.std(inst_freq)

# Temporal Energy Variance (Gaussian)
def temporal_energy_variance_gaussian(signal):
    envelope = np.abs(hilbert(signal))
    smooth_envelope = gaussian_filter1d(envelope, sigma=3)
    return np.var(smooth_envelope)

# Wavelet Energy Concentration (High Frequency)
def high_freq_wavelet_energy_concentration(signal):
    coeffs = pywt.wavedec(signal, "db1", level=4)
    high_freq_coeffs = coeffs[-1]  # Highest frequency coefficients
    return np.sum(np.abs(high_freq_coeffs) ** 2)


# Frequency Spread Variability
def frequency_spread_variability(signal, window_size=64):
    # Split signal into windows
    variances = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i + window_size]
        f, Pxx = welch(window)
        mean_freq = np.sum(f * Pxx) / np.sum(Pxx)
        variance = np.sqrt(np.sum((f - mean_freq) ** 2 * Pxx) / np.sum(Pxx))
        variances.append(variance)
    return np.var(variances)

# Envelope Power Variability in Frequency Bands
def envelope_power_variability(signal, num_bands=4):
    envelope = np.abs(hilbert(signal))
    f, Pxx = welch(envelope)
    band_size = len(f) // num_bands
    band_variances = []
    for i in range(num_bands):
        band_power = np.sum(Pxx[i * band_size:(i + 1) * band_size])
        band_variances.append(band_power)
    return np.var(band_variances)

# Instantaneous Frequency Rate of Change
def instantaneous_frequency_rate_of_change(signal):
    inst_freq = np.diff(np.unwrap(np.angle(hilbert(signal))))
    rate_of_change = np.diff(inst_freq)
    return np.std(rate_of_change)

# Higher-Order Cumulants (5th Order)
def fifth_order_cumulant(signal):
    centered_signal = signal - np.mean(signal)
    return np.mean(centered_signal ** 5)

# Instantaneous Phase Deviation Rate
def instantaneous_phase_deviation_rate(signal):
    phase = np.unwrap(np.angle(signal))
    phase_diff = np.diff(phase)
    return np.std(phase_diff)

# Constellation Density Measure
def constellation_density(signal, num_bins=10):
    real_part = np.real(signal)
    imag_part = np.imag(signal)
    H, _, _ = np.histogram2d(real_part, imag_part, bins=num_bins)
    return np.var(H)  # Measure variance in density
