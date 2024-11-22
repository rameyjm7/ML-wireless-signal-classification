import numpy as np
from scipy.signal import hilbert, stft, find_peaks, welch
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import fft, fftfreq


class AdvancedFeatureExtractor:
    def __init__(self, signal, fs=1.0):
        self.fs = fs
        self.set_signal(signal)

    def set_signal(self, signal):
        self.signal = signal
        self.complex_signal = hilbert(np.real(signal))
        self.real_signal = np.real(signal)
        self.spectrum = np.abs(fft(signal))
        self.freqs = fftfreq(len(signal), 1 / self.fs)
        self.fft_result = np.fft.fft(signal)
        self.magnitude = np.abs(self.fft_result)
        
    def fft_center_frequency(self):
        """Compute the center frequency based on the peak index of the FFT magnitude."""
        peak_idx = np.argmax(self.magnitude)
        return peak_idx
    
    def fft_peak_power(self):
        """Compute the peak power of the FFT in dB."""
        peak_idx = np.argmax(self.magnitude)
        peak_power = 20 * np.log10(self.magnitude[peak_idx])
        return peak_power

    def fft_avg_power(self):
        """Compute the average power of the FFT in dB."""
        avg_power = 20 * np.log10(np.mean(self.magnitude))
        return avg_power
    
    def fft_std_dev_power(self):
        """Compute the standard deviation of the power of the FFT in dB."""
        std_dev_power = 20 * np.log10(np.std(self.magnitude))
        return std_dev_power

    def instantaneous_amplitude_mean(self):
        analytic_signal = hilbert(self.real_signal)
        amplitude = np.abs(analytic_signal)
        return np.mean(amplitude)

    def instantaneous_amplitude_std(self):
        analytic_signal = hilbert(self.real_signal)
        amplitude = np.abs(analytic_signal)
        return np.std(amplitude)

    def instantaneous_phase_mean(self):
        analytic_signal = hilbert(self.real_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        return np.mean(phase)

    def instantaneous_phase_std(self):
        analytic_signal = hilbert(self.real_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        return np.std(phase)

    def instantaneous_frequency_mean(self):
        analytic_signal = hilbert(self.real_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.diff(phase) / (2.0 * np.pi)
        frequency = np.pad(frequency, (0, 1), mode="edge")
        return np.mean(frequency)

    def instantaneous_frequency_std(self):
        analytic_signal = hilbert(self.real_signal)
        phase = np.unwrap(np.angle(analytic_signal))
        frequency = np.diff(phase) / (2.0 * np.pi)
        frequency = np.pad(frequency, (0, 1), mode="edge")
        return np.std(frequency)

    # Time Domain Features
    def snr(self):
        return 10 * np.log10(np.mean(self.signal**2) / np.var(self.signal))

    def papr(self):
        peak_power = np.max(self.signal**2)
        avg_power = np.mean(self.signal**2)
        return peak_power / avg_power

    def crest_factor(self):
        peak_amplitude = np.max(np.abs(self.signal))
        rms_value = np.sqrt(np.mean(self.signal**2))
        return peak_amplitude / rms_value

    def zero_crossing_rate(self):
        return np.sum(np.diff(np.sign(self.signal)) != 0) / len(self.signal)

    def kurtosis(self):
        return kurtosis(self.signal, fisher=False)

    def skewness(self):
        return skew(self.signal)

    def peak_to_rms_ratio(self):
        return np.max(self.signal) / np.sqrt(np.mean(self.signal**2))

    def peak_to_average_amplitude(self):
        return np.max(self.signal) / np.mean(np.abs(self.signal))

    def rms_value(self):
        return np.sqrt(np.mean(self.signal**2))

    def mean_abs_amplitude(self):
        return np.mean(np.abs(self.signal))

    # Frequency Domain Features
    def fft_peak_power(self):
        peak_idx = np.argmax(self.spectrum)
        return self.spectrum[peak_idx] ** 2

    def fft_avg_power(self):
        power = self.spectrum**2
        return np.mean(power)

    def fft_std_dev_power(self):
        power = self.spectrum**2
        return np.std(power)

    def fft_center_freq(self):
        peak_idx = np.argmax(self.spectrum)
        return self.freqs[peak_idx]

    def bandwidth(self):
        threshold = np.max(self.spectrum) * 0.25
        bandwidth_indices = np.where(self.spectrum >= threshold)[0]
        return self.freqs[bandwidth_indices[-1]] - self.freqs[bandwidth_indices[0]]

    def spectral_entropy(self):
        spectrum_norm = self.spectrum / np.sum(self.spectrum)
        return entropy(spectrum_norm)

    def spectral_flatness(self):
        geom_mean = np.exp(np.mean(np.log(self.spectrum + 1e-10)))
        arith_mean = np.mean(self.spectrum)
        return geom_mean / (arith_mean + 1e-10)

    def spectral_kurtosis(self):
        return kurtosis(self.spectrum, fisher=False)

    def spectral_skewness(self):
        return skew(self.spectrum)

    def spectral_peaks(self):
        return len(find_peaks(self.spectrum)[0])

    def spectral_sparsity(self):
        return np.sum(self.spectrum < np.mean(self.spectrum))

    # Envelope-based Features
    def envelope_mean(self):
        envelope = np.abs(self.complex_signal)
        return np.mean(envelope)

    def envelope_variance(self):
        envelope = np.abs(self.complex_signal)
        return np.var(envelope)

    def envelope_skewness(self):
        envelope = np.abs(self.complex_signal)
        return skew(envelope)

    def envelope_kurtosis(self):
        envelope = np.abs(self.complex_signal)
        return kurtosis(envelope, fisher=False)

    def envelope_peak_ratio(self):
        envelope = np.abs(self.complex_signal)
        return np.max(envelope) / np.mean(envelope)

    # Instantaneous Features
    def phase_variance(self):
        return np.var(np.angle(self.complex_signal))

    def rms_instant_freq(self):
        inst_freq = np.diff(np.unwrap(np.angle(self.complex_signal)))
        return np.sqrt(np.mean(inst_freq**2))

    def entropy_instant_freq(self):
        inst_freq = np.diff(np.unwrap(np.angle(self.complex_signal)))
        return entropy(inst_freq)

    def phase_skewness(self):
        phase = np.angle(self.complex_signal)
        return skew(phase)

    def phase_kurtosis(self):
        phase = np.angle(self.complex_signal)
        return kurtosis(phase, fisher=False)

    # Derived Spectral Features
    def spectral_rolloff(self, rolloff=0.85):
        cumulative_energy = np.cumsum(self.spectrum**2)
        total_energy = cumulative_energy[-1]
        rolloff_threshold = rolloff * total_energy
        rolloff_index = np.where(cumulative_energy >= rolloff_threshold)[0][0]
        return self.freqs[rolloff_index]

    def spectral_centroid(self):
        return np.sum(self.freqs * self.spectrum) / np.sum(self.spectrum + 1e-10)

    def spectral_energy(self):
        return np.sum(self.spectrum**2)

    def spectral_slope(self):
        freqs_centered = self.freqs - np.mean(self.freqs)
        spectrum_centered = self.spectrum - np.mean(self.spectrum)
        slope = np.sum(freqs_centered * spectrum_centered) / np.sum(
            freqs_centered**2 + 1e-10
        )
        return slope

    def spectral_spread(self):
        centroid = self.spectral_centroid()
        spread = np.sqrt(
            np.sum(((self.freqs - centroid) ** 2) * self.spectrum)
            / np.sum(self.spectrum + 1e-10)
        )
        return spread


    # Method to get all features
    def get_features(self):
        feature_dict = {
            method: getattr(self, method)
            for method in dir(self)
            if callable(getattr(self, method))
            and not method.startswith("__")
            and not "get_features" in method
            and not "set_signal" in method
        }
        return feature_dict
