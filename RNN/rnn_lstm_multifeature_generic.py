import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from abc import ABC, abstractmethod


# Base Abstract Class
class BaseModulationClassifier(ABC):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        self.data_path = data_path
        self.model_path = model_path
        self.stats_path = stats_path
        self.data_pickle_path = f"{model_path}_data.pkl"  # Pickle path for preprocessed data
        self.data = None
        self.label_encoder = None
        self.model = None
        self.stats = {
            "date_created": None,
            "epochs_trained": 0,
            "best_accuracy": 0,
            "current_accuracy": 0,
            "last_trained": None
        }
        self.learning_rate = 0.0001  # Default learning rate
        self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded model stats from {self.stats_path}")
        else:
            self.stats["date_created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.save_stats()

    def save_stats(self):
        with open(self.stats_path, 'w') as f:
            json.dump(self.stats, f, indent=4)
        print(f"Saved model stats to {self.stats_path}")

    def load_data(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    @abstractmethod
    def prepare_data(self):
        """
        Abstract method for preparing data that must be implemented by inheriting classes.
        """
        pass

    def save_model(self):
        mpath = f'{self.model_path}'
        self.model.save(mpath, save_format='keras')
        print(f"Model saved to {mpath}")


# Child class inheriting from the abstract class, implementing `prepare_data`
class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        super().__init__(data_path, model_path, stats_path)

    def compute_fft_features(self, signal):
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)

        peak_idx = np.argmax(magnitude)
        center_frequency = peak_idx
        peak_power = 20 * np.log10(magnitude[peak_idx])
        avg_power = 20 * np.log10(np.mean(magnitude))
        std_dev_power = 20 * np.log10(np.std(magnitude))

        return center_frequency, peak_power, avg_power, std_dev_power

    def compute_instantaneous_features(self, signal):
        analytic_signal = hilbert(np.real(signal))
        instantaneous_amplitude = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), mode='edge')

        return instantaneous_amplitude, instantaneous_phase, instantaneous_frequency

    def autocorrelation(self, signal):
        result = np.correlate(signal, signal, mode='full')
        return result[result.size // 2:]

    def is_digital_signal(self, autocorr_signal, threshold=0.1):
        normalized_autocorr = autocorr_signal / np.max(autocorr_signal)
        is_digital = np.any(normalized_autocorr < threshold)
        return 1 if is_digital else 0

    def compute_kurtosis(self, signal):
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        kurtosis = np.mean((signal - mean_signal)**4) / (std_signal**4)
        return kurtosis

    def compute_skewness(self, signal):
        mean_signal = np.mean(signal)
        std_signal = np.std(signal)
        skewness = np.mean((signal - mean_signal)**3) / (std_signal**3)
        return skewness

    def compute_spectral_energy_concentration(self, signal, center_freq_idx, bandwidth):
        fft_result = np.fft.fft(signal)
        magnitude = np.abs(fft_result)
        lower_bound = max(0, center_freq_idx - bandwidth // 2)
        upper_bound = min(len(magnitude), center_freq_idx + bandwidth // 2)
        spectral_energy = np.sum(magnitude[lower_bound:upper_bound]**2)
        total_energy = np.sum(magnitude**2)
        energy_concentration = spectral_energy / total_energy
        return energy_concentration

    def compute_zero_crossing_rate(self, signal):
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        zcr = len(zero_crossings) / len(signal)
        return zcr

    def compute_instantaneous_frequency_jitter(self, instantaneous_frequency):
        return np.std(instantaneous_frequency)

    def prepare_data(self):
        if os.path.exists(self.data_pickle_path):
            print(f"Loading prepared data from {self.data_pickle_path}")
            with open(self.data_pickle_path, 'rb') as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            return X_train, X_test, y_train, y_test

        print("Preparing data from scratch...")

        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T

                # Compute FFT features
                center_freq, peak_power, avg_power, std_dev_power = self.compute_fft_features(signal[0] + 1j * signal[1])

                # Compute instantaneous features
                instantaneous_amplitude, instantaneous_phase, instantaneous_frequency = self.compute_instantaneous_features(signal[0] + 1j * signal[1])

                # Compute autocorrelation and digital/analog flag
                autocorr_signal = self.autocorrelation(signal[0])
                is_digital = self.is_digital_signal(autocorr_signal)

                # Higher-order statistics
                kurtosis = self.compute_kurtosis(iq_signal)
                skewness = self.compute_skewness(iq_signal)

                # Spectral energy concentration
                energy_concentration = self.compute_spectral_energy_concentration(signal[0] + 1j * signal[1], center_freq, bandwidth=10)

                # Zero-crossing rate
                zcr = self.compute_zero_crossing_rate(signal[0])

                # Instantaneous frequency jitter
                freq_jitter = self.compute_instantaneous_frequency_jitter(instantaneous_frequency)

                # SNR feature
                snr_signal = np.full((128, 1), snr)

                # Append all features
                combined_signal = np.hstack([
                    iq_signal,                
                    snr_signal,               
                    np.full((128, 1), center_freq),       
                    np.full((128, 1), peak_power),        
                    np.full((128, 1), avg_power),         
                    np.full((128, 1), std_dev_power),     
                    instantaneous_amplitude.reshape(-1, 1),
                    instantaneous_phase.reshape(-1, 1),   
                    instantaneous_frequency.reshape(-1, 1),
                    np.full((128, 1), is_digital),        
                    np.full((128, 1), kurtosis),
                    np.full((128, 1), skewness),
                    np.full((128, 1), energy_concentration),
                    np.full((128, 1), zcr),
                    np.full((128, 1), freq_jitter)
                ])

                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        with open(self.data_pickle_path, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Prepared data saved to {self.data_pickle_path}")

        return X_train, X_test, y_train, y_test


# Usage Example
data_path = '../RML2016.10a_dict.pkl'
model_path = 'rnn_lstm_multifeature_generic.keras'
stats_path = f'{model_path}_stats.json'

classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
classifier.load_data()
X_train, X_test, y_train, y_test = classifier.prepare_data()
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y_train))
classifier.build_model(input_shape, num_classes)
