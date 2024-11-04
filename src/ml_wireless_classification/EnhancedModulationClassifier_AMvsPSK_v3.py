import os
import ctypes
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import (
    BaseModulationClassifier,
)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping

from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.base.SignalUtils import (
    augment_data_progressive,
    cyclical_lr,
    compute_spectral_kurtosis,
    compute_higher_order_cumulants,
    compute_zero_crossing_rate,
    compute_skewness,
    compute_fft_features,
)

from tensorflow.keras.callbacks import TensorBoard


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(
            data_path, model_path, stats_path
        )  # Call the base class constructor
        self.learning_rate = 0.0001  # Default learning rate

    def compute_features(self, iq_signal):
        # Envelope Variance
        envelope = np.abs(iq_signal)
        envelope_variance = np.var(envelope)

        # Instantaneous Frequency Stability (Variance of Instantaneous Frequency)
        instantaneous_phase = np.unwrap(np.angle(iq_signal))
        instantaneous_frequency = np.diff(instantaneous_phase)
        instantaneous_frequency_variance = np.var(instantaneous_frequency)

        # Phase Jitter
        phase_diff = np.diff(instantaneous_phase)
        phase_jitter = np.std(phase_diff)

        # Peak-to-Average Power Ratio (PAPR)
        power = np.abs(iq_signal) ** 2
        peak_power = np.max(power)
        average_power = np.mean(power)
        papr = peak_power / average_power if average_power != 0 else 0

        return envelope_variance, instantaneous_frequency_variance, phase_jitter, papr

    def prepare_data(self):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                # Separate real and imaginary parts for the IQ signal
                real_signal = signal[0]
                imag_signal = signal[1]
                
                # Normalize each channel separately to the range [-1, 1]
                max_real = np.max(np.abs(real_signal))
                max_imag = np.max(np.abs(imag_signal))
                real_signal = real_signal / max_real if max_real != 0 else real_signal
                imag_signal = imag_signal / max_imag if max_imag != 0 else imag_signal
                
                # Stack the normalized real and imaginary parts to form a (128, 2) array
                iq_array = np.vstack([real_signal, imag_signal]).T  # Shape: (128, 2)

                # Compute additional features using the helper methods
                envelope_variance, instantaneous_frequency_variance, phase_jitter, papr = self.compute_features(real_signal + 1j * imag_signal)
                envelope_variance = np.full((128, 1), envelope_variance)
                instantaneous_frequency_variance = np.full((128, 1), instantaneous_frequency_variance)
                phase_jitter = np.full((128, 1), phase_jitter)
                papr = np.full((128, 1), papr)

                # Compute additional advanced features
                spectral_kurtosis = np.mean(compute_spectral_kurtosis(real_signal + 1j * imag_signal))
                # fourth_order_cumulant = compute_higher_order_cumulants(real_signal + 1j * imag_signal, order=4)
                spectral_kurtosis_repeated = np.full((128, 1), np.nan_to_num(spectral_kurtosis))
                # fourth_order_cumulant_repeated = np.full((128, 1), np.nan_to_num(fourth_order_cumulant))

                # Compute FFT-based features
                center_frequency, peak_power, avg_power, std_dev_power = compute_fft_features(real_signal + 1j * imag_signal)
                center_frequency = np.full((128, 1), center_frequency)
                peak_power = np.full((128, 1), peak_power)
                avg_power = np.full((128, 1), avg_power)
                std_dev_power = np.full((128, 1), std_dev_power)

                # Additional features: Zero-crossing rate, skewness, entropy, flatness, and frequency spread
                zero_crossing_rate = np.full((128, 1), compute_zero_crossing_rate(real_signal))
                skewness = np.full((128, 1), compute_skewness(real_signal))
                spectral_entropy = np.full((128, 1), np.mean(-np.log(np.abs(real_signal) ** 2 + 1e-10) * np.abs(real_signal) ** 2))
                spectral_flatness = np.full((128, 1), np.exp(np.mean(np.log(np.abs(real_signal) ** 2 + 1e-10))) / np.mean(np.abs(real_signal) ** 2))
                frequency_spread = np.full((128, 1), np.std(np.diff(np.unwrap(np.angle(real_signal + 1j * imag_signal)))))

                # Stack IQ data (real and imaginary parts) with additional features and SNR
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([
                    iq_array,
                    snr_signal,
                    envelope_variance,
                    instantaneous_frequency_variance,
                    phase_jitter,
                    papr,
                    spectral_kurtosis_repeated,
                    center_frequency,
                    peak_power,
                    avg_power,
                    std_dev_power,
                    zero_crossing_rate,
                    skewness,
                    spectral_entropy,
                    spectral_flatness,
                    frequency_spread
                ])

                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)

        # Encode labels and split the data
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    
    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential(
                [
                    LSTM(128, input_shape=input_shape, return_sequences=True),
                    Dropout(0.2),
                    LSTM(128, return_sequences=False),
                    Dropout(0.2),
                    Dense(128, activation="relu"),
                    Dropout(0.2),
                    Dense(num_classes, activation="softmax"),
                ]
            )
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )


if __name__ == "__main__":
    # set the model name
    model_name = "EnhancedModulationClassifier_AMvsPSK_v3_2_2_2"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory

    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.models_dir = os.path.join(script_dir, "models")
    model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

    # Usage Example
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Stats path:", stats_path)

    # Initialize the classifier
    classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
    classifier.main()
