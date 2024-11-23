from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import ctypes
import gc
import json
from datetime import datetime
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)

from ml_wireless_classification.base.SignalUtils import (
    autocorrelation,
    is_digital_signal,
    compute_kurtosis,
    compute_skewness,
    compute_spectral_energy_concentration,
    compute_zero_crossing_rate,
    compute_instantaneous_frequency_jitter,
    # compute_fft_features,
    compute_instantaneous_features,
    augment_data_progressive,
    cyclical_lr
)
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier
# decrease debug messages
tf.get_logger().setLevel("ERROR")



class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(data_path, model_path, stats_path)
        self.name = "ConvLSTM_FFT_Power_SNR"
        
    def compute_fft_features(self, signal):
        # Perform 128-point FFT on the signal
        fft_result = np.fft.fft(signal, n=128)
        power_spectrum = np.abs(fft_result) ** 2  # Power spectrum of the FFT result

        # Calculate additional frequency-domain features
        avg_power = np.mean(power_spectrum)
        peak_power = np.max(power_spectrum)
        std_dev_power = np.std(power_spectrum)

        return power_spectrum, avg_power, std_dev_power, peak_power

    # Function to build the new model based on input shape
    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(64, return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )

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
                power_spectrum, avg_power, std_dev_power, peak_power = self.compute_fft_features(signal[0] + 1j * signal[1])

                # Ensure shapes for concatenation
                power_spectrum = power_spectrum[:128].reshape(128, 1)  # Limit to 128 and reshape to (128, 1)
                avg_power = np.full((128, 1), avg_power)               # Repeat avg_power to (128, 1)
                std_dev_power = np.full((128, 1), std_dev_power)       # Repeat std_dev_power to (128, 1)
                peak_power = np.full((128, 1), peak_power)             # Repeat peak_power to (128, 1)

                # Combine all features
                combined_signal = np.hstack([
                    power_spectrum,  # 128-point FFT (128, 1)
                    avg_power,       # Average power (128, 1)
                    std_dev_power,   # Std. dev of power (128, 1)
                    peak_power       # Peak power (128, 1)
                ])

                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)

        # Encode labels and split the data
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Save processed data for future use
        with open(self.data_pickle_path, 'wb') as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Prepared data saved to {self.data_pickle_path}")

        return X_train, X_test, y_train, y_test
