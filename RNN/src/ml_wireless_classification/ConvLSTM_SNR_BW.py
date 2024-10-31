import os
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from RNN.base.CommonVars import common_vars
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier

class ModulationConvLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        super().__init__(data_path, model_path, stats_path)
        self.learning_rate = 0.0001  # Default learning rate

    def compute_bandwidth(self, signal, threshold_ratio=0.5):
        """Compute the bandwidth of a signal based on its FFT and a threshold ratio."""
        fft_magnitude = np.abs(np.fft.fft(signal))
        max_magnitude = np.max(fft_magnitude)
        threshold = threshold_ratio * max_magnitude
        bandwidth_indices = np.where(fft_magnitude >= threshold)[0]
        bandwidth = len(bandwidth_indices)  # Simple count of bins as bandwidth
        return bandwidth

    def prepare_data(self):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                # Perform a 128-point FFT on each signal
                iq_signal = np.fft.fft(signal[0] + 1j * signal[1], n=128).real  # Use real part for Conv1D
                snr_signal = np.full((128, 1), snr)

                # Compute bandwidth as an additional feature
                bandwidth = self.compute_bandwidth(signal[0] + 1j * signal[1])  # Combine I/Q for FFT
                bandwidth_signal = np.full((128, 1), bandwidth)

                # Stack all features: I/Q FFT, SNR, and Bandwidth
                combined_signal = np.hstack([iq_signal.reshape(-1, 1), snr_signal, bandwidth_signal])
                X.append(combined_signal)
                y.append(mod_type)

        X = np.array(X)
        y = np.array(y)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2])
        X_test = X_test.reshape(-1, X_test.shape[1], X_test.shape[2])

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model with Conv1D and LSTM layers")
            self.model = Sequential(
                [
                    Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
                    MaxPooling1D(pool_size=2),
                    LSTM(128, return_sequences=True),
                    Dropout(0.5),
                    LSTM(128, return_sequences=False),
                    Dropout(0.2),
                    Dense(128, activation="relu"),
                    Dropout(0.1),
                    Dense(num_classes, activation="softmax"),
                ]
            )
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
