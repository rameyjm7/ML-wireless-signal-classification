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
    compute_fft_features,
    compute_instantaneous_features,
    augment_data_progressive,
    cyclical_lr
)
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping
# decrease debug messages
tf.get_logger().setLevel("ERROR")


# Child class inheriting from the abstract class, implementing `prepare_data`
class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(data_path, model_path, stats_path)
        self.name = "rnn_lstm_multifeature_generic_5_2_1"

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential()
            self.model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(LSTM(128, return_sequences=False))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )

    def train(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=20,
        batch_size=64,
        use_clr=False,
        clr_step_size=10,
    ):
        early_stopping_custom = CustomEarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=5, restore_best_weights=True)

        # Add it to the list of callbacks
        callbacks = [early_stopping_custom]


        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 5
        for epoch in range(epochs//stats_interval):
            # X_train_augmented = augment_data_progressive(X_train.copy(), epoch, epochs)
            history = self.model.fit(
                X_train,
                y_train,
                epochs=stats_interval,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
            )

            self.update_epoch_stats(epochs)
            current_accuracy = max(history.history["val_accuracy"])
            self.update_and_save_stats(current_accuracy)

        return history
    
        
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
                center_freq, peak_power, avg_power, std_dev_power = compute_fft_features(signal[0] + 1j * signal[1])

                # Compute instantaneous features
                instantaneous_amplitude, instantaneous_phase, instantaneous_frequency = compute_instantaneous_features(signal[0] + 1j * signal[1])

                # Compute autocorrelation and digital/analog flag
                autocorr_signal = autocorrelation(signal[0])
                is_digital = is_digital_signal(autocorr_signal)

                # Higher-order statistics
                kurtosis = compute_kurtosis(iq_signal)
                skewness = compute_skewness(iq_signal)

                # Spectral energy concentration
                energy_concentration = compute_spectral_energy_concentration(signal[0] + 1j * signal[1], center_freq, bandwidth=10)

                # Zero-crossing rate
                zcr = compute_zero_crossing_rate(signal[0])

                # Instantaneous frequency jitter
                freq_jitter = compute_instantaneous_frequency_jitter(instantaneous_frequency)

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
