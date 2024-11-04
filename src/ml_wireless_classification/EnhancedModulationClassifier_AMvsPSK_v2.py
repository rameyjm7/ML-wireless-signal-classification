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
        data_pickle_path = os.path.join(
            common_vars.data_dir, f"{common_vars.model_name}_data.pkl"
        )
        # Check if the data pickle file exists
        if os.path.exists(data_pickle_path):
            print(f"Loading prepared data from {data_pickle_path}")
            with open(data_pickle_path, "rb") as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            return X_train, X_test, y_train, y_test

        # Prepare data from scratch if pickle file does not exist
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = signal[0] + 1j * signal[1]  # Convert to complex form

                # Normalize the IQ data
                max_value = np.max(np.abs(iq_signal))
                if max_value != 0:
                    iq_signal /= max_value  # Normalize to [-1, 1]

                iq_array = np.vstack([signal[0], signal[1]]).T  # Shape: (128, 2)

                # Calculate the existing additional features
                (
                    envelope_variance,
                    instantaneous_frequency_variance,
                    phase_jitter,
                    papr,
                ) = self.compute_features(iq_signal)

                # Calculate new additional features
                spectral_kurtosis = compute_spectral_kurtosis(iq_signal)
                fourth_order_cumulant = compute_higher_order_cumulants(
                    iq_signal, order=4
                )

                # Ensure the shapes for concatenation
                envelope_variance = np.full((128, 1), envelope_variance)
                instantaneous_frequency_variance = np.full(
                    (128, 1), instantaneous_frequency_variance
                )
                phase_jitter = np.full((128, 1), phase_jitter)
                papr = np.full((128, 1), papr)
                spectral_kurtosis = np.nan_to_num(np.mean(spectral_kurtosis), nan=0.0)
                spectral_kurtosis_repeated = np.full((128, 1), spectral_kurtosis)
                fourth_order_cumulant = np.nan_to_num(fourth_order_cumulant, nan=0.0)
                fourth_order_cumulant_repeated = np.full(
                    (128, 1), fourth_order_cumulant
                )

                # Stack IQ data with additional features along with the SNR
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack(
                    [
                        iq_array,
                        snr_signal,
                        envelope_variance,
                        instantaneous_frequency_variance,
                        phase_jitter,
                        papr,
                        spectral_kurtosis_repeated,
                        fourth_order_cumulant_repeated,
                    ]
                )

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

        # Save the prepared data to a pickle file
        with open(data_pickle_path, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Prepared data saved to {data_pickle_path}")

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
    model_name = "EnhancedModulationClassifier_AMvsPSK_v2_2_2_2"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory

    common_vars.model_name = model_name
    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.data_dir = os.path.join(script_dir, "data")
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
