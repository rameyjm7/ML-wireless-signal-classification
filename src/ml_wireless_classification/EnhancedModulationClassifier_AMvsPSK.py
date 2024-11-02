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
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)
from ml_wireless_classification.base.CustomEarlyStopping import CustomEarlyStopping

from ml_wireless_classification.base.CommonVars import common_vars
from ml_wireless_classification.base.SignalUtils import augment_data_progressive, cyclical_lr



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
                iq_signal = signal[0] + 1j * signal[1]  # Convert to complex form
                iq_array = np.vstack([signal[0], signal[1]]).T  # Shape: (128, 2)

                # Calculate the four additional features
                envelope_variance, instantaneous_frequency_variance, phase_jitter, papr = self.compute_features(iq_signal)

                # Repeat the additional features to match the length of IQ data (128 points)
                envelope_variance = np.full((128, 1), envelope_variance)
                instantaneous_frequency_variance = np.full((128, 1), instantaneous_frequency_variance)
                phase_jitter = np.full((128, 1), phase_jitter)
                papr = np.full((128, 1), papr)

                # Stack IQ data with additional features along with the SNR
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([iq_array, snr_signal, envelope_variance, instantaneous_frequency_variance, phase_jitter, papr])
                
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
    



if __name__ == "__main__":
    # set the model name
    model_name = "EnhancedModulationClassifier_AMvsPSK"
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
