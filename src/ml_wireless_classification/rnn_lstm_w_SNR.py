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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Sequential
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

from tensorflow.keras.layers import Add, Conv2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(
            data_path, model_path, stats_path
        )  # Call the base class constructor
        self.learning_rate = 0.0001  # Default learning rate
        self.name = "rnn_lstm_w_SNR"

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
                iq_signal = np.vstack([real_signal, imag_signal]).T  # Shape: (128, 2)
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([iq_signal, snr_signal])
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
        if 0:  # os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()
            
            # Initial LSTM layers with increased units and Dropout
            self.model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
            self.model.add(Dropout(0.5))
            self.model.add(LSTM(256, return_sequences=False))
            self.model.add(Dropout(0.4))
            
            # Fully connected layers for classification
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            
            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))
            
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model


    def build_model_alt(self, input_shape, num_classes):
        if 0:#os.path.exists(self.model_path):
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

if __name__ == "__main__":
    # set the model name
    model_name = "rnn_lstm_w_SNR2"
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
