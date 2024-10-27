import os
import ctypes
import json
from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from BaseModulationClassifier import BaseModulationClassifier

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)

from SignalUtils import augment_data_progressive, cyclical_lr


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(
            data_path, model_path, stats_path
        )  # Call the base class constructor
        self.learning_rate = 0.0001  # Default learning rate

    def prepare_data(self):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T
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
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print(f"Building new model")
            self.model = Sequential(
                [
                    LSTM(128, input_shape=input_shape, return_sequences=True),
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
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        callbacks = [early_stopping]

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

    def update_model_dropout(self, new_dropout_rate=0.3):
        model_config = self.model.get_config()
        new_model = Sequential()

        for layer in model_config["layers"]:
            layer_type = layer["class_name"]
            if layer_type == "LSTM":
                new_model.add(
                    LSTM(
                        units=layer["config"]["units"],
                        input_shape=layer["config"]["batch_input_shape"][1:],
                        return_sequences=layer["config"]["return_sequences"],
                    )
                )
            elif layer_type == "Dense":
                new_model.add(
                    Dense(
                        units=layer["config"]["units"],
                        activation=layer["config"]["activation"],
                    )
                )
            elif layer_type == "Dropout":
                new_model.add(Dropout(rate=new_dropout_rate))

        optimizer = Adam(learning_rate=self.learning_rate)
        new_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )
        new_model.set_weights(self.model.get_weights())
        self.model = new_model
        print(f"Updated Dropout layers to {new_dropout_rate} and recompiled the model.")

# set the model name 
model_name = "rnn_lstm_w_SNR_5_2_1"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths with the script directory as the base
data_path = os.path.join(
    script_dir, "..", "RML2016.10a_dict.pkl"
)  # One level up from the script's directory
model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

# Usage Example
print("Data path:", data_path)
print("Model path:", model_path)
print("Stats path:", stats_path)

# Initialize the classifier
classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)

# Load the dataset
classifier.load_data()

# Prepare the data
X_train, X_test, y_train, y_test = classifier.prepare_data()

# Build the LSTM model (load if it exists)
input_shape = (
    X_train.shape[1],
    X_train.shape[2],
)  # Time steps and features (with SNR as additional feature)
num_classes = len(np.unique(y_train))  # Number of unique modulation types
classifier.build_model(input_shape, num_classes)

# Set the learning rate
classifier.set_learning_rate(1e-4)

# Train continuously with cyclical learning rates
classifier.train_continuously(
    X_train, y_train, X_test, y_test, batch_size=64, use_clr=True, clr_step_size=10
)

# Evaluate the model
classifier.evaluate(X_test, y_test)

# Optional: Make predictions on the test set
predictions = classifier.predict(X_test)
print("Predicted Labels: ", predictions[:5])
print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))
