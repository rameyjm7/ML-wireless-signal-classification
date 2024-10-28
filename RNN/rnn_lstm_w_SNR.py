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

from tensorflow.keras.callbacks import EarlyStopping, Callback

class CustomEarlyStopping(Callback):
    def __init__(self, monitor="val_accuracy", min_delta=0.01, patience=5, restore_best_weights=True):
        """
        Custom early stopping to stop training if validation accuracy exceeds the current highest by min_delta.

        Parameters:
        - monitor (str): Metric to monitor (default is 'val_accuracy').
        - min_delta (float): Minimum improvement over best accuracy to continue training.
        - patience (int): Number of epochs to wait after last improvement.
        - restore_best_weights (bool): Whether to restore the weights of the best epoch.
        """
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best = -float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        if current is None:
            print(f"Warning: Metric {self.monitor} is not available.")
            return
        
        # If current accuracy exceeds the best by min_delta, update best and reset wait counter
        if current > self.best + self.min_delta:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            # Increment the wait counter if no improvement
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    print("Restoring model weights from the best epoch.")
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")



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
        # Define the custom early stopping callback
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
