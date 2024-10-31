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
                    Dropout(0.25),
                    LSTM(128, return_sequences=False),
                    Dropout(0.25),
                    Dense(128, activation="relu"),
                    Dropout(0.25),
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
        early_stopping_custom = CustomEarlyStopping(
            monitor="val_accuracy",
            min_delta=0.01,
            patience=5,
            restore_best_weights=True,
        )

        # Add it to the list of callbacks
        callbacks = [early_stopping_custom]

        if use_clr:
            clr_scheduler = LearningRateScheduler(
                lambda epoch: cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 5
        for epoch in range(epochs // stats_interval):
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

    def transfer_weights(self, original_model_path, new_model_path, dropout_rates=(0.5, 0.2, 0.1)):
        # Load the original model
        original_model = load_model(original_model_path)
        
        # Verify the layer structure of the original model
        for i, layer in enumerate(original_model.layers):
            print(f"Original Model - Layer {i}: {layer.name}, Type: {layer.__class__.__name__}, Dropout: {getattr(layer, 'rate', 'N/A')}")

        # Build a new model with the specified dropout rates, ensuring the same structure as the original
        new_model = Sequential([
            LSTM(128, input_shape=original_model.input_shape[1:], return_sequences=True),
            Dropout(dropout_rates[0]),
            LSTM(128, return_sequences=False),
            Dropout(dropout_rates[1]),
            Dense(128, activation="relu"),
            Dropout(dropout_rates[2]),
            Dense(original_model.output_shape[-1], activation="softmax")
        ])
        
        # Transfer weights from the original model to the new model
        for i, layer in enumerate(original_model.layers):
            if layer.__class__.__name__ in ["LSTM", "Dense"]:
                new_model.layers[i].set_weights(layer.get_weights())
                print(f"Transferred weights for layer {i} ({layer.name})")
            else:
                print(f"Skipped weight transfer for non-trainable layer {i} ({layer.name})")

        # Compile the new model to finalize it
        new_model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=Adam(learning_rate=0.0001),
            metrics=["accuracy"]
        )

        # Save the new model
        new_model.save(new_model_path)
        print(f"New model with transferred weights saved to {new_model_path}")
        
        return new_model


def main(model_name, make_new_dropout_model=False):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths with the script directory as the base
    data_path = os.path.join(
        script_dir, "..", "RML2016.10a_dict.pkl"
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

    if make_new_dropout_model:
        dropouts = [0.5, 0.2, 0.1]
        # make a new model with the same weights, but different dropout rates
        new_model_name = (
            "rnn_lstm_w_SNR_"
            + str(dropouts[0])
            + "_"
            + str(dropouts[1])
            + "_"
            + str(dropouts[2])
        )
        new_model_path = os.path.join(script_dir, "models", f"{new_model_name}.keras")
        classifier.transfer_weights(
            model_path,
            new_model_path
        )
        return 0

    # Train continuously with cyclical learning rates
    classifier.train_continuously(
        X_train, y_train, X_test, y_test, batch_size=64, use_clr=False, clr_step_size=10
    )

    # Evaluate the model
    classifier.evaluate(X_test, y_test)

    # Optional: Make predictions on the test set
    predictions = classifier.predict(X_test)
    print("Predicted Labels: ", predictions[:5])
    print("True Labels: ", classifier.label_encoder.inverse_transform(y_test[:5]))


if __name__ == "__main__":
    # set the model name 
    model_name = "rnn_lstm_w_SNR"
    main(model_name)
    # set the model name
    model_name = "rnn_lstm_w_SNR_5_2_1"
    main(model_name, False)
