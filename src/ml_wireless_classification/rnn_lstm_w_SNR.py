import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.layers import (
    Dense,
    Dropout,
)
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import (
    BaseModulationClassifier,
)

from ml_wireless_classification.base.CommonVars import common_vars, RUN_MODE


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
                # # Stack the normalized real and imaginary parts to form a (128, 2) array
                # iq_signal = np.vstack([real_signal, imag_signal]).T  # Shape: (128, 2)
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

    def build_model_v3(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
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
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    def build_model_v4(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()

            # Initial Bidirectional LSTM layers with increased units and Dropout
            self.model.add(
                Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape)
            )
            self.model.add(Dropout(0.5))
            self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
            self.model.add(Dropout(0.4))

            # Fully connected layers for classification
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    def build_model_v5(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()

            # Initial Bidirectional LSTM layers with increased units and Dropout
            self.model.add(
                Bidirectional(LSTM(256, return_sequences=True), input_shape=input_shape)
            )
            self.model.add(Dropout(0.5))
            self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
            self.model.add(Dropout(0.4))

            # Fully connected layers for classification
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    def build_model_v6(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()

            # Initial Bidirectional LSTM layers with increased units and Dropout
            self.model.add(
                Bidirectional(LSTM(512, input_shape=input_shape, return_sequences=True))
            )
            self.model.add(Dropout(0.5))
            self.model.add(Bidirectional(LSTM(512, return_sequences=True)))
            self.model.add(Dropout(0.4))
            self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
            self.model.add(Dropout(0.3))

            # Fully connected layers for classification
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    # add recurrent dropout to LSMT layers
    def build_model_v7(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()

            # Initial Bidirectional LSTM layers with increased units and Dropout
            self.model.add(
                Bidirectional(
                    LSTM(
                        512,
                        input_shape=input_shape,
                        return_sequences=True,
                        dropout=0.5,  # Regular dropout applied to inputs
                        recurrent_dropout=0.3,  # Recurrent dropout applied to recurrent connections
                    )
                )
            )
            self.model.add(Dropout(0.5))
            self.model.add(
                Bidirectional(
                    LSTM(
                        512,
                        input_shape=input_shape,
                        return_sequences=True,
                        dropout=0.4,  # Regular dropout applied to inputs
                        recurrent_dropout=0.2,  # Recurrent dropout applied to recurrent connections
                    )
                )
            )
            self.model.add(Dropout(0.4))
            self.model.add(
                Bidirectional(
                    LSTM(
                        256,
                        input_shape=input_shape,
                        return_sequences=False,
                        dropout=0.3,  # Regular dropout applied to inputs
                        recurrent_dropout=0.1,  # Recurrent dropout applied to recurrent connections
                    )
                )
            )
            self.model.add(Dropout(0.3))

            # Fully connected layers for classification
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    def build_model_v8(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new complex model")
            self.model = Sequential()

            # Initial LSTM layers with increased units and Dropout
            self.model.add(Bidirectional(LSTM(256, input_shape=input_shape, return_sequences=True)))
            self.model.add(Dropout(0.5))
            self.model.add(Bidirectional(LSTM(256, return_sequences=False)))
            self.model.add(Dropout(0.4))

            # Fully connected layers for classification
            self.model.add(
                Dense(512, activation="relu")
            )  # New dense layer with 512 units
            self.model.add(Dropout(0.5))  # Higher dropout for the added layer
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dropout(0.4))
            self.model.add(Dense(128, activation="relu"))
            self.model.add(Dropout(0.3))
            self.model.add(
                Dense(96, activation="relu")
            )  # New dense layer with 96 units
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64, activation="relu"))
            self.model.add(Dropout(0.2))
            self.model.add(
                Dense(32, activation="relu")
            )  # Final dense layer before output
            self.model.add(
                Dropout(0.1)
            )  # Slight dropout for last fully connected layer

            # Output layer
            self.model.add(Dense(num_classes, activation="softmax"))

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
        return self.model

    def build_model_0(self, input_shape, num_classes):
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
            
    def build_model(self, input_shape, num_classes):
        return self.build_model_v3(input_shape, num_classes)


if __name__ == "__main__":
    # set the model name
    model_name = "rnn_lstm_w_SNR_5_2_1"
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory

    common_vars.stats_dir = os.path.join(script_dir, "stats")
    common_vars.models_dir = os.path.join(script_dir, " models")
    model_path = os.path.join(script_dir, "models", f"{model_name}.keras")
    stats_path = os.path.join(script_dir, "stats", f"{model_name}_stats.json")

    # Usage Example
    print("Data path:", data_path)
    print("Model path:", model_path)
    print("Stats path:", stats_path)

    # Initialize the classifier
    classifier = ModulationLSTMClassifier(data_path, model_path, stats_path)
    classifier.main(mode=RUN_MODE.EVALUATE_ONLY)
