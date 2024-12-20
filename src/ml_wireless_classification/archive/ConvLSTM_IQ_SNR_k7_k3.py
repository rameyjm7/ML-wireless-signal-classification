import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        super().__init__(data_path, model_path, stats_path)
        self.learning_rate = 0.0001  # Default learning rate
        self.name = "ConvLSTM_IQ_SNR_k7_k3"

    def prepare_data(self):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                # Perform a 128-point FFT on each signal
                iq_signal = np.fft.fft(signal[0] + 1j * signal[1], n=128).real  # Use real part for Conv1D
                snr_signal = np.full((128, 1), snr)
                combined_signal = np.hstack([iq_signal.reshape(-1, 1), snr_signal])
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
            print(f"Building new model with parallel Conv1D and LSTM layers")

            # Input layer
            inputs = Input(shape=input_shape)

            # Parallel Conv1D layers with kernel sizes 7 and 3
            conv_7 = Conv1D(filters=64, kernel_size=7, activation='relu', padding='same')(inputs)
            conv_3 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)

            # Pooling layers
            pool_7 = MaxPooling1D(pool_size=2)(conv_7)
            pool_3 = MaxPooling1D(pool_size=2)(conv_3)

            # Concatenate the outputs from both parallel layers
            concatenated = Concatenate()([pool_7, pool_3])

            # LSTM layers
            lstm_1 = LSTM(128, return_sequences=True)(concatenated)
            dropout_1 = Dropout(0.5)(lstm_1)
            lstm_2 = LSTM(128, return_sequences=False)(dropout_1)
            dropout_2 = Dropout(0.2)(lstm_2)

            # Fully connected dense layers
            dense_1 = Dense(128, activation="relu")(dropout_2)
            dropout_3 = Dropout(0.1)(dense_1)
            outputs = Dense(num_classes, activation="softmax")(dropout_3)

            # Create the model
            self.model = Model(inputs=inputs, outputs=outputs)

            # Compile the model
            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
