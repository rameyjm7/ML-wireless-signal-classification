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
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Input, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    LearningRateScheduler,
)


from SignalUtils import (
    autocorrelation,
    is_digital_signal,
    compute_kurtosis,
    compute_skewness,
    compute_spectral_energy_concentration,
    compute_zero_crossing_rate,
    compute_instantaneous_frequency_jitter,
    compute_instantaneous_features,
    augment_data_progressive,
)
from BaseModulationClassifier import BaseModulationClassifier
from CustomEarlyStopping import CustomEarlyStopping

# decrease debug messages
tf.get_logger().setLevel("ERROR")


def compute_fft_features(signal):
    # Perform 128-point FFT on the signal
    fft_result = np.fft.fft(signal, n=128)
    power_spectrum = np.abs(fft_result) ** 2  # Power spectrum of the FFT result

    # Calculate additional frequency-domain features
    avg_power = np.mean(power_spectrum)
    peak_power = np.max(power_spectrum)
    std_dev_power = np.std(power_spectrum)

    return power_spectrum, avg_power, std_dev_power, peak_power


# Child class inheriting from the abstract class, implementing `prepare_data`
class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(data_path, model_path, stats_path)


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(
        self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"
    ):
        super().__init__(data_path, model_path, stats_path)


    def cyclical_lr(self, epoch, base_lr=1e-7, max_lr=1e-6, step_size=10):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr

    def build_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
        else:
            print("Building new model")

            # IQ Input
            iq_input = Input(shape=input_shape)

            # FFT Branch: Compute FFT and transform for input into ConvLSTM
            fft_layer = Lambda(
                lambda x: tf.abs(tf.signal.fft(tf.cast(x, tf.complex64)))
            )(iq_input)
            fft_layer = Lambda(lambda x: tf.expand_dims(x, axis=-1))(
                fft_layer
            )  # Expand dims for ConvLSTM compatibility
            x1 = ConvLSTM2D(32, (3, 3), activation="relu", return_sequences=True)(
                fft_layer
            )
            x1 = Dropout(0.2)(x1)
            x1 = ConvLSTM2D(64, (3, 3), activation="relu", return_sequences=False)(x1)
            x1 = Dropout(0.2)(x1)
            x1 = Flatten()(x1)

            # IQ Branch
            iq_layer = Lambda(lambda x: tf.expand_dims(x, axis=-1))(
                iq_input
            )  # Expand dims for ConvLSTM compatibility
            x2 = ConvLSTM2D(32, (3, 3), activation="relu", return_sequences=True)(
                iq_layer
            )
            x2 = Dropout(0.2)(x2)
            x2 = ConvLSTM2D(64, (3, 3), activation="relu", return_sequences=False)(x2)
            x2 = Dropout(0.2)(x2)
            x2 = Flatten()(x2)

            # Concatenate both branches
            combined = Concatenate()([x1, x2])

            # Dense layers
            x = Dense(128, activation="relu")(combined)
            x = Dropout(0.2)(x)
            x = Dense(64, activation="relu")(x)
            x = Dropout(0.2)(x)
            output = Dense(num_classes, activation="softmax")(x)

            # Define the model with both inputs
            self.model = Model(inputs=iq_input, outputs=output)

            optimizer = Adam(learning_rate=self.learning_rate)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )

    def prepare_data(self):
        if os.path.exists(self.data_pickle_path):
            print(f"Loading prepared data from {self.data_pickle_path}")
            with open(self.data_pickle_path, "rb") as f:
                X_train, X_test, y_train, y_test = pickle.load(f)
            return X_train, X_test, y_train, y_test

        print("Preparing data from scratch...")

        X = []
        y = []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                iq_signal = np.vstack([signal[0], signal[1]]).T

                # Compute FFT features
                power_spectrum, avg_power, std_dev_power, peak_power = (
                    compute_fft_features(signal[0] + 1j * signal[1])
                )

                # Ensure shapes for concatenation
                power_spectrum = power_spectrum[:128].reshape(
                    128, 1
                )  # Limit to 128 and reshape to (128, 1)
                avg_power = np.full((128, 1), avg_power)  # Repeat avg_power to (128, 1)
                std_dev_power = np.full(
                    (128, 1), std_dev_power
                )  # Repeat std_dev_power to (128, 1)
                peak_power = np.full(
                    (128, 1), peak_power
                )  # Repeat peak_power to (128, 1)

                # Combine all features
                combined_signal = np.hstack(
                    [
                        power_spectrum,  # 128-point FFT (128, 1)
                        avg_power,  # Average power (128, 1)
                        std_dev_power,  # Std. dev of power (128, 1)
                        peak_power,  # Peak power (128, 1)
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

        # Save processed data for future use
        with open(self.data_pickle_path, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(f"Prepared data saved to {self.data_pickle_path}")

        return X_train, X_test, y_train, y_test

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
                lambda epoch: self.cyclical_lr(epoch, step_size=clr_step_size)
            )
            callbacks.append(clr_scheduler)

        stats_interval = 20
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

    def cyclical_lr(self, epoch, base_lr=1e-7, max_lr=1e-6, step_size=10):
        cycle = np.floor(1 + epoch / (2 * step_size))
        x = np.abs(epoch / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * max(0, (1 - x))
        print(f"Learning rate for epoch {epoch+1}: {lr}")
        return lr


def main(model_name):
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


if __name__ == "__main__":
    main(model_name="ConvLSTM_FFT_Power_SNR")
