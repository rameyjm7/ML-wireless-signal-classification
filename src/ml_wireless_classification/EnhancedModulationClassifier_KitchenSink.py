
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import threading

from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier
from ml_wireless_classification.base.SignalUtils import (
    compute_fft_features, compute_instantaneous_features, compute_kurtosis,
    compute_skewness, compute_spectral_energy_concentration, compute_zero_crossing_rate,
    compute_instantaneous_frequency_jitter, compute_spectral_kurtosis, compute_higher_order_cumulants,
    compute_spectral_flatness, compute_instantaneous_envelope_mean, compute_variance_of_phase,
    compute_crest_factor, compute_spectral_entropy, compute_energy_spread, compute_autocorrelation_decay,
    compute_rms_of_instantaneous_frequency, compute_entropy_of_instantaneous_frequency,
    compute_spectral_asymmetry, compute_envelope_variance, compute_papr, compute_modulation_index
)
from ml_wireless_classification.base.CommonVars import common_vars




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


class ModulationLSTMClassifier(BaseModulationClassifier):
    def __init__(self, data_path, model_path="saved_model.h5", stats_path="model_stats.json"):
        super().__init__(data_path, model_path, stats_path)
        self.learning_rate = 0.0001  # Default learning rate

    def prepare_data(self):
        X, y = [], []
        data_pickle_path = os.path.join("data_prepared.pkl")

        # if os.path.exists(data_pickle_path):
        #     with open(data_pickle_path, "rb") as f:
        #         X_train, X_test, y_train, y_test = pickle.load(f)
                
        #     # Encode labels and split data
        #     self.label_encoder = LabelEncoder()
        #     y_encoded = self.label_encoder.fit_transform(y)
        #     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        #     return X_train, X_test, y_train, y_test

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                real_signal, imag_signal = signal[0], signal[1]
                iq_signal = real_signal + 1j * imag_signal

                # Normalization
                real_signal = real_signal / (np.max(np.abs(real_signal)) or 1)
                imag_signal = imag_signal / (np.max(np.abs(imag_signal)) or 1)
                iq_array = np.vstack([real_signal, imag_signal]).T  # Shape: (128, 2)

                # Compute features
                features = [
                    np.full((128, 1), compute_envelope_variance(iq_signal)),
                    np.full((128, 1), compute_variance_of_phase(real_signal)),
                    np.full((128, 1), compute_instantaneous_frequency_jitter(np.diff(np.unwrap(np.angle(iq_signal))))),
                    np.full((128, 1), compute_papr(iq_signal)),
                    np.full((128, 1), compute_spectral_kurtosis(iq_signal).mean()),
                    np.full((128, 1), compute_skewness(real_signal)),
                    np.full((128, 1), compute_kurtosis(real_signal)),
                    np.full((128, 1), compute_zero_crossing_rate(real_signal)),
                    np.full((128, 1), compute_spectral_flatness(real_signal)),
                    np.full((128, 1), compute_instantaneous_envelope_mean(real_signal)),
                    np.full((128, 1), compute_crest_factor(real_signal)),
                    np.full((128, 1), compute_spectral_entropy(real_signal)),
                    np.full((128, 1), compute_rms_of_instantaneous_frequency(real_signal)),
                    np.full((128, 1), compute_entropy_of_instantaneous_frequency(real_signal)),
                    np.full((128, 1), compute_spectral_asymmetry(real_signal)),
                    np.full((128, 1), compute_modulation_index(iq_signal)),
                ]

                # FFT-based features
                center_frequency, peak_power, avg_power, std_dev_power = compute_fft_features(iq_signal)
                features.extend([
                    np.full((128, 1), center_frequency),
                    np.full((128, 1), peak_power),
                    np.full((128, 1), avg_power),
                    np.full((128, 1), std_dev_power),
                ])

                # Stack all features
                combined_features = np.hstack([iq_array] + features + [np.full((128, 1), snr)])
                X.append(combined_features)
                y.append(mod_type)

        self.X = np.array(X)
        y = np.array(y)

        # Encode labels and split data
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = self.split_data()

        with open(data_pickle_path, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)

        return X_train, X_test, y_train, y_test
    
    def split_data(self, random_state = 42):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_encoded, test_size=0.2, random_state=random_state)
        return X_train, X_test, y_train, y_test
        

    def build_model_alt(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
        else:
            self.model = Sequential([
                LSTM(128, input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(128, return_sequences=False),
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, activation="softmax"),
            ])
            self.model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(self.learning_rate), metrics=["accuracy"])

    def build_model(self, input_shape, num_classes):
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
            else:
                self.model = Sequential([
                    LSTM(128, input_shape=input_shape, return_sequences=True),
                    Dropout(0.2),
                    LSTM(128, return_sequences=True),  # Additional LSTM layer
                    Dropout(0.2),
                    LSTM(64, return_sequences=False),  # Reduced units in last LSTM for regularization
                    Dropout(0.2),
                    Dense(256, activation="relu"),  # Increased neurons in dense layer
                    Dropout(0.3),  # Slightly higher dropout rate to prevent overfitting
                    Dense(128, activation="relu"),
                    Dropout(0.3),
                    Dense(num_classes, activation="softmax"),
                ])
                self.model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(self.learning_rate), metrics=["accuracy"])


    def train_continuously(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size=64,
        use_clr=False,
        clr_step_size=10,
    ):
        try:
            epoch = 1
            while True:
                print(f"\nStarting epoch {epoch}")
                try:
                    self.train(
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        epochs=20,
                        batch_size=batch_size,
                        use_clr=use_clr,
                        clr_step_size=clr_step_size,
                    )
                    epoch += 1
                    print("Resplitting data")
                    X_train, X_test, y_train, y_test = self.split_data()
                    

                except Exception as e:
                    print(e)
                    pass

        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")
            self.evaluate(X_test, y_test)
            self.save_stats()



if __name__ == "__main__":
    # set the model name
    model_name = "EnhancedModulationClassifier_KitchenSink_v2"
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
