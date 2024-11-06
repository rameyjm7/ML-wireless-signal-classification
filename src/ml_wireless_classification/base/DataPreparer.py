import numpy as np
import pickle
import json
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from ml_wireless_classification.base.BaseModulationClassifier import BaseModulationClassifier
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ml_wireless_classification.base.SignalUtils import (
    compute_fft_features,
    compute_instantaneous_features,
    compute_skewness,
    compute_spectral_energy_concentration,
    compute_zero_crossing_rate,
    compute_instantaneous_frequency_jitter,
    compute_spectral_kurtosis,
    compute_higher_order_cumulants,
    compute_spectral_flatness,
    compute_instantaneous_envelope_mean,
    compute_variance_of_phase,
    compute_crest_factor,
    compute_spectral_entropy,
    compute_energy_spread,
    compute_autocorrelation_decay,
    compute_rms_of_instantaneous_frequency,
    compute_entropy_of_instantaneous_frequency,
    compute_spectral_asymmetry
)


class DataPreparer(BaseModulationClassifier):
    def __init__(  self, feature_list, snr=True, data_path = None, model_path="saved_model.h5", stats_path="model_stats.json",
        ):
            super().__init__(
                data_path, model_path, stats_path
            )  # Call the base class constructor
            self.learning_rate = 0.0001  # Default learning rate
                
            self.feature_list = feature_list
            self.snr = snr
            self.feature_functions = {
                "iq_signal": self.extract_iq_signal,
                "envelope_variance": compute_instantaneous_envelope_mean,
                # "instantaneous_features" : compute_instantaneous_features,
                # "instantaneous_frequency_variance": compute_variance_of_phase,
                # "phase_jitter": compute_instantaneous_frequency_jitter,
                # "papr": compute_crest_factor,
                "spectral_kurtosis": compute_spectral_kurtosis,
                # "fourth_order_cumulant": compute_higher_order_cumulants,
                # "fft_features": compute_fft_features,
                # "zero_crossing_rate": compute_zero_crossing_rate,
                # "skewness": compute_skewness,
                # "spectral_entropy": compute_spectral_entropy,
                # "spectral_flatness": compute_spectral_flatness,
                # "frequency_spread": compute_energy_spread,
                # "autocorrelation_decay": compute_autocorrelation_decay,
                # "rms_instantaneous_frequency": compute_rms_of_instantaneous_frequency,
                # "entropy_of_instantaneous_frequency": compute_entropy_of_instantaneous_frequency,
                # "spectral_asymmetry": compute_spectral_asymmetry,
                # "spectral_energy_concentration": compute_spectral_energy_concentration
            }
            
    def prepare_data(self, pickle_path="prepared_data.pkl", feature_path="features.json"):
        X, y = [], []

        for (mod_type, snr), signals in self.data.items():
            for signal in signals:
                feature_values = []
                for feature_name in self.feature_list:
                    if feature_name in self.feature_functions:
                        feature_func = self.feature_functions[feature_name]
                        try:
                            # Handle special arguments for specific features
                            if feature_name == "iq_signal":
                                feature_array = feature_func(signal, snr)
                            elif feature_func.__code__.co_argcount == 1:
                                feature_array = feature_func(signal)
                            elif feature_func.__code__.co_argcount == 2:
                                feature_array = feature_func(signal, snr)
                            else:
                                raise ValueError(f"Unexpected number of arguments for {feature_name}")

                            # Check if the feature result is scalar and convert it if needed
                            if np.isscalar(feature_array):
                                feature_array = np.full((128, 1), feature_array)
                            elif feature_array.ndim == 1:
                                feature_array = feature_array[:, np.newaxis]  # Convert to (128, 1) if 1D

                            # Debug output to confirm shape
                            print(f"Feature '{feature_name}' shape: {feature_array.shape}")

                            feature_values.append(feature_array)

                        except Exception as e:
                            # Log feature extraction issue and use a placeholder
                            print(f"Warning: Issue with feature '{feature_name}' for mod_type '{mod_type}' with SNR {snr}. Error: {e}")
                            feature_values.append(np.zeros((128, 1)))  # Default placeholder array

                # Concatenate all valid features or log an error
                if feature_values:
                    try:
                        combined_features = np.hstack(feature_values)
                        X.append(combined_features)
                        y.append(mod_type)
                    except ValueError as e:
                        print(f"Error during hstack for mod_type '{mod_type}' with SNR {snr}: {e}")
                else:
                    print(f"Error: No valid features extracted for mod_type '{mod_type}' with SNR {snr}")

        # Convert to numpy arrays and handle label encoding
        X = np.array(X)
        y = np.array(y)

        if len(X) == 0 or len(y) == 0:
            raise ValueError("No valid data found after processing all signals.")

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Save data and features list
        with open(pickle_path, "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test, self.label_encoder), f)
        with open(feature_path, "w") as f:
            json.dump({"features": self.feature_list}, f)

        return X_train, X_test, y_train, y_test


    def extract_iq_signal(self, signal, snr):
        real_signal, imag_signal = signal[0], signal[1]
        max_real, max_imag = np.max(np.abs(real_signal)), np.max(np.abs(imag_signal))
        real_signal = real_signal / max_real if max_real != 0 else real_signal
        imag_signal = imag_signal / max_imag if max_imag != 0 else imag_signal
        iq_array = np.vstack([real_signal, imag_signal]).T
        if self.snr:
            snr_signal = np.full((128, 1), snr)
            return np.hstack([iq_array, snr_signal])
        return iq_array

    def features_test(self):
        feature_accuracies = {}
        for feature in self.feature_functions.keys():
            if feature != "iq_signal":
                print(f"Feature: {feature} preparing data")
                self.feature_list = ["iq_signal", feature]
                X_train, X_test, y_train, y_test = self.prepare_data()

                # Train and test model
                print(f"Feature: {feature} preparing build and training model")
                model, accuracy = self._build_and_train_model(X_train, y_train, X_test, y_test)
                feature_accuracies[feature] = accuracy
                print(f"Feature: {feature}, Accuracy: {accuracy:.4f}")

        best_feature = max(feature_accuracies, key=feature_accuracies.get)
        print(f"Best feature: {best_feature} with accuracy: {feature_accuracies[best_feature]:.4f}")
        return feature_accuracies
    
    def _build_and_train_model(self, X_train, y_train, X_test, y_test):
        # Reshape the input data to 2D (flatten the last two dimensions)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        num_classes = len(np.unique(y_train))
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=1)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
        
        return model, accuracy


    def find_best_feature_per_class(self, target_class):
        # Ensure label_encoder is initialized
        self.feature_list = ["iq_signal"]
        if not hasattr(self, 'label_encoder') or self.label_encoder is None:
            _, _, _, y = self.prepare_data()  # Call prepare_data to initialize label_encoder
        
        # Get the target class index
        try:
            target_class_index = self.label_encoder.transform([target_class])[0]
        except ValueError:
            raise ValueError(f"Target class '{target_class}' not found in label encoder classes.")
        
        best_accuracy = 0
        best_feature = None
        
        for feature in self.feature_functions.keys():
            if feature != "iq_signal":
                print(f"Testing feature: {feature} for class: {target_class}")
                self.feature_list = ["iq_signal", feature]
                X_train, X_test, y_train, y_test = self.prepare_data()
                
                # Filter the training and test data for the target class
                target_train_indices = (y_train == target_class_index)
                target_test_indices = (y_test == target_class_index)
                
                X_train_class = X_train[target_train_indices]
                y_train_class = y_train[target_train_indices]
                X_test_class = X_test[target_test_indices]
                y_test_class = y_test[target_test_indices]
                
                # Train and evaluate model
                model, accuracy = self._build_and_train_model(X_train_class, y_train_class, X_test_class, y_test_class)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_feature = feature

                print(f"Feature: {feature}, Class-specific Accuracy: {accuracy:.4f}")

        print(f"Best feature for class '{target_class}': {best_feature} with accuracy: {best_accuracy:.4f}")
        return best_feature, best_accuracy


if __name__ == "__main__":
        # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(
        script_dir, "..", "..", "..", "RML2016.10a_dict.pkl"
    )  # One level up from the script's directory
    
    data_preparer = DataPreparer(feature_list=[],
                                 data_path=data_path)
    # feature_accuracies = data_preparer.features_test()
    
    # Specify the target class (e.g., 'AM-SSB')
    target_class = "AM-SSB"
    best_features = data_preparer.find_best_feature_per_class(target_class)